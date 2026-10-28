use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use futures::channel::oneshot;

use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding,
            CommandBufferExecutable, CommandBufferPending, TimestampSpan, allocator::bump::BumpAllocator,
        },
        rocm::{
            Rocm,
            buffer::{RocmBufferExt, dense::RocmBuffer},
            context::RocmContext,
            error::RocmError,
            hip::HipStream,
        },
    },
    utils::pointers::{SendPtr, SendPtrMut},
};

pub struct RocmCommandBuffer;

impl CommandBuffer for RocmCommandBuffer {
    type Backend = Rocm;

    type Encoding = RocmCommandBufferEncoding;
    type Executable = RocmCommandBufferExecutable;
    type Pending = RocmCommandBufferPending;
    type Completed = RocmCommandBufferCompleted;
}

enum RocmRecordedCommand {
    Gpu(Box<dyn FnOnce(&HipStream) + Send>),
    Host(Box<dyn FnOnce() + Send>),
}

pub struct RocmCommandBufferEncoding {
    commands: Vec<RocmRecordedCommand>,
    constant_allocator: BumpAllocator<<Rocm as Backend>::GlobalBuffer>,
    allocation_pool: Arc<<Rocm as Backend>::AllocationPool>,
    context: Arc<RocmContext>,
}

impl RocmCommandBufferEncoding {
    pub fn new(
        constant_allocator: BumpAllocator<<Rocm as Backend>::GlobalBuffer>,
        allocation_pool: Arc<<Rocm as Backend>::AllocationPool>,
        context: Arc<RocmContext>,
    ) -> RocmCommandBufferEncoding {
        RocmCommandBufferEncoding {
            commands: Vec::new(),
            constant_allocator,
            allocation_pool,
            context,
        }
    }

    pub fn push_command(
        &mut self,
        command: impl FnOnce() + Send + 'static,
    ) {
        self.commands.push(RocmRecordedCommand::Host(Box::new(command)))
    }

    pub fn push_gpu_command(
        &mut self,
        command: impl FnOnce(&HipStream) + Send + 'static,
    ) {
        self.commands.push(RocmRecordedCommand::Gpu(Box::new(command)))
    }
}

impl CommandBufferEncoding for RocmCommandBufferEncoding {
    type CommandBuffer = RocmCommandBuffer;

    fn context(&self) -> &RocmContext {
        &self.context
    }

    fn allocate_constant(
        &mut self,
        size: usize,
    ) -> Result<<Rocm as Backend>::ConstantBuffer, RocmError> {
        self.constant_allocator.allocate(size, |size| Ok(RocmBuffer::new(size)))
    }

    fn allocate_scratch(
        &mut self,
        size: usize,
    ) -> Result<<Rocm as Backend>::ScratchBuffer, RocmError> {
        self.allocation_pool.allocate(size, |size| Ok(RocmBuffer::new(size)))
    }

    fn encode_copy(
        &mut self,
        src: impl BufferRef<Backend = Rocm>,
        dst: impl BufferMut<Backend = Rocm>,
    ) {
        let (src, src_range) = src.parts();
        let (dst, dst_range) = dst.parts();
        let size = src_range.iter().len();
        assert_eq!(size, dst_range.iter().len());

        let src_ptr = SendPtr(unsafe { src.device_address().as_ptr().cast::<u8>().add(src_range.start) });
        let dst_ptr = SendPtrMut(unsafe { dst.device_address().as_ptr().cast::<u8>().add(dst_range.start) });

        self.push_gpu_command(move |stream| unsafe {
            stream
                .memcpy_async(dst_ptr.as_ptr().cast(), src_ptr.as_ptr().cast(), size)
                .expect("HIP async memcpy failed");
        });
    }

    fn encode_fill(
        &mut self,
        dst: impl BufferMut<Backend = Rocm>,
        value: u8,
    ) {
        let (dst, range) = dst.parts();
        let size = range.iter().len();
        let dst_ptr = SendPtrMut(unsafe { dst.device_address().as_ptr().cast::<u8>().add(range.start) });
        self.push_gpu_command(move |stream| unsafe {
            stream.memset_d8_async(dst_ptr.as_ptr().cast(), value, size).expect("HIP async memset failed");
        });
    }

    fn debug_group(
        &mut self,
        _name: &str,
    ) -> <Rocm as Backend>::DebugGroupSpan<'_> {
        self
    }

    fn debug_point(
        &mut self,
        _name: &str,
    ) {
    }

    fn with_timestamps(self) -> Self {
        panic!("the ROCm backend does not record timestamps")
    }

    fn end_encoding(self) -> RocmCommandBufferExecutable {
        assert!(self.constant_allocator.is_done(), "attempted to end encoding while constants are still alive");
        RocmCommandBufferExecutable {
            commands: self.commands,
            constant_allocator: self.constant_allocator,
            allocation_pool: self.allocation_pool,
            context: self.context,
        }
    }
}

pub struct RocmCommandBufferExecutable {
    commands: Vec<RocmRecordedCommand>,
    constant_allocator: BumpAllocator<<Rocm as Backend>::GlobalBuffer>,
    allocation_pool: Arc<<Rocm as Backend>::AllocationPool>,
    context: Arc<RocmContext>,
}

impl CommandBufferExecutable for RocmCommandBufferExecutable {
    type CommandBuffer = RocmCommandBuffer;

    fn submit(self) -> RocmCommandBufferPending {
        let (return_sender, return_receiver) = oneshot::channel();
        let allocation_pool = self.allocation_pool.clone();
        let context = self.context.clone();

        self.context
            .command_queue
            .send(Box::new(move || {
                let start = Instant::now();
                let mut gpu_dirty = false;

                for command in self.commands {
                    match command {
                        RocmRecordedCommand::Gpu(cmd) => {
                            cmd(&context.stream);
                            gpu_dirty = true;
                        },
                        RocmRecordedCommand::Host(cmd) => {
                            if gpu_dirty {
                                context.stream.synchronize().expect("HIP stream synchronize failed");
                                gpu_dirty = false;
                            }
                            cmd();
                        },
                    }
                }

                if gpu_dirty {
                    context.stream.synchronize().expect("HIP stream synchronize failed");
                }

                let gpu_execution_time = start.elapsed();

                let completed = RocmCommandBufferCompleted {
                    gpu_execution_time,
                    _allocation_pool: self.allocation_pool,
                };

                let _ = return_sender.send(completed);
                drop(self.constant_allocator);
            }))
            .unwrap();

        RocmCommandBufferPending {
            return_receiver,
            _allocation_pool: allocation_pool,
        }
    }
}

pub struct RocmCommandBufferPending {
    return_receiver: oneshot::Receiver<RocmCommandBufferCompleted>,
    _allocation_pool: Arc<<Rocm as Backend>::AllocationPool>,
}

impl CommandBufferPending for RocmCommandBufferPending {
    type CommandBuffer = RocmCommandBuffer;

    async fn wait_until_completed(self) -> Result<RocmCommandBufferCompleted, RocmError> {
        self.return_receiver.await.map_err(|_| RocmError::CommandBufferExecutionFailed)
    }
}

pub struct RocmCommandBufferCompleted {
    gpu_execution_time: Duration,
    _allocation_pool: Arc<<Rocm as Backend>::AllocationPool>,
}

impl CommandBufferCompleted for RocmCommandBufferCompleted {
    type CommandBuffer = RocmCommandBuffer;

    fn gpu_execution_time(&self) -> Duration {
        self.gpu_execution_time
    }

    fn timestamps(&self) -> &[TimestampSpan] {
        &[]
    }
}
