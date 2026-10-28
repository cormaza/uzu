use std::convert::Infallible;

use crate::backends::{
    common::{
        Backend,
        allocator::{
            bump::BumpAllocation,
            pool::{PoolAllocation, PoolAllocator},
        },
    },
    rocm::{
        buffer::{dense::RocmBuffer, sparse::RocmSparseBuffer},
        command_buffer::{RocmCommandBuffer, RocmCommandBufferEncoding},
        context::RocmContext,
        error::RocmError,
        kernel::RocmKernels,
    },
};

#[derive(Debug, Clone)]
pub struct Rocm;

impl Backend for Rocm {
    type Context = RocmContext;
    type CommandBuffer = RocmCommandBuffer;
    type DebugGroupSpan<'a> = &'a mut RocmCommandBufferEncoding;
    type GlobalBuffer = RocmBuffer;
    type ConstantBuffer = BumpAllocation<Self::GlobalBuffer>;
    type ScratchBuffer = PoolAllocation<Self::GlobalBuffer>;
    type SparseBuffer = RocmSparseBuffer;
    type AllocationPool = PoolAllocator<Self::GlobalBuffer>;
    type CaptureSpan = Infallible;
    type Kernels = RocmKernels;
    type Error = RocmError;

    const NAME: &'static str = "rocm";
}
