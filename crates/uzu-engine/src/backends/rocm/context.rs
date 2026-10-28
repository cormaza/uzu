use std::{
    convert::Infallible,
    path::Path,
    sync::{Arc, Weak, mpsc},
    thread,
};

use crate::backends::{
    common::{
        Backend, Context, DeviceCapabilities,
        allocator::{bump::BumpAllocator, pool::PoolAllocator},
    },
    rocm::{
        Rocm,
        buffer::dense::RocmBuffer,
        command_buffer::RocmCommandBufferEncoding,
        error::RocmError,
        hip::{HipFunction, HipModule, HipRuntime, HipStream},
    },
};

static ROCM_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rocm_kernels.hsaco"));

pub struct RocmHipKernels {
    _module: HipModule,
    pub matmul_fp: HipFunction,
    pub matmul_quant: HipFunction,
    pub matmul_trellis: HipFunction,
    pub activation: HipFunction,
    pub tensor_add_bias: HipFunction,
    pub tensor_add_scale: HipFunction,
    pub logit_transform: HipFunction,
    pub sigmoid_gate: HipFunction,
    pub pooling_mean: HipFunction,
    pub softmax: HipFunction,
    pub normalization: HipFunction,
    pub qkv_norm: HipFunction,
    pub kv_cache_update: HipFunction,
    pub gated_act_mul_fp: HipFunction,
    pub attention_single_pass: HipFunction,
    pub input_embedding_lookup_dense: HipFunction,
}

impl RocmHipKernels {
    pub fn load() -> Result<Self, RocmError> {
        let module = HipModule::load_data(ROCM_HSACO)?;
        Ok(Self {
            matmul_fp: module.get_function("rocm_matmul_fp")?,
            matmul_quant: module.get_function("rocm_matmul_quant")?,
            matmul_trellis: module.get_function("rocm_matmul_trellis")?,
            activation: module.get_function("rocm_activation")?,
            tensor_add_bias: module.get_function("rocm_tensor_add_bias")?,
            tensor_add_scale: module.get_function("rocm_tensor_add_scale")?,
            logit_transform: module.get_function("rocm_logit_transform")?,
            sigmoid_gate: module.get_function("rocm_sigmoid_gate")?,
            pooling_mean: module.get_function("rocm_pooling_mean")?,
            softmax: module.get_function("rocm_softmax")?,
            normalization: module.get_function("rocm_normalization")?,
            qkv_norm: module.get_function("rocm_qkv_norm")?,
            kv_cache_update: module.get_function("rocm_kv_cache_update")?,
            gated_act_mul_fp: module.get_function("rocm_gated_act_mul_fp")?,
            attention_single_pass: module.get_function("rocm_attention_single_pass")?,
            input_embedding_lookup_dense: module.get_function("rocm_input_embedding_lookup_dense")?,
            _module: module,
        })
    }
}

pub struct RocmContext {
    pub(super) command_queue: mpsc::Sender<Box<dyn FnOnce() + Send>>,
    pub(crate) stream: HipStream,
    pub(crate) kernels: RocmHipKernels,
    device_name: String,
    weak_self: Weak<RocmContext>,
}

impl Context for RocmContext {
    type Backend = Rocm;

    async fn new() -> Result<Arc<Self>, RocmError> {
        let rt = HipRuntime::get()?;
        let device_name = rt.init_device(0)?;
        let stream = HipStream::new()?;
        let kernels = RocmHipKernels::load()?;

        let (command_queue_sender, command_queue_receiver) = mpsc::channel::<Box<dyn FnOnce() + Send>>();

        thread::spawn(move || {
            if let Ok(rt) = HipRuntime::get() {
                let _ = unsafe { (rt.hip_set_device)(0) };
            }
            for command_buffer in command_queue_receiver {
                command_buffer();
            }
        });

        Ok(Arc::new_cyclic(|weak_self| RocmContext {
            command_queue: command_queue_sender,
            stream,
            kernels,
            device_name,
            weak_self: weak_self.clone(),
        }))
    }

    fn device_name(&self) -> Option<&str> {
        Some(&self.device_name)
    }

    fn create_command_buffer(
        &self,
        _name: Option<&str>,
        allocation_pool: Option<Arc<<Rocm as Backend>::AllocationPool>>,
    ) -> Result<RocmCommandBufferEncoding, RocmError> {
        let constant_allocator = BumpAllocator::new(256 * 1024);
        let allocation_pool = allocation_pool.unwrap_or_else(|| self.create_allocation_pool());
        Ok(RocmCommandBufferEncoding::new(constant_allocator, allocation_pool, self.weak_self.upgrade().unwrap()))
    }

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<<Rocm as Backend>::GlobalBuffer, RocmError> {
        Ok(RocmBuffer::new(size))
    }

    fn create_sparse_buffer(
        &self,
        _capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error> {
        Err(RocmError::NotSupported)
    }

    fn create_allocation_pool(&self) -> Arc<<Rocm as Backend>::AllocationPool> {
        PoolAllocator::new()
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        None
    }

    fn start_capture(
        &self,
        _output_path: &Path,
    ) -> Result<Infallible, RocmError> {
        Err(RocmError::NotSupported)
    }

    fn device_capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::empty()
    }
}
