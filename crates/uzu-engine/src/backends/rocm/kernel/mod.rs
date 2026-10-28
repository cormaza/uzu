use crate::backends::{
    common::{Kernels, kernel::Unsupported},
    rocm::Rocm,
};

pub(crate) mod hip_ops;
mod matmul;
mod radix_top_k_small;

include!(concat!(env!("OUT_DIR"), "/rocm.rs"));

pub struct RocmKernels;

impl Kernels for RocmKernels {
    type Backend = Rocm;

    autogen_kernels!();
    type AttentionKernel = attention::AttentionRocmKernel;
    type DeltaNetChunkedPrefill = Unsupported<Rocm>;
    type DeltaNetTreeVerify = Unsupported<Rocm>;
    type MatmulKernel = matmul::MatmulRocmKernel;
    type RadixTopKSmall = radix_top_k_small::RocmRadixTopKSmall;
}
