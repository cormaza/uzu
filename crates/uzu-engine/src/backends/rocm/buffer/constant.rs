use std::{os::raw::c_void, ptr::NonNull};

use crate::backends::{
    common::{Backend, Buffer, BufferCpuAccessible, ConstantBuffer, allocator::bump::BumpAllocation},
    rocm::Rocm,
};

impl Buffer for BumpAllocation<<Rocm as Backend>::GlobalBuffer> {
    type Backend = Rocm;

    fn size(&self) -> usize {
        self.range().iter().len()
    }
}

impl BufferCpuAccessible for BumpAllocation<<Rocm as Backend>::GlobalBuffer> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { self.page().cpu_ptr().byte_add(self.range().start) }
    }
}

impl ConstantBuffer for BumpAllocation<<Rocm as Backend>::GlobalBuffer> {}
