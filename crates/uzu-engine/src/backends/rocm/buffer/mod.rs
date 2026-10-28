use std::{os::raw::c_void, ptr::NonNull};

use crate::{
    backends::{
        common::{Backend, Buffer, BufferCpuAccessible},
        rocm::Rocm,
    },
    utils::downcast::downcast_ref,
};

pub mod constant;
pub mod dense;
pub mod scratch;
pub mod sparse;

pub(super) trait RocmBufferExt: Buffer<Backend = Rocm> {
    fn cpu_address(&self) -> NonNull<c_void> {
        if let Some(buffer) = downcast_ref::<<Rocm as Backend>::GlobalBuffer>(self) {
            buffer.cpu_ptr()
        } else if let Some(buffer) = downcast_ref::<<Rocm as Backend>::ConstantBuffer>(self) {
            buffer.cpu_ptr()
        } else if let Some(buffer) = downcast_ref::<<Rocm as Backend>::ScratchBuffer>(self) {
            buffer.cpu_ptr()
        } else {
            unreachable!("Unsupported Rocm buffer type")
        }
    }

    fn device_address(&self) -> NonNull<c_void> {
        if let Some(buffer) = downcast_ref::<<Rocm as Backend>::GlobalBuffer>(self) {
            buffer.device_ptr()
        } else if let Some(buffer) = downcast_ref::<<Rocm as Backend>::ConstantBuffer>(self) {
            unsafe { buffer.page().device_ptr().byte_add(buffer.range().start) }
        } else if let Some(buffer) = downcast_ref::<<Rocm as Backend>::ScratchBuffer>(self) {
            unsafe { buffer.page().device_ptr().byte_add(buffer.range().start) }
        } else {
            unreachable!("Unsupported Rocm buffer type")
        }
    }
}

impl<T: Buffer<Backend = Rocm> + ?Sized> RocmBufferExt for T {}
