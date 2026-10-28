use std::{
    alloc::{Layout, alloc, dealloc, handle_alloc_error},
    os::raw::c_void,
    ptr::NonNull,
};

use crate::backends::{
    common::{Buffer, BufferCpuAccessible, GlobalBuffer},
    rocm::{Rocm, hip::HipRuntime},
};

#[derive(Debug)]
pub struct RocmBuffer {
    host_ptr: NonNull<c_void>,
    dev_ptr: NonNull<c_void>,
    size: usize,
    fallback_layout: Option<Layout>,
}

impl RocmBuffer {
    pub(in crate::backends::rocm) fn new(size: usize) -> Self {
        assert!(size > 0, "ROCm buffer size must be nonzero");
        if let Ok(rt) = HipRuntime::get()
            && let Ok((host_ptr, dev_ptr)) = rt.alloc_coherent(size)
        {
            unsafe {
                host_ptr.as_ptr().cast::<u8>().write_bytes(0, size);
            }
            return Self {
                host_ptr,
                dev_ptr,
                size,
                fallback_layout: None,
            };
        }
        let layout = Layout::from_size_align(size, 64).expect("invalid ROCm buffer layout");
        let ptr = NonNull::new(unsafe { alloc(layout) }).unwrap_or_else(|| handle_alloc_error(layout)).cast();
        Self {
            host_ptr: ptr,
            dev_ptr: ptr,
            size,
            fallback_layout: Some(layout),
        }
    }

    pub fn device_ptr(&self) -> NonNull<c_void> {
        self.dev_ptr
    }
}

unsafe impl Send for RocmBuffer {}
unsafe impl Sync for RocmBuffer {}

impl Drop for RocmBuffer {
    fn drop(&mut self) {
        if let Some(layout) = self.fallback_layout {
            unsafe { dealloc(self.host_ptr.as_ptr().cast(), layout) };
        } else if let Ok(rt) = HipRuntime::get() {
            unsafe { rt.free_coherent(self.host_ptr) };
        }
    }
}

impl Buffer for RocmBuffer {
    type Backend = Rocm;

    fn size(&self) -> usize {
        self.size
    }
}

impl BufferCpuAccessible for RocmBuffer {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.host_ptr
    }
}

impl GlobalBuffer for RocmBuffer {}
