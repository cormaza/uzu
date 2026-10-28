use std::{
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    ptr::NonNull,
    sync::OnceLock,
};

use crate::backends::rocm::error::RocmError;

pub type HipErrorT = c_int;
pub type HipStreamT = *mut c_void;
pub type HipModuleT = *mut c_void;
pub type HipFunctionT = *mut c_void;

pub const HIP_SUCCESS: HipErrorT = 0;
pub const HIP_MEMCPY_DEFAULT: c_uint = 4;
pub const HIP_HOST_MALLOC_MAPPED: c_uint = 0x02;
pub const HIP_HOST_MALLOC_COHERENT: c_uint = 0x40000000;

type HipInitFn = unsafe extern "C" fn(flags: c_uint) -> HipErrorT;
type HipGetDeviceCountFn = unsafe extern "C" fn(count: *mut c_int) -> HipErrorT;
type HipSetDeviceFn = unsafe extern "C" fn(device_id: c_int) -> HipErrorT;
type HipDeviceGetNameFn = unsafe extern "C" fn(name: *mut c_char, len: c_int, device: c_int) -> HipErrorT;
type HipStreamCreateFn = unsafe extern "C" fn(stream: *mut HipStreamT) -> HipErrorT;
type HipStreamDestroyFn = unsafe extern "C" fn(stream: HipStreamT) -> HipErrorT;
type HipStreamSynchronizeFn = unsafe extern "C" fn(stream: HipStreamT) -> HipErrorT;
type HipHostMallocFn = unsafe extern "C" fn(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> HipErrorT;
type HipHostFreeFn = unsafe extern "C" fn(ptr: *mut c_void) -> HipErrorT;
type HipHostGetDevicePointerFn =
    unsafe extern "C" fn(dev_ptr: *mut *mut c_void, hst_ptr: *mut c_void, flags: c_uint) -> HipErrorT;
type HipMemcpyAsyncFn = unsafe extern "C" fn(
    dst: *mut c_void,
    src: *const c_void,
    size_bytes: usize,
    kind: c_uint,
    stream: HipStreamT,
) -> HipErrorT;
type HipMemsetD8AsyncFn =
    unsafe extern "C" fn(dst: *mut c_void, value: u8, count: usize, stream: HipStreamT) -> HipErrorT;
type HipModuleLoadDataFn = unsafe extern "C" fn(module: *mut HipModuleT, image: *const c_void) -> HipErrorT;
type HipModuleUnloadFn = unsafe extern "C" fn(module: HipModuleT) -> HipErrorT;
type HipModuleGetFunctionFn =
    unsafe extern "C" fn(function: *mut HipFunctionT, module: HipModuleT, kname: *const c_char) -> HipErrorT;
type HipModuleLaunchKernelFn = unsafe extern "C" fn(
    f: HipFunctionT,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    stream: HipStreamT,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> HipErrorT;

pub struct HipRuntime {
    _handle: *mut c_void,
    pub hip_init: HipInitFn,
    pub hip_get_device_count: HipGetDeviceCountFn,
    pub hip_set_device: HipSetDeviceFn,
    pub hip_device_get_name: HipDeviceGetNameFn,
    pub hip_stream_create: HipStreamCreateFn,
    pub hip_stream_destroy: HipStreamDestroyFn,
    pub hip_stream_synchronize: HipStreamSynchronizeFn,
    pub hip_host_malloc: HipHostMallocFn,
    pub hip_host_free: HipHostFreeFn,
    pub hip_host_get_device_pointer: HipHostGetDevicePointerFn,
    pub hip_memcpy_async: HipMemcpyAsyncFn,
    pub hip_memset_d8_async: HipMemsetD8AsyncFn,
    pub hip_module_load_data: HipModuleLoadDataFn,
    pub hip_module_unload: HipModuleUnloadFn,
    pub hip_module_get_function: HipModuleGetFunctionFn,
    pub hip_module_launch_kernel: HipModuleLaunchKernelFn,
}

unsafe impl Send for HipRuntime {}
unsafe impl Sync for HipRuntime {}

static HIP_RUNTIME: OnceLock<Result<HipRuntime, String>> = OnceLock::new();

unsafe fn load_sym<T: Copy>(
    handle: *mut c_void,
    name: &'static [u8],
) -> Result<T, String> {
    let cstr = CStr::from_bytes_with_nul(name).unwrap();
    let sym = unsafe { libc::dlsym(handle, cstr.as_ptr()) };
    if sym.is_null() {
        return Err(format!("Missing symbol {}", cstr.to_string_lossy()));
    }
    Ok(unsafe { std::mem::transmute_copy(&sym) })
}

impl HipRuntime {
    pub fn get() -> Result<&'static HipRuntime, RocmError> {
        let res = HIP_RUNTIME.get_or_init(|| unsafe {
            let candidates = [
                c"/opt/rocm/lib/libamdhip64.so",
                c"libamdhip64.so.7",
                c"libamdhip64.so.6",
                c"libamdhip64.so",
            ];
            let mut handle = std::ptr::null_mut();
            for candidate in candidates {
                handle = libc::dlopen(candidate.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
                if !handle.is_null() {
                    break;
                }
            }
            if handle.is_null() {
                return Err("Could not dlopen libamdhip64.so".to_string());
            }

            Ok(HipRuntime {
                _handle: handle,
                hip_init: load_sym(handle, b"hipInit\0")?,
                hip_get_device_count: load_sym(handle, b"hipGetDeviceCount\0")?,
                hip_set_device: load_sym(handle, b"hipSetDevice\0")?,
                hip_device_get_name: load_sym(handle, b"hipDeviceGetName\0")?,
                hip_stream_create: load_sym(handle, b"hipStreamCreate\0")?,
                hip_stream_destroy: load_sym(handle, b"hipStreamDestroy\0")?,
                hip_stream_synchronize: load_sym(handle, b"hipStreamSynchronize\0")?,
                hip_host_malloc: load_sym(handle, b"hipHostMalloc\0")?,
                hip_host_free: load_sym(handle, b"hipHostFree\0")?,
                hip_host_get_device_pointer: load_sym(handle, b"hipHostGetDevicePointer\0")?,
                hip_memcpy_async: load_sym(handle, b"hipMemcpyAsync\0")?,
                hip_memset_d8_async: load_sym(handle, b"hipMemsetD8Async\0")?,
                hip_module_load_data: load_sym(handle, b"hipModuleLoadData\0")?,
                hip_module_unload: load_sym(handle, b"hipModuleUnload\0")?,
                hip_module_get_function: load_sym(handle, b"hipModuleGetFunction\0")?,
                hip_module_launch_kernel: load_sym(handle, b"hipModuleLaunchKernel\0")?,
            })
        });
        match res {
            Ok(rt) => Ok(rt),
            Err(msg) => Err(RocmError::LibraryLoadFailed(msg.clone())),
        }
    }

    pub fn init_device(
        &self,
        device_id: i32,
    ) -> Result<String, RocmError> {
        unsafe {
            let err = (self.hip_init)(0);
            if err != HIP_SUCCESS {
                return Err(RocmError::HipError("hipInit", err));
            }
            let mut count: c_int = 0;
            let err = (self.hip_get_device_count)(&mut count);
            if err != HIP_SUCCESS {
                return Err(RocmError::HipError("hipGetDeviceCount", err));
            }
            if count <= device_id {
                return Err(RocmError::NoDeviceAvailable);
            }
            let err = (self.hip_set_device)(device_id);
            if err != HIP_SUCCESS {
                return Err(RocmError::HipError("hipSetDevice", err));
            }
            let mut name_buf = [0 as c_char; 256];
            let err = (self.hip_device_get_name)(name_buf.as_mut_ptr(), 256, device_id);
            if err != HIP_SUCCESS {
                return Err(RocmError::HipError("hipDeviceGetName", err));
            }
            let name = CStr::from_ptr(name_buf.as_ptr()).to_string_lossy().into_owned();
            Ok(name)
        }
    }

    pub fn alloc_coherent(
        &self,
        size: usize,
    ) -> Result<(NonNull<c_void>, NonNull<c_void>), RocmError> {
        let padded_size = size.max(64).next_multiple_of(64);
        unsafe {
            let mut host_ptr: *mut c_void = std::ptr::null_mut();
            let mut err =
                (self.hip_host_malloc)(&mut host_ptr, padded_size, HIP_HOST_MALLOC_MAPPED | HIP_HOST_MALLOC_COHERENT);
            if err != HIP_SUCCESS || host_ptr.is_null() {
                err = (self.hip_host_malloc)(&mut host_ptr, padded_size, HIP_HOST_MALLOC_MAPPED);
            }
            if err != HIP_SUCCESS || host_ptr.is_null() {
                return Err(RocmError::HipError("hipHostMalloc", err));
            }
            let mut dev_ptr: *mut c_void = std::ptr::null_mut();
            let err = (self.hip_host_get_device_pointer)(&mut dev_ptr, host_ptr, 0);
            let dev_ptr = if err == HIP_SUCCESS && !dev_ptr.is_null() {
                dev_ptr
            } else {
                host_ptr
            };
            Ok((
                NonNull::new(host_ptr).expect("non-null host_ptr"),
                NonNull::new(dev_ptr).expect("non-null dev_ptr"),
            ))
        }
    }

    pub unsafe fn free_coherent(
        &self,
        host_ptr: NonNull<c_void>,
    ) {
        let _ = unsafe { (self.hip_host_free)(host_ptr.as_ptr()) };
    }
}

#[derive(Debug)]
pub struct HipStream {
    raw: HipStreamT,
}

unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

impl HipStream {
    pub fn new() -> Result<Self, RocmError> {
        let rt = HipRuntime::get()?;
        let mut raw: HipStreamT = std::ptr::null_mut();
        let err = unsafe { (rt.hip_stream_create)(&mut raw) };
        if err != HIP_SUCCESS {
            return Err(RocmError::HipError("hipStreamCreate", err));
        }
        Ok(Self {
            raw,
        })
    }

    pub fn synchronize(&self) -> Result<(), RocmError> {
        let rt = HipRuntime::get()?;
        let err = unsafe { (rt.hip_stream_synchronize)(self.raw) };
        if err != HIP_SUCCESS {
            return Err(RocmError::HipError("hipStreamSynchronize", err));
        }
        Ok(())
    }

    pub unsafe fn memcpy_async(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: usize,
    ) -> Result<(), RocmError> {
        if size_bytes == 0 {
            return Ok(());
        }
        let rt = HipRuntime::get()?;
        let err = unsafe { (rt.hip_memcpy_async)(dst, src, size_bytes, HIP_MEMCPY_DEFAULT, self.raw) };
        if err != HIP_SUCCESS {
            return Err(RocmError::HipError("hipMemcpyAsync", err));
        }
        Ok(())
    }

    pub unsafe fn memset_d8_async(
        &self,
        dst: *mut c_void,
        value: u8,
        count: usize,
    ) -> Result<(), RocmError> {
        if count == 0 {
            return Ok(());
        }
        let rt = HipRuntime::get()?;
        let err = unsafe { (rt.hip_memset_d8_async)(dst, value, count, self.raw) };
        if err != HIP_SUCCESS {
            return Err(RocmError::HipError("hipMemsetD8Async", err));
        }
        Ok(())
    }

    pub unsafe fn launch_kernel(
        &self,
        func: HipFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), RocmError> {
        if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
            return Ok(());
        }
        let rt = HipRuntime::get()?;
        let err = unsafe {
            (rt.hip_module_launch_kernel)(
                func.0,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_mem_bytes,
                self.raw,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if err != HIP_SUCCESS {
            return Err(RocmError::HipError("hipModuleLaunchKernel", err));
        }
        Ok(())
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if let Ok(rt) = HipRuntime::get() {
            let _ = unsafe { (rt.hip_stream_destroy)(self.raw) };
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HipFunction(pub HipFunctionT);

unsafe impl Send for HipFunction {}
unsafe impl Sync for HipFunction {}

#[derive(Debug)]
pub struct HipModule {
    raw: HipModuleT,
}

unsafe impl Send for HipModule {}
unsafe impl Sync for HipModule {}

impl HipModule {
    pub fn load_data(hsaco_bytes: &[u8]) -> Result<Self, RocmError> {
        let rt = HipRuntime::get()?;
        let mut raw: HipModuleT = std::ptr::null_mut();
        let err = unsafe { (rt.hip_module_load_data)(&mut raw, hsaco_bytes.as_ptr().cast()) };
        if err != HIP_SUCCESS {
            return Err(RocmError::HipError("hipModuleLoadData", err));
        }
        Ok(Self {
            raw,
        })
    }

    pub fn get_function(
        &self,
        name: &str,
    ) -> Result<HipFunction, RocmError> {
        let rt = HipRuntime::get()?;
        let cname = CString::new(name).unwrap();
        let mut func: HipFunctionT = std::ptr::null_mut();
        let err = unsafe { (rt.hip_module_get_function)(&mut func, self.raw, cname.as_ptr()) };
        if err != HIP_SUCCESS {
            return Err(RocmError::HipError("hipModuleGetFunction", err));
        }
        Ok(HipFunction(func))
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        if let Ok(rt) = HipRuntime::get() {
            let _ = unsafe { (rt.hip_module_unload)(self.raw) };
        }
    }
}
