use thiserror::Error;

#[derive(Debug, Error)]
pub enum RocmError {
    #[error("Not supported")]
    NotSupported,
    #[error("No ROCm GPU device available")]
    NoDeviceAvailable,
    #[error("HIP runtime error in {0}: code {1}")]
    HipError(&'static str, i32),
    #[error("Failed to load HIP runtime library: {0}")]
    LibraryLoadFailed(String),
    #[error("Command buffer execution failed")]
    CommandBufferExecutionFailed,
}
