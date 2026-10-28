use std::convert::Infallible;

use crate::backends::{
    common::{Backend, Buffer, SparseBuffer},
    rocm::Rocm,
};

#[derive(Debug)]
pub struct RocmSparseBuffer {
    never: Infallible,
}

impl Buffer for RocmSparseBuffer {
    type Backend = Rocm;

    fn size(&self) -> usize {
        match self.never {}
    }
}

impl SparseBuffer for RocmSparseBuffer {
    fn map(
        &mut self,
        _until: usize,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        match self.never {}
    }
}
