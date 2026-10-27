use std::pin::Pin;

use crate::traits::backend::Error;

pub type Config = ();

pub trait Backend: Send + Sync {
    fn instance(
        &self,
        reference: String,
        config: Config,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn Instance>, Error>> + Send + '_>>;
}

pub trait Instance: Send + Sync {}
