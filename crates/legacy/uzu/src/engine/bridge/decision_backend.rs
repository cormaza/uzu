use std::{path::PathBuf, sync::Arc};

use shoji::traits::backend::{Error as BackendError, decision::Instance as DecisionInstance};
use uzu_engine::{
    backends::Backend,
    engine::{Engine, decision_model::DecisionModel},
};

pub struct UzuDecisionBackendInstance<B: Backend> {
    model: Arc<DecisionModel<B>>,
}

impl<B: Backend> UzuDecisionBackendInstance<B> {
    pub async fn new(model_path: String) -> Result<Self, BackendError> {
        let engine = Engine::<B>::new().await.map_err(|err| err.to_string())?;
        let model_path = PathBuf::from(model_path);
        let model = engine.load_decision_model(&model_path).map_err(|err| err.to_string())?;
        Ok(Self {
            model: Arc::new(model),
        })
    }
}

impl<B: Backend> DecisionInstance for UzuDecisionBackendInstance<B> {}
