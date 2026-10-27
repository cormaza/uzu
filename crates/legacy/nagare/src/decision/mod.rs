mod error;

use std::sync::Arc;

pub use error::DecisionSessionError;
use serde::{Deserialize, Serialize};
use shoji::{
    traits::{Backend, backend::decision::Instance},
    types::model::{Model, ModelSpecialization},
};
use tokio::sync::Mutex;

#[bindings::export(Enumeration)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecisionSessionState {
    Idle,
    Deciding,
}

struct InstanceHolder {
    instance: Box<dyn Instance>,
}

#[bindings::export(Class)]
#[derive(Clone)]
pub struct DecisionSession {
    holder: Arc<Mutex<InstanceHolder>>,
    state: Arc<Mutex<DecisionSessionState>>,
}

impl DecisionSession {
    pub async fn new(
        backend: Arc<dyn Backend>,
        model: Model,
        path: Option<String>,
    ) -> Result<Self, DecisionSessionError> {
        if !model.specializations.contains(&ModelSpecialization::Decision {}) {
            return Err(DecisionSessionError::UnsupportedModel {});
        }
        let reference = path.unwrap_or_else(|| model.identifier.clone());

        let holder = tokio::spawn(async move {
            let decision_backend = backend.as_decision_capable().ok_or(DecisionSessionError::UnsupportedModel {})?;
            let instance =
                decision_backend.instance(reference, ()).await.map_err(|error| DecisionSessionError::Backend {
                    message: error.to_string(),
                })?;

            Ok(InstanceHolder {
                instance,
            })
        })
        .await
        .map_err(|error| DecisionSessionError::Backend {
            message: error.to_string(),
        })??;

        Ok(Self {
            holder: Arc::new(Mutex::new(holder)),
            state: Arc::new(Mutex::new(DecisionSessionState::Idle)),
        })
    }
}

#[bindings::export(Implementation)]
impl DecisionSession {
    #[bindings::export(Method(Getter))]
    pub async fn state(&self) -> DecisionSessionState {
        *self.state.lock().await
    }
}
