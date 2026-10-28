use rocket::{State, get, serde::json::Json};
use serde::{Deserialize, Serialize};

use crate::server::{ServerState, request_info::RequestInfo};

#[derive(Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Serialize, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<Model>,
}

#[get("/models")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_models(
    state: &State<ServerState>,
    request_info: &RequestInfo,
) -> Json<ModelsResponse> {
    let active_model = state.active.lock().await.model_name.clone();
    let mut data = vec![Model {
        id: active_model.clone(),
        object: "model".to_string(),
        created: 0,
        owned_by: "uzu".to_string(),
    }];
    if let Ok(catalog) = state.engine.models().await {
        for model in catalog {
            if model.identifier != active_model && model.is_chat_capable() {
                data.push(Model {
                    id: model.identifier,
                    object: "model".to_string(),
                    created: 0,
                    owned_by: "uzu".to_string(),
                });
            }
        }
    }
    Json(ModelsResponse {
        object: "list".to_string(),
        data,
    })
}
