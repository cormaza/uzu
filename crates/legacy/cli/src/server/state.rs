use std::sync::Arc;

use tokio::sync::Mutex;
use uzu::{engine::Engine, session::chat::ChatSession};

use crate::common::thinking::ThinkingSupport;

pub struct ActiveModelSession {
    pub model_name: String,
    pub session: Arc<Mutex<ChatSession>>,
    pub thinking_support: ThinkingSupport,
}

pub struct ServerState {
    pub engine: Engine,
    pub active: Arc<Mutex<ActiveModelSession>>,
    pub prefix_cache: bool,
}
