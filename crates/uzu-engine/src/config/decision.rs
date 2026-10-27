use uzu_engine_macros::uzu_config;

#[uzu_config]
pub struct DecisionConfig {
    pub format_version: u32,
    pub format_id: String,
    pub prompt: String,
    pub base_model: String,
    pub revision: String,
    pub codes: Box<[String]>,
    pub token_ids: Box<[u32]>,
    pub temperature: f32,
    pub attention_mode: String,
    pub pooling: String,
    pub max_length: Option<u32>,
    pub readout_dtype: String,
}

#[uzu_config]
pub struct DecisionNote {
    note: String,
}
