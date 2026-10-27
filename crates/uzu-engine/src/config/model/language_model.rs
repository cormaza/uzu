use uzu_engine_macros::uzu_config;

use crate::config::{decision::DecisionNote, decoder::DecoderConfig, model::generation::GenerationConfig};

#[uzu_config(super::ModelConfig)]
pub struct LanguageModelConfig {
    pub decoder_config: DecoderConfig,
    pub generation_config: GenerationConfig,
    #[serde(default)]
    pub decision: Option<DecisionNote>,
}
