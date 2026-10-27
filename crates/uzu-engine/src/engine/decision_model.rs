use std::{
    fs::File,
    io::{self, BufReader},
    path::Path,
    sync::Arc,
};

use derive_more::Debug;
use thiserror::Error;
use tokenizers::Tokenizer;

pub use crate::config::decision::DecisionConfig;
use crate::{
    backends::common::Backend,
    config::model::language_model::LanguageModelConfig,
    data_type::DataType,
    encodable_block::decoder::{Decoder, DecoderError},
    engine::Engine,
    safetensors::{ParameterLoader, ParameterLoaderError},
};

#[derive(Debug, Error)]
pub enum EngineLoadDecisionModelError<B: Backend> {
    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Model config has no decision note")]
    NotDecisionModel,
    #[error("ParameterLoader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Decoder error: {0}")]
    Decoder(#[from] DecoderError<B>),
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
}

pub struct DecisionModel<B: Backend> {
    context: Arc<B::Context>,
    // decoder: Decoder<B>,
    decision_config: DecisionConfig,
    tokenizer: Arc<Tokenizer>,
}

impl<B: Backend> DecisionModel<B> {
    pub fn decision_config(&self) -> &DecisionConfig {
        &self.decision_config
    }

    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }
}

impl<B: Backend> Engine<B> {
    pub fn load_decision_model(
        &self,
        model_path: &Path,
    ) -> Result<DecisionModel<B>, EngineLoadDecisionModelError<B>> {
        let context = self.context.clone();

        let config: LanguageModelConfig =
            serde_json::from_reader(BufReader::new(File::open(model_path.join("config.json"))?))?;
        if config.decision.is_none() {
            return Err(EngineLoadDecisionModelError::NotDecisionModel);
        }
        let decision_config: DecisionConfig =
            serde_json::from_reader(BufReader::new(File::open(model_path.join("decision_config.json"))?))?;

        // let data_type = DataType::BF16;
        // let weights_file = File::open(model_path.join("model.safetensors"))?;
        // let weight_loader = ParameterLoader::new(&weights_file, context.as_ref())?;
        // let decoder = Decoder::new(
        //     data_type,
        //     &config.decoder_config,
        //     &weight_loader.tree().subtree("decoder"),
        //     context.as_ref(),
        // )?;
        // weight_loader.assert_all_tensors_read()?;

        let tokenizer = Arc::new(Tokenizer::from_file(model_path.join("tokenizer.json"))?);

        Ok(DecisionModel {
            context,
            // decoder,
            decision_config,
            tokenizer,
        })
    }
}
