use std::{io, io::Write};

use uzu::engine::{Engine, EngineConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let cache_home = match std::env::var("XDG_CACHE_HOME") {
        Ok(cache_home) => cache_home,
        Err(_) => format!("{}/.cache", std::env::var("HOME")?),
    };
    let model_path = format!(
        "{cache_home}/huggingface/hub/models--trymirai--d3-flash-M/snapshots/6f81630ca912dec730bc916b459a8f289cb35150"
    );
    let model = engine.model(model_path).await?.ok_or("Model not found")?;
    // let model = engine.model("trymirai/d3-flash-M".to_string()).await?.ok_or("Model not found")?;

    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        print!("\r\u{001B}[2KDownload progress: {:.2}%", update.progress() * 100.0);
        io::stdout().flush()?;
    }
    println!();

    let session = engine.decision(model).await?;
    println!("Session created");

    Ok(())
}
