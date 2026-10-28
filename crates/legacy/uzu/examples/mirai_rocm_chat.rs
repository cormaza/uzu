use std::{
    env,
    io::{self, BufRead, Write},
};

use tokio_stream::StreamExt;
use uzu::{
    engine::{Engine, EngineConfig},
    session::chat::ChatSessionStreamChunk,
    storage::DownloadPhase,
    types::session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
};
use uzu_engine::backends::{Backend, BackendSelection, select_backend};

const DEFAULT_MODEL_ID: &str = "liquidai:lfm2.5:230m:mirai:mirai-m:4";

struct DetectBackend;
impl BackendSelection for DetectBackend {
    type Output = &'static str;
    type Error = String;

    async fn select<B: Backend>(self) -> Result<Self::Output, Self::Error> {
        Ok(B::NAME)
    }
}

fn print_usage() {
    println!("Usage: cargo run -p uzu --example mirai_rocm_chat -- [OPTIONS]");
    println!("   or: ./run_mirai.sh [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --list                  List all available models in MiraiRegistry");
    println!("  --search <QUERY>        Filter available models by substring (e.g. 'lfm' or 'qwen')");
    println!("  --model <MODEL_ID>      Model ID or shorthand (default: {DEFAULT_MODEL_ID})");
    println!("  --prompt <TEXT>         Run a single prompt and exit");
    println!("  --system <TEXT>         Custom system prompt (default: 'You are a helpful assistant.')");
    println!("  --help, -h              Show this help message");
    println!();
    println!("Environment Variables:");
    println!("  UZU_REGISTRY_SHOW_ALL=1 Include all catalog models from MiraiRegistry");
}

async fn wait_for_full_catalog(engine: &Engine) -> Result<Vec<uzu::types::model::Model>, Box<dyn std::error::Error>> {
    let mut updates = engine.catalog_subscribe();
    while engine.catalog_is_refreshing() {
        if updates.next().await.is_none() {
            break;
        }
    }
    Ok(engine.models().await?)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let mut list_only = false;
    let mut search_query: Option<String> = None;
    let mut model_query = DEFAULT_MODEL_ID.to_string();
    let mut single_prompt: Option<String> = None;
    let mut system_prompt = "You are a helpful assistant.".to_string();

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            },
            "--list" | "-l" => {
                list_only = true;
            },
            "--search" | "-s" => {
                list_only = true;
                search_query = Some(args.next().ok_or("--search requires a query argument")?);
            },
            "--model" | "-m" => {
                model_query = args.next().ok_or("--model requires a model identifier")?;
            },
            "--prompt" | "-p" => {
                single_prompt = Some(args.next().ok_or("--prompt requires text")?);
            },
            "--system" => {
                system_prompt = args.next().ok_or("--system requires text")?;
            },
            other => {
                if single_prompt.is_none() && !other.starts_with('-') {
                    single_prompt = Some(other.to_string());
                } else {
                    eprintln!("Unknown option: {other}");
                    print_usage();
                    std::process::exit(1);
                }
            },
        }
    }

    let backend_name =
        select_backend(DetectBackend, "No backend available".to_string()).await?;
    println!(
        "[uzu] Backend: {} | Toolchain: {}",
        backend_name,
        Engine::toolchain_version()
    );

    let engine = Engine::new(EngineConfig::default()).await?;
    let models = wait_for_full_catalog(&engine).await?;

    if list_only {
        let query_lower = search_query.as_deref().unwrap_or("").to_lowercase();
        let filtered: Vec<_> = models
            .iter()
            .filter(|m| {
                if query_lower.is_empty() {
                    return true;
                }
                m.identifier.to_lowercase().contains(&query_lower)
                    || m.name().to_lowercase().contains(&query_lower)
            })
            .collect();

        println!(
            "\nFound {} models in MiraiRegistry{}:",
            filtered.len(),
            search_query.as_ref().map(|q| format!(" matching '{q}'")).unwrap_or_default()
        );
        println!("{:<48} | {:<38} | {:<10}", "MODEL ID", "NAME", "STATUS");
        println!("{}", "-".repeat(102));
        for model in filtered {
            let phase = engine.download_state(model).await.map(|s| s.phase);
            let status = match phase {
                Some(DownloadPhase::Downloaded {}) => "downloaded",
                Some(DownloadPhase::Downloading { .. }) => "downloading",
                _ => "remote",
            };
            println!("{:<48} | {:<38} | {:<10}", model.identifier, model.name(), status);
        }
        return Ok(());
    }

    let resolved_model = if let Some(exact) = engine.model(model_query.clone()).await? {
        exact
    } else {
        let q = model_query.to_lowercase();
        models
            .into_iter()
            .find(|m| m.identifier.to_lowercase().contains(&q) || m.name().to_lowercase().contains(&q))
            .ok_or_else(|| format!("Model '{model_query}' not found in MiraiRegistry. Run with --list to see models."))?
    };

    println!("[uzu] Selected model: {} ({})", resolved_model.identifier, resolved_model.name());

    if resolved_model.is_downloadable() {
        let state = engine.download_state(&resolved_model).await.map(|s| s.phase);
        if !matches!(state, Some(DownloadPhase::Downloaded {})) {
            println!("[uzu] Downloading {}...", resolved_model.identifier);
            let downloader = engine.download(&resolved_model).await?;
            while let Some(update) = downloader.next().await {
                print!("\r\u{001B}[2K[uzu] Download progress: {:6.2}%", update.progress() * 100.0);
                io::stdout().flush()?;
            }
            println!("\n[uzu] Download complete.");
        }
    }

    let session = engine.chat(resolved_model, ChatConfig::default()).await?;
    session.reset().await?;

    if let Some(prompt) = single_prompt {
        let messages = vec![
            ChatMessage::system().with_text(system_prompt),
            ChatMessage::user().with_text(prompt.clone()),
        ];
        println!("\nUser: {prompt}");
        run_turn(&session, messages).await?;
    } else {
        println!("\nInteractive ROCm Chat ready. Type 'exit' or 'quit' to stop.\n");
        let mut first_turn = true;
        let stdin = io::stdin();
        loop {
            print!("You> ");
            io::stdout().flush()?;
            let mut line = String::new();
            if stdin.lock().read_line(&mut line)? == 0 {
                break;
            }
            let input = line.trim();
            if input.is_empty() {
                continue;
            }
            if matches!(input, "exit" | "quit" | "/exit" | "/quit") {
                break;
            }
            let mut messages = Vec::new();
            if first_turn {
                messages.push(ChatMessage::system().with_text(system_prompt.clone()));
                first_turn = false;
            }
            messages.push(ChatMessage::user().with_text(input.to_string()));
            run_turn(&session, messages).await?;
            println!();
        }
    }

    Ok(())
}

async fn run_turn(
    session: &uzu::session::chat::ChatSession,
    messages: Vec<ChatMessage>,
) -> Result<(), Box<dyn std::error::Error>> {
    print!("Assistant> ");
    io::stdout().flush()?;

    let stream = session.reply_with_stream(messages, ChatReplyConfig::default()).await;
    let mut printed_reasoning = 0usize;
    let mut printed_text = 0usize;
    let mut in_reasoning = false;
    let mut final_stats = None;

    while let Some(chunk) = stream.next().await {
        match chunk {
            ChatSessionStreamChunk::ToolResults { .. } => {},
            ChatSessionStreamChunk::Replies { replies } => {
                if let Some(reply) = replies.last() {
                    if let Some(reasoning) = reply.message.reasoning()
                        && reasoning.len() > printed_reasoning
                    {
                        if !in_reasoning {
                            print!("<think>");
                            in_reasoning = true;
                        }
                        print!("{}", &reasoning[printed_reasoning..]);
                        io::stdout().flush()?;
                        printed_reasoning = reasoning.len();
                    }
                    if let Some(text) = reply.message.text()
                        && text.len() > printed_text
                    {
                        if in_reasoning {
                            print!("</think>\n");
                            in_reasoning = false;
                        }
                        print!("{}", &text[printed_text..]);
                        io::stdout().flush()?;
                        printed_text = text.len();
                    }
                    final_stats = Some(reply.stats.clone());
                }
            },
            ChatSessionStreamChunk::Error { error } => {
                eprintln!("\n[uzu] Stream error: {error}");
                return Err(error.into());
            },
        }
    }
    if in_reasoning {
        print!("</think>");
    }
    println!();

    if let Some(stats) = final_stats {
        let ttft = stats.time_to_first_token.unwrap_or_default();
        let prefill_tps = stats.prefill_tokens_per_second.unwrap_or_default();
        let gen_tps = stats.generate_tokens_per_second.unwrap_or_default();
        let out_tokens = stats.tokens_count_output.unwrap_or_default();
        println!(
            "[stats] output_tokens={} | ttft={:.2}s | prefill={:.2} tok/s | decode={:.2} tok/s | total={:.2}s",
            out_tokens, ttft, prefill_tps, gen_tps, stats.duration
        );
    }

    Ok(())
}
