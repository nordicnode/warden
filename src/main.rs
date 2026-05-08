//! Warden MCP Server - A hyper-strict MCP server for sandboxed LLM code execution
//!
//! This server provides a foolproof harness for smaller LLMs to compete on SWE-Bench Verified
//! by abstracting codebase navigation, enforcing strict TDD, and sandboxing all execution.
//!
//! FIXES APPLIED:
//! - Engine state is now persistent: McpRouter is created once and stored in WardenState.
//! - VectorDB is auto-initialized with workspace indexing when workspace is mounted.
//! - LSP servers are auto-initialized on session start.

mod state_machine;
mod workspace;
mod sandbox;
mod tools;
mod mcp_router;
mod tree_sitter_utils;
mod lsp_client;
mod vector_db;
mod context_engine;
mod editing_engine;
mod execution_tracer;
mod scratchpad;

use anyhow::Result;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;
use std::sync::Arc;
use tracing_subscriber::prelude::*;

use crate::state_machine::StateMachine;
use crate::workspace::OverlayfsWorkspace;
use crate::scratchpad::StateScratchpad;
use crate::mcp_router::McpRouter;

pub type SharedState = Arc<Mutex<WardenState>>;

pub struct WardenState {
    pub session_id: uuid::Uuid,
    pub workspace: Option<OverlayfsWorkspace>,
    pub scratchpad: StateScratchpad,
    pub state_machine: Arc<Mutex<StateMachine>>,
    /// Persisted router - created once when workspace is mounted, reused across all calls.
    /// This ensures VectorDB index, LSP servers, and opened documents survive across tool invocations.
    pub router: Option<Arc<McpRouter>>,
}

impl WardenState {
    pub fn new() -> Self {
        Self {
            session_id: uuid::Uuid::new_v4(),
            workspace: None,
            scratchpad: StateScratchpad::new(),
            state_machine: Arc::new(Mutex::new(StateMachine::new())),
            router: None,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let log_env = std::env::var("RUST_LOG").unwrap_or_else(|_| "info,warden=debug".to_string());
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(log_env))
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_ansi(false)
        )
        .init();

    // --- ADDED: Parse CLI args and mount the workspace ---
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        tracing::error!("Usage: warden-mcp <path-to-repository>");
        std::process::exit(1);
    }
    let repo_path = std::path::PathBuf::from(&args[1]);

    tracing::info!("Starting Warden MCP Server v{} on repo: {:?}", env!("CARGO_PKG_VERSION"), repo_path);

    let mut state_data = WardenState::new();
    
    // Initialize and mount the overlayfs workspace immediately
    let mut workspace = workspace::OverlayfsWorkspace::new(&repo_path, state_data.session_id)?;
    workspace.mount()?;
    let merged_path = workspace.merged_path().to_path_buf();
    state_data.workspace = Some(workspace);

    // Construct the persistent McpRouter ONCE with workspace binding.
    // This ensures VectorDB index, LSP servers, and opened documents survive across calls.
    let router = McpRouter::new().with_workspace(merged_path.clone());
    
    // Auto-initialize VectorDB with ONNX embedding model and workspace indexing.
    // The ONNX model (all-MiniLM-L6-v2) is required for semantic search.
    // Model paths can be configured via environment variables or fall back to defaults.
    {
        let ctx = router.context_engine();
        let default_model = std::env::var("WARDEN_EMBEDDING_MODEL_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("/usr/share/warden/models/embedding.onnx"));
        let default_tokenizer = std::env::var("WARDEN_TOKENIZER_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("/usr/share/warden/models/tokenizer.json"));
        if default_model.exists() && default_tokenizer.exists() {
            match ctx.initialize_vector_db_with_indexing(&default_model, &default_tokenizer).await {
                Ok(count) => tracing::info!(indexed_files = count, "VectorDB initialized with ONNX embedding model"),
                Err(e) => {
                    tracing::error!(error = %e, "VectorDB ONNX initialization failed — semantic search will be unavailable");
                }
            }
        } else {
            tracing::warn!(
                model = %default_model.display(),
                tokenizer = %default_tokenizer.display(),
                "ONNX model files not found. Set WARDEN_EMBEDDING_MODEL and WARDEN_TOKENIZER_PATH env vars. Semantic search disabled."
            );
        }
    }
    
    // Auto-initialize LSP servers so the first LSP tool call doesn't incur a
    // cold-start delay that could confuse a weaker model.
    {
        if let Err(e) = router.initialize_lsp_if_needed().await {
            tracing::warn!(error = %e, "LSP auto-initialization failed");
        }
    }
    
    state_data.router = Some(Arc::new(router));

    let state: SharedState = Arc::new(Mutex::new(state_data));
    // -----------------------------------------------------

    run_stdio_transport(state).await?;
    Ok(())
}

async fn run_stdio_transport(state: SharedState) -> Result<()> {
    let stdin = BufReader::new(tokio::io::stdin());
    let mut stdout = tokio::io::stdout();
    let mut lines = stdin.lines();

    tracing::info!("Stdio transport ready, awaiting JSON-RPC requests");

    while let Ok(Some(line)) = lines.next_line().await {
        if line.trim().is_empty() {
            continue;
        }
        tracing::debug!(line_len = line.len(), "Received JSON-RPC request");
        let request: mcp_core::transport::JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to parse JSON-RPC request");
                let error_response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": format!("Parse error: {}", e)
                    },
                    "id": serde_json::Value::Null
                });
                write_json_response(&mut stdout, &error_response).await?;
                continue;
            }
        };
        let response = handle_mcp_request(state.clone(), request).await;
        write_json_response(&mut stdout, &response).await?;
    }
    tracing::info!("Stdin closed, shutting down");
    Ok(())
}

async fn write_json_response(
    stdout: &mut tokio::io::Stdout,
    response: &serde_json::Value,
) -> Result<()> {
    let json_str = serde_json::to_string(response)?;
    stdout.write_all(json_str.as_bytes()).await?;
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;
    Ok(())
}

async fn handle_mcp_request(
    state: SharedState,
    request: mcp_core::transport::JsonRpcRequest,
) -> serde_json::Value {
    match request.method.as_str() {
        "initialize" => {
            tracing::debug!("Handling initialize request");
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": request.id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "warden-mcp",
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": { "listChanged": false }
                    },
                    "instructions": "Welcome to Warden MCP. You are operating in a hyper-strict sandboxed environment for SWE-Bench tasks."
                }
            })
        }
        "tools/list" => {
            tracing::debug!("Handling tools/list request");
            // Clone the Arc before locking to avoid holding the outer Mutex
            // across the inner Mutex await (prevents potential deadlock under
            // concurrent tools/list and tools/call requests).
            let sm_arc = state.lock().await.state_machine.clone();
            let phase = sm_arc.lock().await.current_phase();
            let tools = crate::tools::list_available_tools(&phase);
            serde_json::json!({"jsonrpc": "2.0", "id": request.id, "result": tools})
        }
        "tools/call" => {
            tracing::debug!("Handling tools/call request");
            let params = request.params.unwrap_or_default();
            let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or_default();
            let arguments = params.get("arguments").cloned().unwrap_or_default();
            match mcp_router::route_tool_call(state.clone(), tool_name, arguments).await {
                Ok(result) => serde_json::json!({"jsonrpc": "2.0", "id": request.id, "result": result}),
                Err(e) => {
                    tracing::error!(tool = tool_name, error = %e, "Tool call failed");
                    serde_json::json!({"jsonrpc": "2.0", "id": request.id, "error": {"code": -32603, "message": format!("Internal error: {}", e)}})
                }
            }
        }
        "prompts/list" => {
            tracing::debug!("Handling prompts/list request");
            let prompts = serde_json::json!({
                "prompts": [{
                    "name": "swe_bench_start",
                    "description": "Start a SWE-Bench task with the given issue description. Returns a structured system prompt for the EXPLORE phase.",
                    "arguments": {
                        "name": "issue_text",
                        "description": "Natural language description of the bug or feature request",
                        "required": true,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "issue_text": {
                                    "type": "string",
                                    "description": "The issue description from the SWE-Bench task"
                                }
                            },
                            "required": ["issue_text"]
                        }
                    }
                }]
            });
            serde_json::json!({"jsonrpc": "2.0", "id": request.id, "result": prompts})
        }
        "prompts/get" => {
            tracing::debug!("Handling prompts/get request");
            let params = request.params.unwrap_or_default();
            let prompt_name = params.get("name").and_then(|v| v.as_str()).unwrap_or_default();
            if prompt_name == "swe_bench_start" {
                let issue_text = params.get("arguments")
                    .and_then(|v| v.get("issue_text"))
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let prompt_content = generate_swe_bench_start_prompt(issue_text);
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": request.id,
                    "result": {
                        "description": "SWE-Bench start prompt for exploring a bug and writing a failing test",
                        "messages": [{"role": "system", "content": prompt_content}]
                    }
                })
            } else {
                serde_json::json!({"jsonrpc": "2.0", "id": request.id, "error": {"code": -32602, "message": format!("Unknown prompt: {}", prompt_name)}})
            }
        }
        "resources/list" => {
            serde_json::json!({"jsonrpc": "2.0", "id": request.id, "result": {"resources": []}})
        }
        _ => {
            tracing::warn!(method = %request.method, "Unknown MCP method");
            serde_json::json!({"jsonrpc": "2.0", "id": request.id, "error": {"code": -32601, "message": format!("Unknown method: {}", request.method)}})
        }
    }
}

fn generate_swe_bench_start_prompt(issue_text: &str) -> String {
    let mut prompt = String::new();
    prompt.push_str("You are operating within the **Warden MCP harness**, a hyper-strict sandboxed environment designed for solving SWE-Bench tasks.\n\n");
    prompt.push_str("OPERATING CONTEXT\n===============\n");
    prompt.push_str("Current Phase: EXPLORE\n");
    prompt.push_str("Harness: Warden MCP with overlayfs sandbox and Tree-sitter code analysis\n");
    prompt.push_str("Objective: Fix the bug described below while following strict TDD methodology\n\n");
    prompt.push_str("YOUR TASK\n========\n```\n");
    prompt.push_str(issue_text);
    prompt.push_str("\n```\n\n");
    prompt.push_str("CRITICAL RULES (Violating these will cause task failure)\n=====================================================\n\n");
    prompt.push_str("1. EXPLORE Phase Constraints: You are in the EXPLORE phase. You may ONLY use:\n");
    prompt.push_str("   - find_relevant_code - Query the semantic code search with the issue description above\n");
    prompt.push_str("   - get_skeleton - Extract function signatures and imports from specific files\n");
    prompt.push_str("   - read_file, list_directory, search_files - Read-only file operations\n");
    prompt.push_str("   - lsp_* tools - Code navigation (find definition, references, hover)\n");
    prompt.push_str("   - get_scratchpad, set_scratchpad, get_status - State management\n");
    prompt.push_str("   - transition_phase - Move to the next phase when ready\n\n");
    prompt.push_str("2. NO CODE EDITING in EXPLORE: You cannot use modify_ast, apply_unified_diff, or any write operations until you reach the PATCH phase.\n\n");
    prompt.push_str("3. IMMEDIATE ACTIONS Required:\n");
    prompt.push_str("   a) First: Use find_relevant_code with the issue text to find relevant code\n");
    prompt.push_str("   b) Second: Write your analysis and plan in the scratchpad using set_scratchpad\n");
    prompt.push_str("   c) Third: Explore the codebase to understand the bug\n");
    prompt.push_str("   d) Fourth: Write a failing test in REPRODUCE phase before attempting any fix\n\n");
    prompt.push_str("4. TDD Methodology:\n");
    prompt.push_str("   - You MUST write a failing test that reproduces the bug BEFORE making any fix\n");
    prompt.push_str("   - The test must be written in the REPRODUCE phase (transition after exploration)\n");
    prompt.push_str("   - Only after the test fails correctly can you transition to PATCH and apply fixes\n");
    prompt.push_str("   - Fixes require a Why Gate explanation before editing tools are unlocked\n\n");
    prompt.push_str("5. Sandbox Security: All code execution happens inside a bwrap sandbox with overlayfs.\n");
    prompt.push_str("   - You cannot read/write files outside the sandboxed workspace\n");
    prompt.push_str("   - All test execution is sandboxed and traced\n\n");
    prompt.push_str("YOUR PLAN (Write this to scratchpad immediately)\n==============================================\n\n");
    prompt.push_str("After using find_relevant_code, use set_scratchpad to write:\n");
    prompt.push_str("1. Summary of the bug and affected components\n");
    prompt.push_str("2. List of relevant files to examine\n");
    prompt.push_str("3. Your hypothesis about the root cause\n");
    prompt.push_str("4. The failing test you will write to reproduce the bug\n\n");
    prompt.push_str("Example scratchpad content:\n```\n## Bug Analysis\n[Your analysis]\n\n## Relevant Files\n- file1.py: [why relevant]\n\n## Root Cause Hypothesis\n[Your theory]\n\n## Test Plan\n[Describe the failing test]\n```\n\n");
    prompt.push_str("WHEN YOU ARE READY TO PROCEED\n==============================\n\n");
    prompt.push_str("1. Ensure your plan is in the scratchpad via set_scratchpad\n");
    prompt.push_str("2. Use get_status to confirm you are in EXPLORE phase\n");
    prompt.push_str("3. When you have enough information, use transition_phase with target \"REPRODUCE\"\n");
    prompt.push_str("4. In REPRODUCE phase, use write_test to write a failing test\n");
    prompt.push_str("5. Run the test with run_test to verify it fails as expected\n");
    prompt.push_str("6. Only then transition to PATCH phase and apply your fix\n\n");
    prompt.push_str("Remember: The bug fix is not complete until the test passes AND the fix is verified in the VERIFY phase.\n");
    prompt
}
