//! MCP Router - Intercepts and validates all LLM tool calls against the state machine
//!
//! Every tool call must pass through the router which validates:
//! 1. The tool is allowed in the current phase
//! 2. The tool call is properly formatted
//! 3. Pre-conditions (like Why Gate) are satisfied
//!
//! FIXES APPLIED:
//! - route_tool_call now uses the PERSISTED McpRouter stored in WardenState.router
//!   instead of creating a new one on every call. This preserves VectorDB index,
//!   LSP servers, and opened documents across tool invocations.
//! - transition_phase handler is properly connected.

use crate::state_machine::{StateMachine, Phase};
use crate::context_engine::ContextEngine;
use crate::editing_engine::EditingEngine;
use crate::execution_tracer::ExecutionTracer;
use anyhow::Result;
use serde_json::Value;
use std::path::PathBuf;
use tracing::{info, warn};

pub struct McpRouter {
    context_engine: ContextEngine,
    editing_engine: EditingEngine,
    execution_tracer: ExecutionTracer,
    workspace_path: Option<PathBuf>,
}

impl Clone for McpRouter {
    fn clone(&self) -> Self {
        Self {
            context_engine: self.context_engine.clone(),
            editing_engine: self.editing_engine.clone(),
            execution_tracer: self.execution_tracer.clone(),
            workspace_path: self.workspace_path.clone(),
        }
    }
}

impl McpRouter {
    pub fn new() -> Self {
        Self {
            context_engine: ContextEngine::new(),
            editing_engine: EditingEngine::new(),
            execution_tracer: ExecutionTracer::new(),
            workspace_path: None,
        }
    }

    pub fn with_workspace(mut self, workspace_path: PathBuf) -> Self {
        self.workspace_path = Some(workspace_path.clone());
        self.context_engine = self.context_engine.with_workspace(workspace_path.clone());
        self.execution_tracer = ExecutionTracer::with_workspace(workspace_path.clone());
        self.editing_engine = self.editing_engine.with_workspace(workspace_path);
        self
    }

    /// Returns a reference to the ContextEngine for initialization purposes
    pub fn context_engine(&self) -> &ContextEngine {
        &self.context_engine
    }

    /// Clone the router for use in tool calls.
    /// With Arc<McpRouter> the caller should just Arc::clone the Arc
    /// rather than deep-copying the struct.
    #[allow(dead_code)]
    pub fn clone_router(&self) -> Self {
        self.clone()
    }

    /// Eagerly initialize LSP servers for common workspace languages.
    /// Prevents cold-start delays on the first LSP tool call.
    pub async fn initialize_lsp_if_needed(&self) -> Result<()> {
        use tracing::warn;
        // Try common languages; ensure_server is a no-op if already running
        for lang in &["python", "rust", "javascript", "typescript"] {
            if let Err(e) = self.context_engine.ensure_lsp_server(lang).await {
                warn!(language = lang, error = %e, "LSP server not available");
            }
        }
        Ok(())
    }
}

/// Generate phase-aware next-step guidance for successful tool responses
fn get_next_steps(tool_name: &str, phase: &Phase) -> String {
    match (tool_name, phase) {
        ("find_relevant_code", Phase::Explore) => {
            "Review the returned code snippets. Use read_file to examine the most relevant files in detail. Use get_skeleton for a high-level overview. Write your analysis to the scratchpad with set_scratchpad. When you understand the bug, use transition_phase to move to REPRODUCE.".to_string()
        }
        ("read_file", Phase::Explore) => {
            "Examine the code for the bug described in the issue. Use get_skeleton for overview, find_relevant_code for semantic search, or lsp_get_definition to trace symbols. Record findings in the scratchpad.".to_string()
        }
        ("write_test", Phase::Reproduce) => {
            "Test file written. Now run it with run_test to verify it fails with an error matching the bug description. The test must fail before you can transition to PATCH.".to_string()
        }
        ("run_test", Phase::Reproduce) => {
            "If the test failed as expected, you can transition to PATCH phase. If it passed unexpectedly, revise your test to better reproduce the bug.".to_string()
        }
        ("provide_why_explanation", Phase::Patch) => {
            "Why Gate satisfied. You can now use modify_ast or apply_unified_diff to apply your fix.".to_string()
        }
        ("modify_ast", Phase::Patch) | ("apply_unified_diff", Phase::Patch) => {
            "Fix applied. Use transition_phase to move to VERIFY, then run run_tests to confirm the fix works.".to_string()
        }
        ("run_tests", Phase::Verify) => {
            "If all tests pass, the fix is complete. If tests fail, use transition_phase to go back to PATCH and adjust your fix.".to_string()
        }
        _ => String::new(),
    }
}

/// Route a tool call - validates and executes against the PERSISTED McpRouter
/// 
/// CRITICAL FIX: Uses the McpRouter stored in WardenState.router instead of
/// creating a new one on every invocation. This preserves:
/// - VectorDB index (semantic search results)
/// - LSP server connections (code navigation)
/// - Opened documents (hover, completion context)
pub async fn route_tool_call(
    state: crate::SharedState,
    tool_name: &str,
    arguments: Value,
) -> Result<Value> {
    // Extract what we need under a short lock, then drop it before dispatching
    let (state_machine, router) = {
        let state_guard = state.lock().await;
        let state_machine = state_guard.state_machine.clone();
        // Clone the Arc instead of cloning the entire McpRouter struct.
        // McpRouter already shares engine state via Arc internally.
        let router: std::sync::Arc<McpRouter> = match &state_guard.router {
            Some(r) => std::sync::Arc::clone(r),
            None => {
                let r = McpRouter::new();
                let r = if let Some(workspace) = &state_guard.workspace {
                    r.with_workspace(workspace.merged_path().to_path_buf())
                } else {
                    r
                };
                std::sync::Arc::new(r)
            }
        };
        (state_machine, router)
    }; // state_guard dropped here
    
    // Validate tool is allowed using the PERSISTED StateMachine
    let sm = state_machine.lock().await;
    let current_phase = sm.current_phase();
    match sm.is_tool_allowed(tool_name) {
        Ok(_) => info!(tool = tool_name, phase = ?current_phase, "Tool call validated"),
        Err(e) => {
            warn!(tool = tool_name, error = %e, "Tool call rejected by state machine");
            return Ok(serde_json::json!({
                "error": true,
                "message": e.to_string(),
                "phase": current_phase.to_string(),
                "allowed_tools": sm.allowed_tools(),
                "recovery": format!(
                    "Tool '{}' is not available in {} phase. Use one of the allowed tools listed above, or use transition_phase to change phases.",
                    tool_name, current_phase
                ),
            }));
        }
    }
    drop(sm);

    // Route to appropriate handler on the PERSISTED router
    let result = match tool_name {
        // EXPLORE tools
        "get_skeleton" => router.handle_get_skeleton(&arguments).await,
        "find_relevant_code" => router.handle_find_relevant_code(&arguments).await,
        "lsp_find_references" => router.handle_lsp_find_references(&arguments).await,
        "lsp_get_hover" => router.handle_lsp_get_hover(&arguments).await,
        "lsp_get_definition" => router.handle_lsp_get_definition(&arguments).await,
        "lsp_complete" => router.handle_lsp_complete(&arguments).await,
        "read_file" => router.handle_read_file(&arguments).await,
        "list_directory" => router.handle_list_directory(&arguments).await,
        "search_files" => router.handle_search_files(&arguments).await,

        // REPRODUCE tools
        "write_test" => router.handle_write_test(&state_machine, &arguments).await,
        "run_test" => router.handle_run_test(&state_machine, &arguments).await,
        "run_traced_script" => router.handle_run_traced_script(&arguments).await,

        // PATCH tools
        "modify_ast" => router.handle_modify_ast(&state_machine, &arguments).await,
        "apply_unified_diff" => router.handle_apply_unified_diff(&state_machine, &arguments).await,
        "provide_why_explanation" => router.handle_provide_why_explanation(&state_machine, &arguments).await,

        // VERIFY tools
        "run_lint" => router.handle_run_lint(&arguments).await,
        "run_compile" => router.handle_run_compile(&arguments).await,
        "run_tests" => router.handle_run_tests(&arguments).await,

        // Universal tools
        "get_scratchpad" => {
            let state_guard = state.lock().await;
            router.handle_get_scratchpad(&state_guard).await
        }
        "set_scratchpad" => {
            let mut state_guard = state.lock().await;
            router.handle_set_scratchpad(&mut state_guard, &arguments).await
        }
        "get_status" => {
            let state_guard = state.lock().await;
            router.handle_get_status(&state_machine, &state_guard).await
        }
        // Phase transitions
        "transition_phase" => router.handle_transition_phase(&state_machine, &arguments).await,

        _ => Ok(serde_json::json!({
            "error": true,
            "message": format!("Unknown tool: {}", tool_name),
            "recovery": format!("Tool '{}' is not recognized. Use get_status to see available tools and current phase.", tool_name),
        })),
    };

    // Inject next_steps guidance into successful responses
    let mut response = result?;
    if !response.get("error").and_then(|v| v.as_bool()).unwrap_or(false) {
        let next = get_next_steps(tool_name, &current_phase);
        if !next.is_empty() {
            response["next_steps"] = serde_json::Value::String(next);
        }
    }
    Ok(response)
}

impl McpRouter {
    // ==================== EXPLORE handlers ====================

    async fn handle_get_skeleton(&self, args: &Value) -> Result<Value> {
        let filepath = args.get("filepath")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing filepath"))?;

        let skeleton = self.context_engine.get_skeleton(PathBuf::from(filepath)).await?;
        Ok(serde_json::json!({"skeleton": skeleton}))
    }

    async fn handle_find_relevant_code(&self, args: &Value) -> Result<Value> {
        let issue_text = args.get("issue_text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing issue_text"))?;

        let results = self.context_engine.find_relevant_code(issue_text).await?;
        Ok(serde_json::json!({"relevant_code": results}))
    }

    async fn handle_lsp_find_references(&self, args: &Value) -> Result<Value> {
        let file_path = args.get("file_path").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("Missing file_path"))?;
        let line = args.get("line").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
        let column = args.get("column").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
        let include_declaration = args.get("include_declaration").and_then(|v| v.as_bool()).unwrap_or(true);

        let references = self.context_engine.lsp_find_references(file_path, line, column, include_declaration).await?;
        Ok(serde_json::json!({"references": references}))
    }

    async fn handle_lsp_get_hover(&self, args: &Value) -> Result<Value> {
        let file_path = args.get("file_path").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("Missing file_path"))?;
        let line = args.get("line").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
        let column = args.get("column").and_then(|v| v.as_u64()).unwrap_or(1) as u32;

        let hover = self.context_engine.lsp_get_hover(file_path, line, column).await?;
        Ok(serde_json::json!({"hover": hover}))
    }

    async fn handle_lsp_get_definition(&self, args: &Value) -> Result<Value> {
        let file_path = args.get("file_path").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("Missing file_path"))?;
        let line = args.get("line").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
        let column = args.get("column").and_then(|v| v.as_u64()).unwrap_or(1) as u32;

        let definition = self.context_engine.lsp_get_definition(file_path, line, column).await?;
        Ok(serde_json::json!({"definition": definition}))
    }

    async fn handle_lsp_complete(&self, args: &Value) -> Result<Value> {
        let file_path = args.get("file_path").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("Missing file_path"))?;
        let line = args.get("line").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
        let column = args.get("column").and_then(|v| v.as_u64()).unwrap_or(1) as u32;

        let completions = self.context_engine.lsp_complete(file_path, line, column).await?;
        Ok(serde_json::json!({"completions": completions}))
    }

    async fn handle_read_file(&self, args: &Value) -> Result<Value> {
        let path = args.get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing path"))?;

        let start_line = args.get("start_line").and_then(|v| v.as_i64()).map(|v| v as usize);
        let end_line = args.get("end_line").and_then(|v| v.as_i64()).map(|v| v as usize);

        let content = self.context_engine.read_file(path, start_line, end_line).await?;
        Ok(serde_json::json!({"content": content}))
    }

    async fn handle_list_directory(&self, args: &Value) -> Result<Value> {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let entries = self.context_engine.list_directory(path).await?;
        Ok(serde_json::json!({"entries": entries}))
    }

    async fn handle_search_files(&self, args: &Value) -> Result<Value> {
        let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or_default();
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let file_type = args.get("file_type").and_then(|v| v.as_str());

        let results = self.context_engine.search_files(pattern, path, file_type).await?;
        Ok(serde_json::json!({"results": results}))
    }

    // ==================== REPRODUCE handlers ====================

    async fn handle_write_test(
        &self,
        state_machine: &std::sync::Arc<tokio::sync::Mutex<StateMachine>>,
        args: &Value,
    ) -> Result<Value> {
        let path = args.get("path").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("Missing path"))?;
        let content = args.get("content").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("Missing content"))?;

        let _result = self.editing_engine.write_file(path, content).await?;
        
        // Mark test as written in the PERSISTED state machine
        state_machine.lock().await.mark_test_written();

        Ok(serde_json::json!({
            "success": true,
            "path": path,
            "message": "Test file written successfully. Run it to verify it fails matching the bug description."
        }))
    }

    async fn handle_run_test(
        &self,
        state_machine: &std::sync::Arc<tokio::sync::Mutex<StateMachine>>,
        args: &Value,
    ) -> Result<Value> {
        let test_path = args.get("test_path").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("Missing test_path"))?;
        let timeout = args.get("timeout_seconds").and_then(|v| v.as_i64()).unwrap_or(30) as u64;

        // RED/GREEN LOCK: Compile-check the test file BEFORE executing it.
        // If compilation fails, the test has a syntax/compilation error, NOT a bug assertion.
        // Do NOT unlock the state machine in that case — the LLM must fix the syntax first.
        //
        // Also handle the case where run_compile returns Err (unknown extension, no sandbox)
        // gracefully — return structured JSON instead of leaking a raw anyhow error.
        let compile_result = match self.execution_tracer.run_compile(Some(&[test_path.to_string()])).await {
            Ok(r) => r,
            Err(e) => {
                return Ok(serde_json::json!({
                    "error": true,
                    "message": format!("Cannot compile-check the test file: {}", e),
                }));
            }
        };
        if compile_result.exit_code != 0 {
            return Ok(serde_json::json!({
                "error": true,
                "message": "Test failed due to a syntax/compilation error, not a bug assertion. Fix the syntax and run again.",
                "compile_exit_code": compile_result.exit_code,
                "compile_stderr": compile_result.stderr,
                "compile_stdout": compile_result.stdout,
            }));
        }

        // Compilation passed — now run the actual test
        let result = self.execution_tracer.run_test(test_path, timeout).await?;

        // Only unlock the state machine if compilation passed AND the test execution
        // itself fails (indicating a real bug assertion failure, not a syntax error).
        if result.exit_code != 0 {
            state_machine.lock().await.mark_test_failing_matching_bug();
        }

        Ok(serde_json::json!({
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time_ms": result.execution_time_ms,
            "test_fails_matching_bug": result.exit_code != 0,
        }))
    }

    async fn handle_run_traced_script(&self, args: &Value) -> Result<Value> {
        let command = args.get("command")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).map(String::from).collect::<Vec<String>>())
            .ok_or_else(|| anyhow::anyhow!("Missing command array"))?;

        let working_dir = args.get("working_directory").and_then(|v| v.as_str()).map(PathBuf::from);

        let result = self.execution_tracer.run_traced_script(&command, working_dir).await?;
        let summary = crate::sandbox::summarize_error(&result.stderr, "python");

        Ok(serde_json::json!({
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "trace_summary": summary,
            "execution_time_ms": result.execution_time_ms,
        }))
    }

    // ==================== PATCH handlers ====================

    async fn handle_modify_ast(
        &self,
        state_machine: &std::sync::Arc<tokio::sync::Mutex<StateMachine>>,
        args: &Value,
    ) -> Result<Value> {
        let file = args.get("file_path").or_else(|| args.get("file")).and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("Missing file or file_path"))?;
        let node_type = args.get("node_type").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("Missing node_type"))?;
        let node_name = args.get("node_name").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("Missing node_name"))?;
        let new_code = args.get("new_code").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("Missing new_code"))?;

        // Why gate check on the PERSISTED state machine
        if state_machine.lock().await.get_why_explanation().is_none() {
            return Ok(serde_json::json!({
                "error": true,
                "message": "Why Gate not satisfied. Provide explanation with provide_why_explanation first."
            }));
        }

        let result = self.editing_engine.modify_ast(file, node_type, node_name, new_code).await?;

        // Re-index modified file for RAG so subsequent semantic searches reflect changes
        if let Some(ws) = &self.workspace_path {
            let full_path = ws.join(file);
            if let Ok(content) = std::fs::read_to_string(&full_path) {
                let _ = self.context_engine.index_content(&full_path, &content).await;
            }
        }

        Ok(serde_json::json!({
            "success": true,
            "diff": result,
            "message": "AST modification applied. Proceed to VERIFY phase."
        }))
    }

    async fn handle_apply_unified_diff(
        &self,
        state_machine: &std::sync::Arc<tokio::sync::Mutex<StateMachine>>,
        args: &Value,
    ) -> Result<Value> {
        let patch = args.get("patch").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("Missing patch"))?;

        // Why gate check on the PERSISTED state machine
        if state_machine.lock().await.get_why_explanation().is_none() {
            return Ok(serde_json::json!({
                "error": true,
                "message": "Why Gate not satisfied. Provide explanation with provide_why_explanation first."
            }));
        }

        let result = self.editing_engine.apply_unified_diff(patch).await?;

        // Re-index modified files for RAG
        for changed_file in &result.changed_files {
            if let Some(ws) = &self.workspace_path {
                let full_path = ws.join(changed_file);
                if let Ok(content) = std::fs::read_to_string(&full_path) {
                    let _ = self.context_engine.index_content(&full_path, &content).await;
                }
            }
        }

        Ok(serde_json::json!({
            "success": true,
            "diff": result,
            "message": "Patch applied. Proceed to VERIFY phase."
        }))
    }

    async fn handle_provide_why_explanation(
        &self,
        state_machine: &std::sync::Arc<tokio::sync::Mutex<StateMachine>>,
        args: &Value,
    ) -> Result<Value> {
        let explanation = args.get("explanation").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("Missing explanation"))?;

        match state_machine.lock().await.set_why_explanation(explanation.to_string()) {
            Ok(_) => Ok(serde_json::json!({
                "success": true,
                "message": "Why Gate satisfied. Edit tools are now unlocked for this patch."
            })),
            Err(e) => Ok(serde_json::json!({
                "error": true,
                "message": e.to_string(),
            })),
        }
    }

    // ==================== VERIFY handlers ====================

    async fn handle_run_lint(&self, args: &Value) -> Result<Value> {
        let files = args.get("files").and_then(|v| v.as_array()).map(|arr| {
            arr.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<String>>()
        });

        let result = self.execution_tracer.run_lint(files.as_deref()).await?;

        Ok(serde_json::json!({
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }))
    }

    async fn handle_run_compile(&self, args: &Value) -> Result<Value> {
        let files = args.get("files").and_then(|v| v.as_array()).map(|arr| {
            arr.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<String>>()
        });

        let result = self.execution_tracer.run_compile(files.as_deref()).await?;

        Ok(serde_json::json!({
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }))
    }

    async fn handle_run_tests(&self, args: &Value) -> Result<Value> {
        let test_filter = args.get("test_filter").and_then(|v| v.as_str());

        let result = self.execution_tracer.run_all_tests(test_filter).await?;

        // If tests pass, we can consider the fix verified
        if result.exit_code == 0 {
            info!("All tests passed - fix verified!");
        }

        Ok(serde_json::json!({
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "tests_passed": result.exit_code == 0,
        }))
    }

    // ==================== Universal handlers ====================

    async fn handle_get_scratchpad(&self, state: &crate::WardenState) -> Result<Value> {
        Ok(serde_json::json!({
            "content": state.scratchpad.content(),
            "tracked_files": state.scratchpad.tracked_files(),
            "current_goal": state.scratchpad.current_goal(),
        }))
    }

    async fn handle_set_scratchpad(&self, state: &mut crate::WardenState, args: &Value) -> Result<Value> {
        let content = args.get("content").and_then(|v| v.as_str()).unwrap_or_default();
        state.scratchpad.set_content(content.to_string());
        
        // Increment turn counter and auto-prune old tracked files.
        // Without increment_turn, prune never removes anything because
        // all files are tracked at turn 0 and cutoff = 0.saturating_sub(3) = 0.
        state.scratchpad.increment_turn();
        state.scratchpad.prune_old_entries(Some(3));
        
        // Integrate scratchpad parsing: auto-parse tracked files and goal from content
        state.scratchpad.parse_tracked_from_content();
        if let Some(goal) = state.scratchpad.parse_goal() {
            state.scratchpad.set_current_goal(goal);
        }

        Ok(serde_json::json!({
            "success": true,
            "message": "Scratchpad updated. Parsed goal and tracked files from content.",
            "parsed_goal": state.scratchpad.current_goal(),
            "tracked_file_count": state.scratchpad.tracked_files().len(),
        }))
    }

    async fn handle_get_status(
        &self,
        state_machine: &std::sync::Arc<tokio::sync::Mutex<StateMachine>>,
        state: &crate::WardenState,
    ) -> Result<Value> {
        let sm = state_machine.lock().await;
        let phase = sm.current_phase();

        let next_action = match phase {
            Phase::Explore => "Use find_relevant_code with the issue text to locate relevant code. Read files with read_file. Record your analysis in the scratchpad. When ready, transition to REPRODUCE.",
            Phase::Reproduce => {
                if !sm.is_test_written() {
                    "Write a failing test with write_test that reproduces the bug described in the issue."
                } else if !sm.is_test_failing() {
                    "Run your test with run_test. It must fail to confirm it reproduces the bug."
                } else {
                    "Test fails as expected. Use transition_phase to move to PATCH."
                }
            }
            Phase::Patch => {
                if sm.get_why_explanation().is_none() {
                    "Provide a 1-sentence explanation of your fix using provide_why_explanation before editing."
                } else {
                    "Use modify_ast or apply_unified_diff to apply your fix. Then transition to VERIFY."
                }
            }
            Phase::Verify => "Run run_tests to verify your fix. If tests pass, you're done. If they fail, transition back to PATCH.",
        };

        Ok(serde_json::json!({
            "phase": phase.to_string(),
            "state_machine": sm.get_state_json(),
            "session_id": state.session_id.to_string(),
            "workspace_mounted": state.workspace.as_ref().map(|w| w.is_mounted()).unwrap_or(false),
            "current_goal": state.scratchpad.current_goal(),
            "tracked_files": state.scratchpad.tracked_files().iter().map(|f| f.path.clone()).collect::<Vec<_>>(),
            "scratchpad_summary": state.scratchpad.generate_context_summary().chars().take(500).collect::<String>(),
            "next_action": next_action,
            "allowed_tools": sm.allowed_tools(),
        }))
    }

    // ==================== Phase transition handler ====================

    async fn handle_transition_phase(
        &self,
        state_machine: &std::sync::Arc<tokio::sync::Mutex<StateMachine>>,
        args: &Value,
    ) -> Result<Value> {
        let target_phase = args.get("target_phase")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing target_phase"))?;

        let target = match target_phase.to_uppercase().as_str() {
            "EXPLORE" => Phase::Explore,
            "REPRODUCE" => Phase::Reproduce,
            "PATCH" => Phase::Patch,
            "VERIFY" => Phase::Verify,
            _ => {
                return Ok(serde_json::json!({
                    "error": true,
                    "message": format!("Invalid phase: {}. Valid phases are: EXPLORE, REPRODUCE, PATCH, VERIFY", target_phase),
                }));
            }
        };

        let mut sm = state_machine.lock().await;
        let previous_phase = sm.current_phase().to_string();
        match sm.transition(target) {
            Ok(_) => {
                info!(target_phase = %target_phase, "Phase transition successful");
                Ok(serde_json::json!({
                    "success": true,
                    "previous_phase": previous_phase,
                    "current_phase": target_phase.to_uppercase(),
                    "message": format!("Transitioned from {} to {}.", previous_phase, target_phase),
                }))
            }
            Err(e) => {
                warn!(target_phase = %target_phase, error = %e, "Phase transition failed");
                Ok(serde_json::json!({
                    "error": true,
                    "message": e.to_string(),
                    "valid_transitions": match sm.current_phase() {
                        Phase::Explore => vec!["REPRODUCE"],
                        Phase::Reproduce => vec!["EXPLORE", "PATCH"],
                        Phase::Patch => vec!["VERIFY", "EXPLORE"],
                        Phase::Verify => vec!["EXPLORE", "PATCH"],
                    },
                }))
            }
        }
    }
}