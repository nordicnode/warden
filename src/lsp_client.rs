//! LSP Client - Real Language Server Protocol client
//!
//! Spawns language servers (pylsp, rust-analyzer, typescript-language-server) via stdio
//! and communicates using the Language Server Protocol over JSON-RPC.
//!
//! REFACTORED: Fully async I/O using tokio::process and tokio::io.
//! Previously used std::process + std::io::BufReader wrapped in spawn_blocking,
//! which caused a permanent deadlock when tokio::time::timeout fired:
//!   - timeout cancelled the spawn_blocking JoinHandle future
//!   - the OS thread remained blocked inside the Mutex<HashMap<..., LspConnection>>
//!   - all subsequent tool calls deadlocked trying to acquire the same Mutex
//!
//! Now LspConnection is fully async. The Mutex is held only for brief
//! remove/insert operations on the HashMap — never during I/O.

use anyhow::{Result, anyhow};
use lsp_types::{
    CompletionItem as LspCompletionItem, GotoDefinitionResponse, Location as LspLocation,
};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::time::{timeout, Duration};
use tracing::{info, warn};

use crate::context_engine::{CompletionItem, LspLocation as ContextLspLocation};
use lsp_types::Url;

/// A live connection to a language server.
/// All I/O is async — callers must .await.
struct LspConnection {
    process: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    root_uri: String,
}

impl LspConnection {
    /// Send an initialize request, then the 'initialized' notification.
    async fn initialize(&mut self) -> Result<()> {
        let init_params = serde_json::json!({
            "processId": std::process::id(),
            "rootUri": self.root_uri,
            "capabilities": {
                "workspace": {
                    "configuration": true,
                    "didChangeConfiguration": { "dynamicRegistration": true }
                },
                "textDocument": {
                    "hover": { "dynamicRegistration": true },
                    "references": { "dynamicRegistration": true },
                    "definition": { "dynamicRegistration": true },
                    "completion": {
                        "dynamicRegistration": true,
                        "completionItem": { "snippetSupport": false }
                    },
                    "didOpen": { "dynamicRegistration": true },
                    "didChange": { "dynamicRegistration": true }
                }
            },
            "workspaceFolders": [{
                "uri": self.root_uri,
                "name": "workspace"
            }]
        });

        self.send_request("initialize", init_params).await?;

        // Send initialized notification
        let initialized = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {}
        });
        let msg = serde_json::to_string(&initialized)?;
        let header = format!("Content-Length: {}\r\n\r\n", msg.len());
        self.stdin.write_all(header.as_bytes()).await?;
        self.stdin.write_all(msg.as_bytes()).await?;
        self.stdin.flush().await?;

        info!(root_uri = %self.root_uri, "LSP server initialized");
        Ok(())
    }

    /// Send a JSON-RPC request and return the response.
    async fn send_request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let msg = serde_json::to_string(&request)?;
        let header = format!("Content-Length: {}\r\n\r\n", msg.len());

        self.stdin.write_all(header.as_bytes()).await?;
        self.stdin.write_all(msg.as_bytes()).await?;
        self.stdin.flush().await?;

        self.read_response(id).await
    }

    /// Read LSP response from stdout, skipping server notifications.
    /// Loops until a message with matching id is found.
    async fn read_response(&mut self, expected_id: u64) -> Result<Value> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            if std::time::Instant::now() > deadline {
                return Err(anyhow!("LSP response timeout: no matching response for request id {} within 30s", expected_id));
            }

            let mut content_length = 0usize;
            let mut buffer = String::new();

            // Read headers
            loop {
                buffer.clear();
                let bytes = self.stdout.read_line(&mut buffer).await?;
                if bytes == 0 {
                    return Err(anyhow!("LSP server closed connection"));
                }
                let line = buffer.trim();
                if line.is_empty() {
                    break;
                }
                if let Some(len_str) = line.strip_prefix("Content-Length: ") {
                    content_length = len_str.trim().parse()?;
                }
            }

            if content_length == 0 {
                return Err(anyhow!("No Content-Length header in LSP response"));
            }

            // Read body through BufReader (don't bypass buffer)
            let mut body = vec![0u8; content_length];
            self.stdout.read_exact(&mut body).await?;

            let message: Value = serde_json::from_slice(&body)?;

            // If this message has no id, it's a notification — skip it
            if message.get("id").is_none() {
                continue;
            }

            // If the id doesn't match, skip it (stale response from prior request)
            if message.get("id").and_then(|i| i.as_u64()) != Some(expected_id) {
                continue;
            }

            // Check for error
            if let Some(error) = message.get("error") {
                warn!(?error, "LSP error response");
                return Ok(Value::Null);
            }

            return Ok(message.get("result").cloned().unwrap_or(Value::Null));
        }
    }

    /// Send a notification (no response expected).
    async fn send_notification(&mut self, method: &str, params: Value) -> Result<()> {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });

        let msg = serde_json::to_string(&notification)?;
        let header = format!("Content-Length: {}\r\n\r\n", msg.len());
        self.stdin.write_all(header.as_bytes()).await?;
        self.stdin.write_all(msg.as_bytes()).await?;
        self.stdin.flush().await?;
        Ok(())
    }
}

/// Real LSP client that manages connections to multiple language servers.
///
/// Uses `std::sync::Mutex` for the server pool because the lock is only held
/// for brief remove/insert operations — never during I/O. The async I/O on
/// individual LspConnection instances happens outside the lock.
///
/// Timeouts: every LSP operation is wrapped in `tokio::time::timeout`.
/// If a timeout fires, the connection's async future is dropped, which drops
/// the Child process (sends SIGKILL), and the connection is not re-inserted
/// into the pool. Subsequent calls will return empty results and ensure_server
/// will re-spawn a fresh server on demand.
pub struct LspClient {
    workspace_root: PathBuf,
    servers: Arc<Mutex<HashMap<String, LspConnection>>>,
}

impl Clone for LspClient {
    fn clone(&self) -> Self {
        // Share the same server pool via Arc so that cloned instances
        // (e.g. through ContextEngine derive) retain LSP connections.
        Self {
            workspace_root: self.workspace_root.clone(),
            servers: Arc::clone(&self.servers),
        }
    }
}

impl LspClient {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            workspace_root,
            servers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Ensure a language server is running for the given language.
    /// Spawns the server process using tokio::process::Command and
    /// sends the LSP initialize handshake.
    pub async fn ensure_server(&self, language: &str) -> Result<()> {
        // Quick check: already running?
        {
            let servers = self.servers.lock().unwrap();
            if servers.contains_key(language) {
                return Ok(());
            }
        }

        let lang = language.to_string();
        let workspace = self.workspace_root.clone();

        let (bin, args): (&str, &[&str]) = match lang.as_str() {
            "python" => ("pylsp", &[] as &[&str]),
            "rust" => ("rust-analyzer", &[]),
            "typescript" | "javascript" => {
                if which::which("typescript-language-server").is_ok() {
                    ("typescript-language-server", &["--stdio"] as &[&str])
                } else {
                    warn!("typescript-language-server not found, LSP for TS/JS disabled");
                    return Ok(());
                }
            }
            _ => {
                warn!(language = lang.as_str(), "No LSP server configured for language");
                return Ok(());
            }
        };

        if which::which(bin).is_err() {
            warn!(binary = bin, language = lang.as_str(), "LSP server not installed");
            return Ok(());
        }

        // Spawn the process (tokio::process::Command is async-friendly)
        let mut cmd = Command::new(bin);
        cmd.args(args);
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::null());
        cmd.current_dir(&workspace);
        cmd.kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow!("Failed to spawn {} LSP server: {}", lang, e))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("Failed to get stdin for LSP server"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Failed to get stdout for LSP server"))?;

        let root_uri = Url::from_directory_path(&workspace)
            .map(|u| u.to_string())
            .unwrap_or_else(|_| format!("file://{}", workspace.display()));

        let mut conn = LspConnection {
            process: child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
            root_uri,
        };

        // Initialize handshake with timeout
        let init_result = timeout(Duration::from_secs(30), conn.initialize()).await;

        match init_result {
            Ok(Ok(())) => {
                let mut servers = self.servers.lock().unwrap();
                servers.insert(lang.clone(), conn);
                info!(language = lang.as_str(), "LSP server started");
                Ok(())
            }
            Ok(Err(e)) => {
                warn!(language = lang.as_str(), error = %e, "LSP server init failed");
                let _ = conn.process.kill().await;
                Err(e)
            }
            Err(_elapsed) => {
                warn!(language = lang.as_str(), "LSP server init timed out");
                let _ = conn.process.kill().await;
                Err(anyhow!("LSP server {} initialization timed out after 30s", lang))
            }
        }
    }

    /// Notify the LSP server that a document has been opened.
    pub async fn open_document(
        &self,
        language: &str,
        path: &Path,
        content: &str,
    ) -> Result<()> {
        let lang = language.to_string();
        let path_buf = path.to_path_buf();
        let content = content.to_string();

        // Extract connection from pool (no lock held during I/O)
        let conn = {
            let mut servers = self.servers.lock().unwrap();
            servers.remove(&lang)
        };

        let mut conn = match conn {
            Some(c) => c,
            None => return Ok(()),
        };

        let result = timeout(Duration::from_secs(10), async {
            let uri = Url::from_file_path(&path_buf)
                .map(|u| u.to_string())
                .map_err(|_| anyhow!("Invalid file path for LSP"))?;

            let params = serde_json::json!({
                "textDocument": {
                    "uri": uri,
                    "languageId": lang,
                    "version": 1,
                    "text": content,
                }
            });

            conn.send_notification("textDocument/didOpen", params).await
        }).await;

        match result {
            Ok(Ok(())) => {
                // Re-insert connection
                let mut servers = self.servers.lock().unwrap();
                servers.insert(lang, conn);
                Ok(())
            }
            Ok(Err(e)) => {
                let _ = conn.process.kill().await;
                Err(e)
            }
            Err(_elapsed) => {
                // Timeout — connection dropped (process killed)
                warn!(language = lang.as_str(), "LSP open_document timed out, server will restart on demand");
                Err(anyhow!("LSP open_document timed out"))
            }
        }
    }

    /// Notify the LSP server that a document has changed.
    pub async fn did_change(
        &self,
        language: &str,
        path: &Path,
        content: &str,
        version: i32,
    ) -> Result<()> {
        let lang = language.to_string();
        let path_buf = path.to_path_buf();
        let content = content.to_string();

        // Extract connection from pool
        let conn = {
            let mut servers = self.servers.lock().unwrap();
            servers.remove(&lang)
        };

        let mut conn = match conn {
            Some(c) => c,
            None => return Ok(()),
        };

        let result = timeout(Duration::from_secs(10), async {
            let uri = Url::from_file_path(&path_buf)
                .map(|u| u.to_string())
                .map_err(|_| anyhow!("Invalid file path for LSP"))?;

            let params = serde_json::json!({
                "textDocument": {
                    "uri": uri,
                    "version": version,
                },
                "contentChanges": [{
                    "text": content,
                }]
            });

            conn.send_notification("textDocument/didChange", params).await
        }).await;

        match result {
            Ok(Ok(())) => {
                let mut servers = self.servers.lock().unwrap();
                servers.insert(lang, conn);
                Ok(())
            }
            Ok(Err(e)) => {
                let _ = conn.process.kill().await;
                Err(e)
            }
            Err(_elapsed) => {
                warn!(language = lang.as_str(), "LSP did_change timed out, server will restart on demand");
                Err(anyhow!("LSP did_change timed out"))
            }
        }
    }

    /// Find all references to the symbol at the given position.
    pub async fn find_references(
        &self,
        language: &str,
        path: &Path,
        line: u32,
        column: u32,
        include_declaration: bool,
    ) -> Result<Vec<LspLocation>> {
        let lang = language.to_string();
        let path_buf = path.to_path_buf();

        let conn = {
            let mut servers = self.servers.lock().unwrap();
            servers.remove(&lang)
        };

        let mut conn = match conn {
            Some(c) => c,
            None => return Ok(Vec::new()),
        };

        let result = timeout(Duration::from_secs(30), async {
            let uri = Url::from_file_path(&path_buf)
                .map_err(|_| anyhow!("Invalid file path for LSP"))?;

            let params = serde_json::json!({
                "textDocument": { "uri": uri.to_string() },
                "position": { "line": line, "character": column },
                "context": { "includeDeclaration": include_declaration }
            });

            let result = conn.send_request("textDocument/references", params).await?;
            if result.is_null() {
                return Ok(Vec::new());
            }
            let locations: Vec<LspLocation> = serde_json::from_value(result).unwrap_or_default();
            Ok(locations)
        }).await;

        match result {
            Ok(Ok(locations)) => {
                let mut servers = self.servers.lock().unwrap();
                servers.insert(lang, conn);
                Ok(locations)
            }
            Ok(Err(e)) => {
                let _ = conn.process.kill().await;
                Err(e)
            }
            Err(_elapsed) => {
                warn!(language = lang.as_str(), "LSP find_references timed out, server will restart on demand");
                Err(anyhow!("LSP find_references timed out after 30s"))
            }
        }
    }

    /// Get hover information at a position.
    pub async fn get_hover(
        &self,
        language: &str,
        path: &Path,
        line: u32,
        column: u32,
    ) -> Result<Option<String>> {
        let lang = language.to_string();
        let path_buf = path.to_path_buf();

        let conn = {
            let mut servers = self.servers.lock().unwrap();
            servers.remove(&lang)
        };

        let mut conn = match conn {
            Some(c) => c,
            None => return Ok(None),
        };

        let result = timeout(Duration::from_secs(15), async {
            let uri = Url::from_file_path(&path_buf)
                .map_err(|_| anyhow!("Invalid file path for LSP"))?;

            let params = serde_json::json!({
                "textDocument": { "uri": uri.to_string() },
                "position": { "line": line, "character": column }
            });

            let result = conn.send_request("textDocument/hover", params).await?;
            if result.is_null() {
                return Ok(None);
            }

            if let Some(contents) = result.get("contents") {
                match contents {
                    Value::String(s) => return Ok(Some(s.clone())),
                    Value::Object(obj) => {
                        if let Some(value) = obj.get("value") {
                            return Ok(Some(value.as_str().unwrap_or("").to_string()));
                        }
                    }
                    _ => {}
                }
            }
            Ok(None)
        }).await;

        match result {
            Ok(Ok(hover)) => {
                let mut servers = self.servers.lock().unwrap();
                servers.insert(lang, conn);
                Ok(hover)
            }
            Ok(Err(e)) => {
                let _ = conn.process.kill().await;
                Err(e)
            }
            Err(_elapsed) => {
                warn!(language = lang.as_str(), "LSP get_hover timed out, server will restart on demand");
                Err(anyhow!("LSP get_hover timed out after 15s"))
            }
        }
    }

    /// Go to definition of the symbol at the given position.
    pub async fn get_definition(
        &self,
        language: &str,
        path: &Path,
        line: u32,
        column: u32,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let lang = language.to_string();
        let path_buf = path.to_path_buf();

        let conn = {
            let mut servers = self.servers.lock().unwrap();
            servers.remove(&lang)
        };

        let mut conn = match conn {
            Some(c) => c,
            None => return Ok(None),
        };

        let result = timeout(Duration::from_secs(15), async {
            let uri = Url::from_file_path(&path_buf)
                .map_err(|_| anyhow!("Invalid file path for LSP"))?;

            let params = serde_json::json!({
                "textDocument": { "uri": uri.to_string() },
                "position": { "line": line, "character": column }
            });

            let result = conn.send_request("textDocument/definition", params).await?;
            if result.is_null() {
                return Ok(None);
            }

            if let Ok(loc) = serde_json::from_value::<LspLocation>(result.clone()) {
                return Ok(Some(GotoDefinitionResponse::Scalar(loc)));
            }
            if let Ok(locs) = serde_json::from_value::<Vec<LspLocation>>(result.clone()) {
                return Ok(Some(GotoDefinitionResponse::Array(locs)));
            }
            Ok(None)
        }).await;

        match result {
            Ok(Ok(def)) => {
                let mut servers = self.servers.lock().unwrap();
                servers.insert(lang, conn);
                Ok(def)
            }
            Ok(Err(e)) => {
                let _ = conn.process.kill().await;
                Err(e)
            }
            Err(_elapsed) => {
                warn!(language = lang.as_str(), "LSP get_definition timed out, server will restart on demand");
                Err(anyhow!("LSP get_definition timed out after 15s"))
            }
        }
    }

    /// Get completion suggestions at a position.
    pub async fn get_completions(
        &self,
        language: &str,
        path: &Path,
        line: u32,
        column: u32,
    ) -> Result<Vec<LspCompletionItem>> {
        let lang = language.to_string();
        let path_buf = path.to_path_buf();

        let conn = {
            let mut servers = self.servers.lock().unwrap();
            servers.remove(&lang)
        };

        let mut conn = match conn {
            Some(c) => c,
            None => return Ok(Vec::new()),
        };

        let result = timeout(Duration::from_secs(15), async {
            let uri = Url::from_file_path(&path_buf)
                .map_err(|_| anyhow!("Invalid file path for LSP"))?;

            let params = serde_json::json!({
                "textDocument": { "uri": uri.to_string() },
                "position": { "line": line, "character": column }
            });

            let result = conn.send_request("textDocument/completion", params).await?;
            if result.is_null() {
                return Ok(Vec::new());
            }

            if let Ok(items) = serde_json::from_value::<Vec<LspCompletionItem>>(result.clone()) {
                return Ok(items);
            }
            if let Some(items) = result.get("items") {
                if let Ok(items) = serde_json::from_value::<Vec<LspCompletionItem>>(items.clone()) {
                    return Ok(items);
                }
            }
            Ok(Vec::new())
        }).await;

        match result {
            Ok(Ok(completions)) => {
                let mut servers = self.servers.lock().unwrap();
                servers.insert(lang, conn);
                Ok(completions)
            }
            Ok(Err(e)) => {
                let _ = conn.process.kill().await;
                Err(e)
            }
            Err(_elapsed) => {
                warn!(language = lang.as_str(), "LSP get_completions timed out, server will restart on demand");
                Err(anyhow!("LSP get_completions timed out after 15s"))
            }
        }
    }
}

/// Check if a command exists in PATH and is executable.
/// Shadows the `which` crate to avoid an extra dependency.
mod which {
    use std::path::PathBuf;

    fn is_executable(path: &std::path::Path) -> bool {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(metadata) = path.metadata() {
                let mode = metadata.permissions().mode();
                // Check owner/group/other execute bit
                return mode & 0o111 != 0;
            }
            false
        }
        #[cfg(not(unix))]
        {
            path.is_file()
        }
    }

    pub fn which(cmd: &str) -> Result<PathBuf, ()> {
        let path_var = std::env::var("PATH").unwrap_or_default();
        for dir in path_var.split(':') {
            let full = std::path::Path::new(dir).join(cmd);
            if full.is_file() && is_executable(&full) {
                return Ok(full);
            }
        }
        Err(())
    }
}

/// Convert an `lsp_types::Location` into the engine's simpler representation.
pub fn location_to_lsp_location(loc: LspLocation) -> ContextLspLocation {
    let file_path = loc
        .uri
        .to_file_path()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    ContextLspLocation {
        file_path,
        line: loc.range.start.line + 1,
        column: loc.range.start.character + 1,
    }
}

/// Convert an `lsp_types::CompletionItem` into the engine's representation.
pub fn completion_to_completion_item(item: LspCompletionItem) -> CompletionItem {
    CompletionItem {
        label: item.label,
        kind: item
            .kind
            .map(|k| format!("{:?}", k))
            .unwrap_or_else(|| "Unknown".to_string()),
        detail: item.detail,
    }
}
