//! Context & Memory Engine - Read optimization, RAG, and LSP integration
//!
//! Provides:
//! - get_skeleton: Uses Tree-sitter to return only imports, class names, function signatures, docstrings
//! - find_relevant_code: Queries local Vector DB for semantic search
//! - LSP tools: find_references, get_hover, get_definition, complete

use anyhow::{anyhow, Result};
use regex::Regex;
use std::path::{Path, PathBuf};
use std::fs;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::tree_sitter_utils::{
    TreeSitterParser, SourceLanguage,
};
use crate::lsp_client::{LspClient, location_to_lsp_location, completion_to_completion_item};
use crate::vector_db::{self, VectorDB};
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub file_path: String,
    pub function_name: Option<String>,
    pub class_name: Option<String>,
    pub line_number: usize,
    pub relevance_score: f32,
    pub snippet: String,
}

#[derive(Clone)]
pub struct ContextEngine {
    vector_db: Arc<RwLock<VectorDB>>,
    /// LspClient uses internal `Arc<Mutex<HashMap<...>>>` (std::sync::Mutex) for
    /// thread safety. No outer tokio::Mutex is needed — that would serialize
    /// all LSP operations (including timeouts) unnecessarily across different languages.
    lsp_client: Arc<LspClient>,
    workspace_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspLocation {
    pub file_path: String,
    pub line: u32,
    pub column: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionItem {
    pub label: String,
    pub kind: String,
    pub detail: Option<String>,
}

impl ContextEngine {
    pub fn new() -> Self {
        Self {
            vector_db: Arc::new(RwLock::new(VectorDB::new(PathBuf::from("/tmp/warden-vector-db")))),
            lsp_client: Arc::new(LspClient::new(PathBuf::from("/"))),
            workspace_root: None,
        }
    }
    
    pub fn with_workspace(mut self, workspace_path: PathBuf) -> Self {
        self.workspace_root = Some(workspace_path.clone());
        self.lsp_client = Arc::new(LspClient::new(workspace_path.clone()));
        
        // Re-create VectorDB with workspace-specific path
        let db_path = workspace_path.join(".warden").join("vector-db");
        self.vector_db = Arc::new(RwLock::new(VectorDB::new(db_path)));
        
        self
    }
    
    /// Initialize the VectorDB with embedding model
    pub async fn initialize_vector_db(&self, model_path: &Path, tokenizer_path: &Path) -> Result<()> {
        let mut db = self.vector_db.write().await;
        db.initialize(model_path, tokenizer_path)?;
        info!("VectorDB initialized with embedding model");
        Ok(())
    }
    
    /// Check if VectorDB is initialized
    pub async fn is_vector_db_initialized(&self) -> bool {
        self.vector_db.read().await.is_initialized()
    }

    /// Ensure an LSP server is running for the given language (no-op if already running).
    /// Public so that McpRouter can eagerly initialize servers before the first tool call.
    pub async fn ensure_lsp_server(&self, language: &str) -> Result<()> {
        self.lsp_client.ensure_server(language).await
    }
    


    /// Initialize VectorDB and automatically index all workspace files
    /// 
    /// This combines initialization with full workspace indexing for RAG.
    /// Returns the number of files indexed.
    pub async fn initialize_vector_db_with_indexing(
        &self, 
        model_path: &Path, 
        tokenizer_path: &Path,
    ) -> Result<usize> {
        // First initialize the VectorDB
        self.initialize_vector_db(model_path, tokenizer_path).await?;
        
        // Then index the entire workspace
        let count = self.index_workspace().await?;
        
        info!(indexed_files = count, "VectorDB initialization with workspace indexing complete");
        Ok(count)
    }

    /// Index all code files in the workspace for semantic search
    pub async fn index_workspace(&self) -> Result<usize> {
        let workspace_root = match &self.workspace_root {
            Some(path) => path.clone(),
            None => {
                info!("No workspace root set, skipping indexing");
                return Ok(0);
            }
        };

        // Check if VectorDB is initialized
        if !self.is_vector_db_initialized().await {
            warn!("VectorDB not initialized, skipping workspace indexing. \
                   Call initialize_vector_db() first.");
            return Ok(0);
        }

        info!(path = %workspace_root.display(), "Starting workspace indexing");
        
        let mut indexed_count = 0;
        let mut skipped_count = 0;
        let supported_extensions: std::collections::HashSet<&str> = 
            vec!["py", "js", "jsx", "ts", "tsx", "rs", "go", "java", "cpp", "c", "h", "hpp", "cs", "rb", "php", "swift", "kt"].into_iter().collect();
        
        // Walk the workspace directory
        for entry in WalkDir::new(&workspace_root)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            
            // Check if file has a supported extension
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if !supported_extensions.contains(ext) {
                    skipped_count += 1;
                    continue;
                }
            } else {
                skipped_count += 1;
                continue;
            }
            
            // Skip hidden files and directories (starting with .)
            if path.components().any(|c| {
                match c {
                    std::path::Component::Normal(s) => {
                        let s_str = s.to_string_lossy();
                        s_str.starts_with('.') || s_str == "target" || s_str == "node_modules" || s_str == "__pycache__"
                    }
                    _ => false,
                }
            }) {
                skipped_count += 1;
                continue;
            }
            
            // Try to read and index the file
            match fs::read_to_string(path) {
                Ok(content) => {
                    match self.index_content(path, &content).await {
                        Ok(_) => {
                            indexed_count += 1;
                            if indexed_count % 10 == 0 {
                                info!(count = indexed_count, "Files indexed so far");
                            }
                        }
                        Err(e) => {
                            warn!(path = %path.display(), error = %e, "Failed to index file");
                            skipped_count += 1;
                        }
                    }
                }
                Err(e) => {
                    debug!(path = %path.display(), error = %e, "Could not read file");
                    skipped_count += 1;
                }
            }
        }
        
        info!(
            indexed = indexed_count, 
            skipped = skipped_count, 
            "Workspace indexing complete"
        );
        
        Ok(indexed_count)
    }
    
    /// Get the number of indexed chunks in the VectorDB
    pub async fn get_indexed_chunk_count(&self) -> usize {
        self.vector_db.read().await.len()
    }

    /// Initialize LSP servers for detected languages in workspace
    pub async fn initialize_lsp_servers(&mut self) -> Result<()> {
        let workspace_root = match &self.workspace_root {
            Some(path) => path.clone(),
            None => return Ok(()),
        };

        // Detect languages by scanning workspace files recursively (not just top-level)
        let mut detected_languages = std::collections::HashSet::new();

        for entry in WalkDir::new(&workspace_root).max_depth(3).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let path = entry.path();
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    let lang = match ext {
                        "py" => "python",
                        "js" | "jsx" => "javascript",
                        "ts" | "tsx" => "typescript",
                        "rs" => "rust",
                            _ => continue,
                    };
                    detected_languages.insert(lang.to_string());
                }
            }
        }

        // Initialize servers for detected languages
        for lang in detected_languages {
            match self.lsp_client.ensure_server(&lang).await {
                Ok(_) => info!(language = %lang, "LSP server initialized"),
                Err(e) => warn!(language = %lang, error = %e, "Failed to initialize LSP server"),
            }
        }

        Ok(())
    }

    /// get_skeleton - Uses Tree-sitter to return only imports, class names, function signatures, and docstrings
    pub async fn get_skeleton(&self, filepath: PathBuf) -> Result<SkeletonResult> {
        let content = fs::read_to_string(&filepath)?;
        let ext = filepath.extension().and_then(|e| e.to_str()).unwrap_or("");
        let language = SourceLanguage::from_extension(ext);

        let mut parser = TreeSitterParser::new()?;
        let skeleton_data = parser.extract_skeleton(&content, language.clone())?;

        // Convert Tree-sitter types to SkeletonResult format
        let imports: Vec<String> = skeleton_data.imports.iter().map(|imp| {
            if imp.names.is_empty() {
                format!("import {}", imp.module)
            } else {
                format!("import {} from {}", imp.names.join(", "), imp.module)
            }
        }).collect();

        let functions: Vec<String> = skeleton_data.functions.iter().map(|func| {
            if let Some(rt) = &func.return_type {
                format!("fn {}({}) -> {}", func.name, func.parameters, rt)
            } else {
                format!("fn {}({})", func.name, func.parameters)
            }
        }).collect();

        let classes: Vec<String> = skeleton_data.classes.iter().map(|cls| {
            format!("class {}", cls.name)
        }).collect();

        let docstrings: Vec<String> = skeleton_data.docstrings.iter().map(|doc| {
            doc.content.clone()
        }).collect();

        let skeleton_lines = imports.len() + functions.len() + classes.len();

        Ok(SkeletonResult {
            file_path: filepath.to_string_lossy().to_string(),
            imports,
            functions,
            classes,
            docstrings,
            total_lines: content.lines().count(),
            skeleton_lines,
        })
    }

    /// find_relevant_code - Queries local Vector DB using semantic search.
    /// Falls back to keyword-grep when the ONNX model is unavailable.
    pub async fn find_relevant_code(&self, issue_text: &str) -> Result<Vec<SearchResult>> {
        // If VectorDB is not initialized (e.g. ONNX model files not found),
        // use a simple keyword-based grep fallback so the EXPLORE phase still
        // works rather than returning a hard error.
        if !self.is_vector_db_initialized().await {
            warn!("VectorDB not initialized, falling back to keyword search");
            return self.fallback_keyword_search(issue_text).await;
        }

        info!(query = issue_text, "Searching for relevant code via ONNX VectorDB");
        
        let results = self.vector_db.read().await.query(issue_text, 5).await?;
        
        let mut search_results = Vec::new();
        for r in results {
            // Use Tree-sitter to extract the real function/class name from the chunk
            let (function_name, class_name) = self.extract_chunk_names(&r.file_path, &r.chunk_text, &r.chunk_type);
            
            search_results.push(SearchResult {
                file_path: r.file_path,
                function_name,
                class_name,
                line_number: r.start_line,
                relevance_score: r.relevance_score,
                snippet: r.chunk_text.chars().take(200).collect(),
            });
        }
        
        Ok(search_results)
    }

    /// Fallback keyword-grep search when VectorDB has no embedding model.
    /// Splits the query into words, greps each word in the workspace, and
    /// ranks files by total keyword matches.
    async fn fallback_keyword_search(&self, query: &str) -> Result<Vec<SearchResult>> {
        let workspace_root = match &self.workspace_root {
            Some(path) => path.clone(),
            None => return Ok(Vec::new()),
        };

        // Split query into lowercase keywords, filter out short/common words
        let keywords: Vec<String> = query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3)
            .map(String::from)
            .collect();

        if keywords.is_empty() {
            return Ok(Vec::new());
        }

        let supported_extensions: std::collections::HashSet<&str> = vec![
            "py", "js", "jsx", "ts", "tsx", "rs", "go", "java", "cpp", "c",
            "h", "hpp", "cs", "rb", "php", "swift", "kt",
        ]
        .into_iter()
        .collect();

        // (file_path, total_matches, best_snippet, best_line)
        let mut file_hits: std::collections::HashMap<String, (usize, String, usize)> =
            std::collections::HashMap::new();

        for entry in WalkDir::new(&workspace_root)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();

            // Only search supported extensions
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !supported_extensions.contains(ext) {
                continue;
            }

            // Skip hidden/dependency directories
            if path.components().any(|c| match c {
                std::path::Component::Normal(s) => {
                    let s_str = s.to_string_lossy();
                    s_str.starts_with('.')
                        || s_str == "target"
                        || s_str == "node_modules"
                        || s_str == "__pycache__"
                }
                _ => false,
            }) {
                continue;
            }

            if let Ok(content) = fs::read_to_string(path) {
                let mut file_match_count = 0usize;
                let mut first_match_line = 0usize;
                let mut first_match_text = String::new();

                for (line_no, line) in content.lines().enumerate() {
                    let lower = line.to_lowercase();
                    for kw in &keywords {
                        if lower.contains(kw.as_str()) {
                            file_match_count += 1;
                            if first_match_line == 0 {
                                first_match_line = line_no + 1;
                                first_match_text = line.to_string();
                            }
                            break; // count each line at most once per file
                        }
                    }
                }

                if file_match_count > 0 {
                    let file_path_str = path.to_string_lossy().to_string();
                    file_hits.insert(
                        file_path_str,
                        (file_match_count, first_match_text, first_match_line),
                    );
                }
            }
        }

        // Sort by match count descending, take top 5
        let mut sorted: Vec<(String, (usize, String, usize))> =
            file_hits.into_iter().collect();
        sorted.sort_by(|a, b| b.1.0.cmp(&a.1.0));
        sorted.truncate(5);

        let max_hits = sorted.first().map(|(_, (c, _, _))| *c as f32).unwrap_or(1.0);

        let results: Vec<SearchResult> = sorted
            .into_iter()
            .map(|(path, (hits, snippet, line))| SearchResult {
                file_path: path,
                function_name: None,
                class_name: None,
                line_number: line,
                relevance_score: (hits as f32 / max_hits).clamp(0.0, 1.0),
                snippet: snippet.chars().take(200).collect(),
            })
            .collect();

        info!(
            keywords = keywords.len(),
            results = results.len(),
            "Keyword fallback search complete"
        );

        Ok(results)
    }

    /// Extract function/class names from a code chunk using Tree-sitter
    fn extract_chunk_names(&self, file_path: &str, chunk_text: &str, chunk_type: &vector_db::ChunkType) -> (Option<String>, Option<String>) {
        let ext = std::path::Path::new(file_path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let language = SourceLanguage::from_extension(ext);

        let first_line = chunk_text.lines().next().unwrap_or("").trim().to_string();

        // Try Tree-sitter first
        if let Ok(mut parser) = TreeSitterParser::new() {
            if let Some(lang) = language.get_language() {
                if parser.set_language(lang).is_ok() {
                    if let Ok(tree) = parser.parse(chunk_text) {
                        let root = tree.root_node();
                        let found = self.find_definition_name(&root, chunk_text, &language);
                        if let Some(name) = found {
                            match chunk_type {
                                vector_db::ChunkType::Function => return (Some(name), None),
                                vector_db::ChunkType::Class => return (None, Some(name)),
                                _ => return (Some(name), None),
                            }
                        }
                    }
                }
            }
        }

        // Fallback: use first non-empty line
        let fallback = if !first_line.is_empty() && first_line.len() < 100 {
            Some(first_line)
        } else {
            None
        };

        match chunk_type {
            vector_db::ChunkType::Function => (fallback, None),
            vector_db::ChunkType::Class => (None, fallback),
            _ => (None, None),
        }
    }

    /// Walk the tree-sitter AST to find the first definition name
    fn find_definition_name(&self, node: &tree_sitter::Node, content: &str, language: &SourceLanguage) -> Option<String> {
        let kind = node.kind();
        
        match language {
            SourceLanguage::Python => {
                if kind == "function_definition" || kind == "class_definition" {
                    for i in 0..node.child_count() {
                        if let Some(child) = node.child(i) {
                            if child.kind() == "identifier" {
                                return Some(content[child.start_byte()..child.end_byte()].to_string());
                            }
                        }
                    }
                }
            }
            SourceLanguage::Rust => {
                if kind == "function_item" || kind == "struct_item" || kind == "enum_item" {
                    for i in 0..node.child_count() {
                        if let Some(child) = node.child(i) {
                            if child.kind() == "identifier" {
                                return Some(content[child.start_byte()..child.end_byte()].to_string());
                            }
                        }
                    }
                }
            }
            SourceLanguage::JavaScript | SourceLanguage::TypeScript => {
                if kind == "function_declaration" || kind == "class_declaration" || kind == "method_definition" {
                    for i in 0..node.child_count() {
                        if let Some(child) = node.child(i) {
                            if child.kind() == "identifier" {
                                return Some(content[child.start_byte()..child.end_byte()].to_string());
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        // Recurse into children
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if let Some(found) = self.find_definition_name(&child, content, language) {
                    return Some(found);
                }
            }
        }
        
        None
    }

    /// Index a file for RAG search
    pub async fn index_file(&self, file_path: &Path) -> Result<()> {
        let content = fs::read_to_string(file_path)?;
        self.vector_db.write().await.index_file(file_path, &content).await?;
        info!(file = %file_path.display(), "Indexed file for RAG");
        Ok(())
    }
    
    /// Index a file with given content
    pub async fn index_content(&self, file_path: &Path, content: &str) -> Result<()> {
        self.vector_db.read().await.index_file(file_path, content).await?;
        info!(file = %file_path.display(), "Indexed content for RAG");
        Ok(())
    }

    // ==================== LSP methods ====================

    /// Open a document in the LSP server for a language
    pub async fn lsp_open_document(&self, file_path: &Path, content: &str) -> Result<()> {
        let lang = self.detect_language_from_path(file_path)?;
        
        // Read file content if not provided
        let file_content = if content.is_empty() {
            fs::read_to_string(file_path)?
        } else {
            content.to_string()
        };
        
        self.lsp_client.open_document(&lang, file_path, &file_content).await
            .map_err(|e| anyhow!(e))?;
        Ok(())
    }

    /// Find all references to a symbol
    pub async fn lsp_find_references(&self, file_path: &str, line: u32, column: u32, include_declaration: bool) -> Result<Vec<LspLocation>> {
        let lang = self.detect_language_from_path(Path::new(file_path))?;
        
        let path = Path::new(file_path);
        
        // LSP uses 0-indexed lines, but we use 1-indexed
        let lsp_line = line.saturating_sub(1);
        let lsp_col = column.saturating_sub(1);
        
        let locations = self.lsp_client.find_references(&lang, path, lsp_line, lsp_col, include_declaration).await
            .map_err(|e| anyhow!(e))?;
        
        Ok(locations.into_iter().map(location_to_lsp_location).collect())
    }

    /// Get hover information at a position
    pub async fn lsp_get_hover(&self, file_path: &str, line: u32, column: u32) -> Result<Option<String>> {
        let lang = self.detect_language_from_path(Path::new(file_path))?;
        
        let path = Path::new(file_path);
        
        // LSP uses 0-indexed lines
        let lsp_line = line.saturating_sub(1);
        let lsp_col = column.saturating_sub(1);
        
        let hover = self.lsp_client.get_hover(&lang, path, lsp_line, lsp_col).await
            .map_err(|e| anyhow!(e))?;
        Ok(hover)
    }

    /// Go to definition of a symbol
    pub async fn lsp_get_definition(&self, file_path: &str, line: u32, column: u32) -> Result<Vec<LspLocation>> {
        let lang = self.detect_language_from_path(Path::new(file_path))?;
        
        let path = Path::new(file_path);
        
        // LSP uses 0-indexed lines
        let lsp_line = line.saturating_sub(1);
        let lsp_col = column.saturating_sub(1);
        
        let result = self.lsp_client.get_definition(&lang, path, lsp_line, lsp_col).await
            .map_err(|e| anyhow!(e))?;
        
        match result {
            Some(lsp_types::GotoDefinitionResponse::Scalar(location)) => {
                Ok(vec![location_to_lsp_location(location)])
            }
            Some(lsp_types::GotoDefinitionResponse::Array(locations)) => {
                Ok(locations.into_iter().map(location_to_lsp_location).collect())
            }
            Some(lsp_types::GotoDefinitionResponse::Link(locations)) => {
                Ok(locations.into_iter().map(|loc| LspLocation {
                    file_path: loc.target_uri.to_file_path()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_default(),
                    line: loc.target_range.start.line + 1,
                    column: loc.target_range.start.character + 1,
                }).collect())
            }
            None => Ok(vec![]),
        }
    }

    /// Get completion suggestions at a position
    pub async fn lsp_complete(&self, file_path: &str, line: u32, column: u32) -> Result<Vec<CompletionItem>> {
        let lang = self.detect_language_from_path(Path::new(file_path))?;
        
        let path = Path::new(file_path);
        
        // LSP uses 0-indexed lines
        let lsp_line = line.saturating_sub(1);
        let lsp_col = column.saturating_sub(1);
        
        let completions = self.lsp_client.get_completions(&lang, path, lsp_line, lsp_col).await
            .map_err(|e| anyhow!(e))?;
        
        Ok(completions.into_iter().map(completion_to_completion_item).collect())
    }

    /// Notify LSP server of document changes
    pub async fn lsp_did_change(&self, file_path: &Path, content: &str, version: i32) -> Result<()> {
        let lang = self.detect_language_from_path(file_path)?;
        self.lsp_client.did_change(&lang, file_path, content, version).await
            .map_err(|e| anyhow!(e))?;
        Ok(())
    }

    fn detect_language_from_path(&self, path: &Path) -> Result<String> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        Ok(match ext {
            "py" => "python".to_string(),
            "js" | "jsx" => "javascript".to_string(),
            "ts" | "tsx" => "typescript".to_string(),
            "rs" => "rust".to_string(),
            _ => "unknown".to_string(),
        })
    }

    // ==================== File reading methods ====================

    pub async fn read_file(&self, path: &str, start_line: Option<usize>, end_line: Option<usize>) -> Result<String> {
        let resolved = self.resolve_workspace_path(path)?;
        let content = fs::read_to_string(&resolved)?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();
        let start = start_line.unwrap_or(1).saturating_sub(1);
        let end = end_line.unwrap_or(total_lines).min(total_lines);

        // If no range specified and file is very large, return skeleton + warning
        if start_line.is_none() && end_line.is_none() && total_lines > 500 {
            let skeleton = self.get_skeleton(resolved).await?;
            let skeleton_str = serde_json::to_string_pretty(&skeleton).unwrap_or_else(|_| String::from("Unable to generate skeleton"));
            return Ok(format!(
                "⚠️⚠️⚠️ TRUNCATION — File has {} lines (threshold: 500). Showing skeleton only.\n⚠️ Use start_line/end_line parameters to read specific sections.\n\n{}",
                total_lines,
                skeleton_str
            ));
        }

        if start >= end {
            return Ok(String::new());
        }

        Ok(lines[start..end].join("\n"))
    }

    pub async fn list_directory(&self, path: &str) -> Result<Vec<DirectoryEntry>> {
        let resolved = self.resolve_workspace_path(path)?;
        let mut entries = Vec::new();

        if resolved.is_dir() {
            for entry in fs::read_dir(&resolved)? {
                let entry = entry?;
                let file_type = if entry.file_type()?.is_dir() {
                    "directory"
                } else if entry.file_type()?.is_symlink() {
                    "symlink"
                } else {
                    "file"
                };

                entries.push(DirectoryEntry {
                    name: entry.file_name().to_string_lossy().to_string(),
                    path: entry.path().to_string_lossy().to_string(),
                    file_type: file_type.to_string(),
                });
            }
        }

        Ok(entries)
    }

    pub async fn search_files(&self, pattern: &str, path: &str, file_type: Option<&str>) -> Result<Vec<SearchMatch>> {
        let mut matches = Vec::new();
        let resolved = self.resolve_workspace_path(path)?;

        let re = Regex::new(pattern)
            .map_err(|e| anyhow!("Invalid regex pattern '{}': {}", pattern, e))?;

        for entry in walkdir::WalkDir::new(&resolved)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            if let Some(ft) = file_type {
                if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                    if ext != ft {
                        continue;
                    }
                }
            }

            if let Ok(content) = fs::read_to_string(entry.path()) {
                for (i, line) in content.lines().enumerate() {
                    if re.is_match(line) {
                        matches.push(SearchMatch {
                            file_path: entry.path().to_string_lossy().to_string(),
                            line_number: i + 1,
                            line_content: line.to_string(),
                        });
                    }
                }
            }
        }

        Ok(matches)
    }

    /// Validate and resolve a path to be within the workspace root
    fn resolve_workspace_path(&self, path: &str) -> Result<PathBuf> {
        let workspace = self.workspace_root.as_ref()
            .ok_or_else(|| anyhow!("No workspace configured"))?;

        let resolved = if Path::new(path).is_absolute() {
            PathBuf::from(path)
        } else {
            workspace.join(path)
        };

        // Canonicalize to resolve .. and symlinks
        let canonical = resolved.canonicalize()
            .unwrap_or_else(|_| resolved.clone());

        let workspace_canonical = workspace.canonicalize()
            .unwrap_or_else(|_| workspace.clone());

        if !canonical.starts_with(&workspace_canonical) {
            return Err(anyhow!(
                "Path '{}' is outside the workspace. All file operations must be within the workspace directory.",
                path
            ));
        }

        Ok(canonical)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletonResult {
    pub file_path: String,
    pub imports: Vec<String>,
    pub functions: Vec<String>,
    pub classes: Vec<String>,
    pub docstrings: Vec<String>,
    pub total_lines: usize,
    pub skeleton_lines: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryEntry {
    pub name: String,
    pub path: String,
    pub file_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMatch {
    pub file_path: String,
    pub line_number: usize,
    pub line_content: String,
}