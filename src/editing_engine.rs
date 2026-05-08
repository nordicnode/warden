//! Editing Engine - Safe code modifications via real AST operations
//!
//! No raw text replacements. The LLM cannot overwrite whole files.
//! - modify_ast: Uses tree-sitter to locate and replace actual AST nodes
//! - apply_unified_diff: Accepts strictly formatted unified diffs and applies them
//! - All patch operations execute inside bwrap sandbox for security

use anyhow::{Result, Context};
use std::path::PathBuf;
use std::fs;
use tracing::info;

use crate::sandbox::{SandboxExecutor, SandboxConfig};
use crate::tree_sitter_utils::{TreeSitterParser, SourceLanguage};

#[derive(Debug, Clone, serde::Serialize)]
pub struct DiffResult {
    pub changed_files: Vec<PathBuf>,
    pub diff_output: String,
}

pub struct EditingEngine {
    workspace_path: Option<PathBuf>,
    sandbox_executor: Option<SandboxExecutor>,
}

// Manual Clone impl needed because Clone is not derived for SandboxExecutor in all cases
impl Clone for EditingEngine {
    fn clone(&self) -> Self {
        Self {
            workspace_path: self.workspace_path.clone(),
            sandbox_executor: self.sandbox_executor.clone(),
        }
    }
}

impl EditingEngine {
    pub fn new() -> Self {
        Self { 
            workspace_path: None,
            sandbox_executor: None,
        }
    }

    pub fn with_workspace(mut self, workspace_path: PathBuf) -> Self {
        let executor = SandboxExecutor::new(workspace_path.clone());
        self.workspace_path = Some(workspace_path);
        self.sandbox_executor = Some(executor);
        self
    }

    /// Write content to a file (only for test files in REPRODUCE phase)
    /// SAFETY: Restricted to paths under tests/ or files with .test.* suffix.
    pub async fn write_file(&self, path: &str, content: &str) -> Result<DiffResult> {
        let workspace = self.workspace_path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No workspace configured"))?;

        // Safety check: only allow writes to test directories or .test.* files
        let path_lower = path.to_lowercase();
        let is_test_file = path.starts_with("tests/") 
            || path.starts_with("test/")
            || path.ends_with("_test.py")
            || path.ends_with("_test.rs")
            || path.ends_with(".test.py")
            || path.ends_with(".test.js")
            || path.ends_with(".test.ts")
            || path.ends_with(".test.rs")
            || path_lower.contains("/test_")
            || path_lower.contains("/tests/");
        
        if !is_test_file {
            return Err(anyhow::anyhow!(
                "write_file is restricted to test files. Path must be under tests/ or have .test.* suffix. Got: {}", 
                path
            ));
        }

        let full_path = workspace.join(path);

        // Canonicalize parent to check for path traversal
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)?;
            let canonical_parent = parent.canonicalize()?;
            let canonical_workspace = workspace.canonicalize()?;
            if !canonical_parent.starts_with(&canonical_workspace) {
                return Err(anyhow::anyhow!(
                    "Path traversal detected: '{}' resolves outside workspace", path
                ));
            }
        }

        // Write new content
        fs::write(&full_path, content)?;

        info!(path = path, "File written successfully");

        Ok(DiffResult {
            changed_files: vec![PathBuf::from(path)],
            diff_output: format!(
                "+{} lines written to {}",
                content.lines().count(),
                path
            ),
        })
    }

    /// modify_ast - Modifies code structurally by targeting specific AST nodes
    /// 
    /// Uses tree-sitter to find the exact AST node by type and name, then replaces
    /// only that node's span with the new code. This is real AST-based editing,
    /// not fragile line-by-line text matching.
    pub async fn modify_ast(
        &self,
        file: &str,
        node_type: &str,
        node_name: &str,
        new_code: &str,
    ) -> Result<DiffResult> {
        let workspace = self.workspace_path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No workspace configured"))?;

        let full_path = workspace.join(file);

        if !full_path.exists() {
            return Err(anyhow::anyhow!("File not found: {}", file));
        }

        let content = fs::read_to_string(&full_path)?;
        let ext = full_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let language = SourceLanguage::from_extension(ext);

        // Use real tree-sitter parsing to find and replace the AST node
        let (new_content, changed) = self.replace_ast_node(
            &content,
            &language,
            node_type,
            node_name,
            new_code,
        )?;

        if !changed {
            return Err(anyhow::anyhow!(
                "Could not find {} '{}' in file {}",
                node_type,
                node_name,
                file
            ));
        }

        // Write modified content
        fs::write(&full_path, &new_content)?;

        info!(file = file, node_type = node_type, node_name = node_name, "AST modification applied via tree-sitter");

        Ok(DiffResult {
            changed_files: vec![PathBuf::from(file)],
            diff_output: format!(
                "Modified {} {} in {} using tree-sitter AST",
                node_type,
                node_name,
                file
            ),
        })
    }

    /// Replace an AST node using tree-sitter for structural accuracy
    fn replace_ast_node(
        &self,
        content: &str,
        language: &SourceLanguage,
        node_type: &str,
        node_name: &str,
        new_code: &str,
    ) -> Result<(String, bool)> {
        let mut parser = TreeSitterParser::new()
            .context("Failed to create tree-sitter parser")?;

        let lang = match language.get_language() {
            Some(l) => l,
            None => return Err(anyhow::anyhow!("Unsupported language for AST editing")),
        };

        parser.set_language(lang)
            .context("Failed to set language for parser")?;

        let tree = parser.parse(content)
            .context("Failed to parse source for AST modification")?;

        let root = tree.root_node();
        
        // Find the target node by walking the AST
        let target_node = self.find_node_by_criteria(&root, content, node_type, node_name, language);
        
        if let Some(node) = target_node {
            // Found the node - replace its span with new code
            let start_byte = node.start_byte();
            let end_byte = node.end_byte();
            
            let mut result = String::new();
            result.push_str(&content[..start_byte]);
            result.push_str(new_code);
            result.push_str(&content[end_byte..]);
            
            return Ok((result, true));
        }
        
        Ok((content.to_string(), false))
    }

    /// Find a node matching the given criteria (type and name)
    fn find_node_by_criteria<'a>(
        &self,
        node: &tree_sitter::Node<'a>,
        content: &str,
        node_type: &str,
        node_name: &str,
        language: &SourceLanguage,
    ) -> Option<tree_sitter::Node<'a>> {
        // Check if this node matches our target criteria
        if self.node_matches_criteria(node, content, node_type, node_name, language) {
            return Some(node.clone());
        }
        
        // Recurse into children
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if let Some(found) = self.find_node_by_criteria(&child, content, node_type, node_name, language) {
                    return Some(found);
                }
            }
        }
        
        None
    }

    /// Check if a node matches the target criteria
    fn node_matches_criteria(
        &self,
        node: &tree_sitter::Node,
        content: &str,
        node_type: &str,
        node_name: &str,
        language: &SourceLanguage,
    ) -> bool {
        let kind = node.kind();
        // Normalize node type for matching
        let target_normalized = node_type.to_lowercase();
        
        match language {
            SourceLanguage::Python => {
                match target_normalized.as_str() {
                    "function_definition" | "function" => {
                        if kind == "function_definition" {
                            // Check if the function name matches
                            if let Some(name_node) = self.get_function_name_node(node, content) {
                                let name = self.get_node_text(&name_node, content);
                                return name == node_name;
                            }
                        }
                        false
                    }
                    "class_definition" | "class" => {
                        if kind == "class_definition" {
                            if let Some(name_node) = self.get_class_name_node(node, content) {
                                let name = self.get_node_text(&name_node, content);
                                return name == node_name;
                            }
                        }
                        false
                    }
                    _ => kind.to_lowercase() == target_normalized,
                }
            }
            SourceLanguage::JavaScript | SourceLanguage::TypeScript => {
                match target_normalized.as_str() {
                    "function_declaration" | "function_definition" | "function" => {
                        if kind == "function_declaration" || kind == "method_definition" {
                            if let Some(name_node) = self.get_js_function_name(node, content) {
                                let name = self.get_node_text(&name_node, content);
                                return name == node_name;
                            }
                        }
                        false
                    }
                    "class_declaration" | "class" => {
                        if kind == "class_declaration" {
                            if let Some(name_node) = self.get_class_name_node(node, content) {
                                let name = self.get_node_text(&name_node, content);
                                return name == node_name;
                            }
                        }
                        false
                    }
                    _ => kind.to_lowercase() == target_normalized,
                }
            }
            SourceLanguage::Rust => {
                match target_normalized.as_str() {
                    "function_item" | "function" | "fn" => {
                        if kind == "function_item" {
                            if let Some(name_node) = self.get_rust_function_name(node, content) {
                                let name = self.get_node_text(&name_node, content);
                                return name == node_name;
                            }
                        }
                        false
                    }
                    "struct_item" | "struct" => {
                        if kind == "struct_item" {
                            if let Some(name_node) = self.get_rust_type_name(node, content) {
                                let name = self.get_node_text(&name_node, content);
                                return name == node_name;
                            }
                        }
                        false
                    }
                    "impl_item" | "impl" => {
                        if kind == "impl_item" {
                            // For impl blocks, we look for the type being implemented
                            let impl_text = self.get_node_text(node, content);
                            return impl_text.contains(node_name);
                        }
                        false
                    }
                    _ => kind.to_lowercase() == target_normalized,
                }
            }
            SourceLanguage::Unknown => false,
        }
    }

    fn get_node_text(&self, node: &tree_sitter::Node, content: &str) -> String {
        let start = node.start_byte();
        let end = node.end_byte();
        content[start..end].to_string()
    }

    // Node extraction helpers
    fn get_function_name_node<'a>(&self, node: &tree_sitter::Node<'a>, content: &str) -> Option<tree_sitter::Node<'a>> {
        match node.kind() {
            "function_definition" => {
                for i in 0..node.child_count() {
                    let child = node.child(i).unwrap();
                    if child.kind() == "identifier" {
                        return Some(child);
                    }
                }
            }
            "function_declaration" | "method_definition" => {
                return self.get_js_function_name(node, content);
            }
            "function_item" => {
                return self.get_rust_function_name(node, content);
            }
            _ => {}
        }
        None
    }

    fn get_class_name_node<'a>(&self, node: &tree_sitter::Node<'a>, content: &str) -> Option<tree_sitter::Node<'a>> {
        match node.kind() {
            "class_definition" => {
                for i in 0..node.child_count() {
                    let child = node.child(i).unwrap();
                    if child.kind() == "identifier" {
                        return Some(child);
                    }
                }
            }
            "class_declaration" => {
                for i in 0..node.child_count() {
                    let child = node.child(i).unwrap();
                    if child.kind() == "identifier" {
                        return Some(child);
                    }
                }
            }
            "struct_item" | "enum_item" | "impl_item" => {
                return self.get_rust_type_name(node, content);
            }
            _ => {}
        }
        None
    }

    // JavaScript/TypeScript node extraction helpers
    fn get_js_function_name<'a>(&self, node: &tree_sitter::Node<'a>, _content: &str) -> Option<tree_sitter::Node<'a>> {
        // For function declarations, name is first identifier
        // For method definitions, name is in property_name
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                let child_kind = child.kind();
                if child_kind == "identifier" {
                    return Some(child);
                }
                if child_kind == "property_name" {
                    return child.child(0); // Get the identifier inside
                }
            }
        }
        None
    }

    // Rust node extraction helpers
    fn get_rust_function_name<'a>(&self, node: &tree_sitter::Node<'a>, _content: &str) -> Option<tree_sitter::Node<'a>> {
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if child.kind() == "identifier" {
                    return Some(child);
                }
            }
        }
        None
    }

    fn get_rust_type_name<'a>(&self, node: &tree_sitter::Node<'a>, _content: &str) -> Option<tree_sitter::Node<'a>> {
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                let kind = child.kind();
                if kind == "identifier" || kind == "type_identifier" {
                    return Some(child);
                }
            }
        }
        None
    }

    /// apply_unified_diff - Applies a strictly formatted unified diff inside the sandbox
    /// 
    /// Security: Executes the patch command inside bwrap sandbox to prevent
    /// path traversal attacks from malformed unified diffs.
    pub async fn apply_unified_diff(&self, patch: &str) -> Result<DiffResult> {
        let workspace = self.workspace_path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No workspace configured"))?;

        let sandbox = self.sandbox_executor.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No sandbox configured - cannot safely apply patches"))?;

        // Validate patch format - must contain proper unified diff markers
        if !patch.contains("---") || !patch.contains("+++") {
            return Err(anyhow::anyhow!(
                "Invalid unified diff format. Must contain '---' and '+++' markers."
            ));
        }

        // Extract file path from patch header for validation
        let lines: Vec<&str> = patch.lines().collect();
        let file_path = {
            let plus_line = lines.iter()
                .find(|l| l.starts_with("+++ "))
                .ok_or_else(|| anyhow::anyhow!("Invalid patch format: no +++ header found"))?;
            plus_line
                .trim_start_matches("+++ ")
                .trim_start_matches("b/")
                .trim_start_matches("a/")
                .split('\t').next().unwrap_or(plus_line)
                .to_string()
        };

        // Validate the file path doesn't contain path traversal attempts
        if file_path.contains("..") || file_path.starts_with('/') {
            return Err(anyhow::anyhow!(
                "Invalid file path in patch: path traversal not allowed"
            ));
        }

        // Write patch file inside workspace (sandbox will contain it)
        let patch_file = workspace.join(".warden_patch.diff");
        fs::write(&patch_file, patch)?;

        // Execute patch command INSIDE the bwrap sandbox.
        // The sandbox bind-mounts the workspace as --bind (read-write),
        // so the patch CAN write anywhere in the workspace.  Path validation
        // (no .. or leading /) is the primary defense; the sandbox adds a
        // second layer but is not absolute isolation for crafted patches.
        // See src/sandbox.rs build_bwrap_command for details.
        let mut config = SandboxConfig::default();
        config.working_directory = Some(workspace.clone());
        config.env_vars.push((
            "PATCH_EXCLUDE".to_string(),
            "*.o:*.pyc:.git/*".to_string(),
        ));

        // Apply patch directly - the sandbox contains the damage
        let apply_cmd = vec![
            "patch".to_string(),
            "-p1".to_string(),
            "-i".to_string(),
            ".warden_patch.diff".to_string(),
            "--forward".to_string(),
        ];

        let apply_result = sandbox.execute(&apply_cmd, Some(config)).await?;

        // Clean up patch file
        let _ = fs::remove_file(&patch_file);

        if apply_result.exit_code != 0 {
            let stderr = &apply_result.stderr;
            return Err(anyhow::anyhow!(
                "Patch failed with exit code {}: {}",
                apply_result.exit_code,
                stderr
            ));
        }

        info!(file = file_path, "Unified diff applied successfully in sandbox");

        Ok(DiffResult {
            changed_files: vec![PathBuf::from(&file_path)],
            diff_output: format!("Applied sandboxed patch to {}", file_path),
        })
    }

    /// Get the diff of what changed (context squeezing)
    /// Note: This uses direct git diff - workspace should already be sandboxed
    #[allow(dead_code)]
    pub async fn get_changed_diff(&self, file: &str) -> Result<String> {
        let workspace = self.workspace_path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No workspace configured"))?;

        // Use git diff to get only the changes
        let output = std::process::Command::new("git")
            .args(["diff", file])
            .current_dir(workspace)
            .output()?;

        let diff = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(diff)
    }
}
