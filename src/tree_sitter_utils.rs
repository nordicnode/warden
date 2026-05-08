//! Tree-sitter AST Utilities for code skeleton extraction
//!
//! Provides AST-based parsing for extracting imports, function signatures,
//! class definitions, and docstrings from source code.

use tree_sitter::{Parser, Language, Node, Tree};
use std::path::Path;
use std::fs;
use anyhow::{Result, Context};
use tracing::info;

// Import official tree-sitter language bindings
use tree_sitter_python::language as ts_python;
use tree_sitter_javascript::language as ts_javascript;
use tree_sitter_rust::language as ts_rust;
use tree_sitter_typescript::language_typescript as ts_typescript;

#[derive(Debug, Clone, Copy)]
pub enum SourceLanguage {
    Python,
    JavaScript,
    TypeScript,
    Rust,
    Unknown,
}

impl SourceLanguage {
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "py" => SourceLanguage::Python,
            "js" | "jsx" => SourceLanguage::JavaScript,
            "ts" | "tsx" => SourceLanguage::TypeScript,
            "rs" => SourceLanguage::Rust,
            _ => SourceLanguage::Unknown,
        }
    }

    pub fn get_language(&self) -> Option<Language> {
        match self {
            SourceLanguage::Python => Some(ts_python()),
            SourceLanguage::JavaScript => Some(ts_javascript()),
            SourceLanguage::TypeScript => Some(ts_typescript()),
            SourceLanguage::Rust => Some(ts_rust()),
            SourceLanguage::Unknown => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub name: String,
    pub parameters: String,
    pub return_type: Option<String>,
    #[allow(dead_code)]
    pub start_byte: usize,
    #[allow(dead_code)]
    pub end_byte: usize,
}

#[derive(Debug, Clone)]
pub struct ClassDefinition {
    pub name: String,
    #[allow(dead_code)]
    pub start_byte: usize,
    #[allow(dead_code)]
    pub end_byte: usize,
}

#[derive(Debug, Clone)]
pub struct ImportStatement {
    pub module: String,
    pub names: Vec<String>,
    #[allow(dead_code)]
    pub start_byte: usize,
    #[allow(dead_code)]
    pub end_byte: usize,
}

#[derive(Debug, Clone)]
pub struct DocComment {
    pub content: String,
    #[allow(dead_code)]
    pub start_byte: usize,
    #[allow(dead_code)]
    pub end_byte: usize,
}

#[derive(Debug, Default)]
pub struct SkeletonData {
    pub imports: Vec<ImportStatement>,
    pub functions: Vec<FunctionSignature>,
    pub classes: Vec<ClassDefinition>,
    pub docstrings: Vec<DocComment>,
}

pub struct TreeSitterParser {
    parser: Parser,
}

impl TreeSitterParser {
    pub fn new() -> Result<Self> {
        let parser = Parser::new();
        Ok(Self { parser })
    }

        #[allow(dead_code)]
    pub fn with_language(language: Language) -> Result<Self> {
        let mut parser = Parser::new();
        parser.set_language(language).context("Failed to set language")?;
        Ok(Self { parser })
    }

    pub fn set_language(&mut self, language: Language) -> Result<()> {
        self.parser.set_language(language).context("Failed to set language")?;
        Ok(())
    }

    pub fn parse(&mut self, source: &str) -> Result<Tree> {
        self.parser
            .parse(source, None)
            .context("Failed to parse source code")
    }

    /// Extract skeleton data from source code
    pub fn extract_skeleton(&mut self, source: &str, language: SourceLanguage) -> Result<SkeletonData> {
        let lang = match language.get_language() {
            Some(l) => l,
            None => return Ok(SkeletonData::default()),
        };

        self.parser.set_language(lang)?;
        let tree = self.parse(source)?;
        let root = tree.root_node();

        let mut data = SkeletonData::default();

        // Walk the tree and extract relevant nodes
        self.walk_and_extract(&root, source, &language, &mut data);

        info!(
            language = ?language,
            imports = data.imports.len(),
            functions = data.functions.len(),
            classes = data.classes.len(),
            "Skeleton extracted successfully"
        );

        Ok(data)
    }

    fn walk_and_extract(&self, node: &Node, source: &str, language: &SourceLanguage, data: &mut SkeletonData) {
        match language {
            SourceLanguage::Python => self.extract_python_nodes(node, source, data),
            SourceLanguage::JavaScript | SourceLanguage::TypeScript => self.extract_js_nodes(node, source, data),
            SourceLanguage::Rust => self.extract_rust_nodes(node, source, data),
            SourceLanguage::Unknown => {}
        }

        // Recurse into children
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                self.walk_and_extract(&child, source, language, data);
            }
        }
    }

    fn extract_python_nodes(&self, node: &Node, source: &str, data: &mut SkeletonData) {
        let kind = node.kind();

        match kind {
            "import_statement" | "import_from_statement" => {
                if let Some(imp) = self.extract_python_import(node, source) {
                    data.imports.push(imp);
                }
            }
            "function_definition" => {
                if let Some(func) = self.extract_python_function(node, source) {
                    data.functions.push(func);
                }
            }
            "class_definition" => {
                if let Some(cls) = self.extract_python_class(node, source) {
                    data.classes.push(cls);
                }
            }
            "expression_statement" => {
                if let Some(doc) = self.extract_python_docstring(node, source) {
                    data.docstrings.push(doc);
                }
            }
            _ => {}
        }
    }

    fn extract_python_import(&self, node: &Node, source: &str) -> Option<ImportStatement> {
        let start = node.start_byte();
        let end = node.end_byte();

        // Parse module name and imported names
        let mut names = Vec::new();
        let mut module_name = String::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                let child_kind = child.kind();
                if child_kind == "identifier" {
                    names.push(self.get_node_text(&child, source));
                } else if child_kind == "dotted_name" {
                    module_name = self.get_node_text(&child, source);
                }
            }
        }

        // For import_statement, use the first identifier as module
        // For import_from_statement, use the dotted_name we found
        if module_name.is_empty() {
            module_name = names.first().cloned().unwrap_or_default();
        }

        Some(ImportStatement {
            module: module_name,
            names,
            start_byte: start,
            end_byte: end,
        })
    }

    fn extract_python_function(&self, node: &Node, source: &str) -> Option<FunctionSignature> {
        let mut name = String::new();
        let mut parameters = String::new();
        let mut return_type = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "identifier" => name = self.get_node_text(&child, source),
                    "parameters" => parameters = self.get_node_text(&child, source),
                    "return_type" | "type" => return_type = Some(self.get_node_text(&child, source)),
                    _ => {}
                }
            }
        }

        if name.is_empty() {
            return None;
        }

        Some(FunctionSignature {
            name,
            parameters,
            return_type,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
        })
    }

    fn extract_python_class(&self, node: &Node, source: &str) -> Option<ClassDefinition> {
        let mut name = String::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if child.kind() == "identifier" {
                    name = self.get_node_text(&child, source);
                    break;
                }
            }
        }

        if name.is_empty() {
            return None;
        }

        Some(ClassDefinition {
            name,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
        })
    }

    fn extract_python_docstring(&self, node: &Node, source: &str) -> Option<DocComment> {
        if node.child_count() > 0 {
            if let Some(first_child) = node.child(0) {
                if first_child.kind() == "string" {
                    let content = self.get_node_text(&first_child, source);
                    let cleaned = content.trim_matches('"').trim_matches('\n');
                    return Some(DocComment {
                        content: cleaned.to_string(),
                        start_byte: node.start_byte(),
                        end_byte: node.end_byte(),
                    });
                }
            }
        }
        None
    }

    fn extract_js_nodes(&self, node: &Node, source: &str, data: &mut SkeletonData) {
        let kind = node.kind();

        match kind {
            "import_statement" | "import_clause" => {
                if let Some(imp) = self.extract_js_import(node, source) {
                    data.imports.push(imp);
                }
            }
            "function_declaration" | "method_definition" => {
                if let Some(func) = self.extract_js_function(node, source) {
                    data.functions.push(func);
                }
            }
            "class_declaration" => {
                if let Some(cls) = self.extract_js_class(node, source) {
                    data.classes.push(cls);
                }
            }
            "lexical_declaration" => {
                for i in 0..node.child_count() {
                    if let Some(child) = node.child(i) {
                        if child.kind() == "variable_declarator" {
                            if let Some(func) = self.extract_js_arrow_function(&child, source) {
                                data.functions.push(func);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn extract_js_import(&self, node: &Node, source: &str) -> Option<ImportStatement> {
        let mut module = String::new();
        let mut names = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "string" => module = self.get_node_text(&child, source),
                    "identifier" => names.push(self.get_node_text(&child, source)),
                    "import_specifier" => {
                        if let Some(name) = child.child(0) {
                            names.push(self.get_node_text(&name, source));
                        }
                    }
                    _ => {}
                }
            }
        }

        Some(ImportStatement {
            module,
            names,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
        })
    }

    fn extract_js_function(&self, node: &Node, source: &str) -> Option<FunctionSignature> {
        let mut name = String::new();
        let mut parameters = String::new();
        let mut return_type = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "identifier" => name = self.get_node_text(&child, source),
                    "formal_parameters" => parameters = self.get_node_text(&child, source),
                    "type_annotation" => return_type = Some(self.get_node_text(&child, source)),
                    _ => {}
                }
            }
        }

        Some(FunctionSignature {
            name,
            parameters,
            return_type,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
        })
    }

    fn extract_js_arrow_function(&self, node: &Node, source: &str) -> Option<FunctionSignature> {
        let mut name = String::new();
        let mut parameters = String::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "identifier" => name = self.get_node_text(&child, source),
                    "arrow_function" => {
                        if let Some(params) = child.child(0) {
                            parameters = self.get_node_text(&params, source);
                        }
                    }
                    _ => {}
                }
            }
        }

        if name.is_empty() {
            return None;
        }

        Some(FunctionSignature {
            name,
            parameters,
            return_type: None,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
        })
    }

    fn extract_js_class(&self, node: &Node, source: &str) -> Option<ClassDefinition> {
        let mut name = String::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if child.kind() == "identifier" {
                    name = self.get_node_text(&child, source);
                    break;
                }
            }
        }

        Some(ClassDefinition {
            name,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
        })
    }

    fn extract_rust_nodes(&self, node: &Node, source: &str, data: &mut SkeletonData) {
        let kind = node.kind();

        match kind {
            "use_declaration" => {
                if let Some(imp) = self.extract_rust_import(node, source) {
                    data.imports.push(imp);
                }
            }
            "function_item" => {
                if let Some(func) = self.extract_rust_function(node, source) {
                    data.functions.push(func);
                }
            }
            "struct_item" | "enum_item" => {
                if let Some(cls) = self.extract_rust_type(node, source) {
                    data.classes.push(cls);
                }
            }
            _ => {}
        }
    }

    fn extract_rust_import(&self, node: &Node, source: &str) -> Option<ImportStatement> {
        let module = self.get_node_text(node, source);

        Some(ImportStatement {
            module,
            names: vec![],
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
        })
    }

    fn extract_rust_function(&self, node: &Node, source: &str) -> Option<FunctionSignature> {
        let mut name = String::new();
        let mut parameters = String::new();
        let mut return_type = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "identifier" => name = self.get_node_text(&child, source),
                    "parameters" => parameters = self.get_node_text(&child, source),
                    "type_annotation" => return_type = Some(self.get_node_text(&child, source)),
                    _ => {}
                }
            }
        }

        Some(FunctionSignature {
            name,
            parameters,
            return_type,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
        })
    }

    fn extract_rust_type(&self, node: &Node, source: &str) -> Option<ClassDefinition> {
        let mut name = String::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "identifier" => name = self.get_node_text(&child, source),
                    "type_identifier" => name = self.get_node_text(&child, source),
                    _ => {}
                }
            }
        }

        Some(ClassDefinition {
            name,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
        })
    }

    fn get_node_text(&self, node: &Node, source: &str) -> String {
        let start = node.start_byte();
        let end = node.end_byte();
        source[start..end].to_string()
    }
}

impl Default for TreeSitterParser {
    fn default() -> Self {
        Self::new().expect("Failed to create Tree-sitter parser")
    }
}

/// Parse a file and extract its skeleton using Tree-sitter
#[allow(dead_code)]
pub fn extract_file_skeleton(file_path: &Path) -> Result<(SkeletonData, SourceLanguage)> {
    let content = fs::read_to_string(file_path)?;
    let ext = file_path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let language = SourceLanguage::from_extension(ext);
    let mut parser = TreeSitterParser::new()?;
    let skeleton = parser.extract_skeleton(&content, language)?;

    Ok((skeleton, language))
}