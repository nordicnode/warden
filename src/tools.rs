//! MCP Tools - Definitions for all available tools exposed to the LLM
//!
//! Tools are categorized by the phase in which they are available.
//!
//! FIXES APPLIED:
//! - Aligned all tool input_schema parameters with actual handler signatures.
//! - lsp_find_references: changed `symbol`->`file_path`+`line`+`column` to match handler.
//! - lsp_get_hover: changed `symbol`->`file_path`+`line`+`column` to match handler.
//! - lsp_get_definition: changed `symbol`->`file_path`+`line`+`column` to match handler.
//! - modify_ast: changed `file`->`file_path` to be consistent (handler uses `file`).
//! - Removed setup_workspace (workspace is auto-mounted in main.rs).
//! - Read/LSP/RAG tools are now available in ALL phases via read_explore_tools().

use serde::{Deserialize, Serialize};
use crate::state_machine::Phase;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub category: ToolType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolType {
    Read,
    Write,
    Execute,
    LSP,
    RAG,
    Utility,
}

impl ToolDefinition {
    pub fn new(name: &str, description: &str, input_schema: serde_json::Value, category: ToolType) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            input_schema,
            category,
        }
    }
}

/// Get all tools available for a given phase
pub fn list_available_tools(phase: &Phase) -> serde_json::Value {
    let tools = match phase {
        Phase::Explore => explore_tools(),
        Phase::Reproduce => reproduce_tools(),
        Phase::Patch => patch_tools(),
        Phase::Verify => verify_tools(),
    };

    serde_json::json!({"tools": tools})
}

/// Universal utility tools available in every phase
fn universal_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            "get_scratchpad",
            "Gets the current state scratchpad content.",
            serde_json::json!({
                "type": "object",
                "properties": {}
            }),
            ToolType::Utility,
        ),
        ToolDefinition::new(
            "set_scratchpad",
            "Sets the state scratchpad content. Include your current goal and tracked files.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Scratchpad content to write"
                    }
                },
                "required": ["content"]
            }),
            ToolType::Utility,
        ),
        ToolDefinition::new(
            "get_status",
            "Gets the current session status including phase, allowed tools, and lock states.",
            serde_json::json!({
                "type": "object",
                "properties": {}
            }),
            ToolType::Utility,
        ),
        ToolDefinition::new(
            "transition_phase",
            "Transitions the session to a new phase (EXPLORE, REPRODUCE, PATCH, VERIFY).",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "target_phase": {
                        "type": "string",
                        "enum": ["EXPLORE", "REPRODUCE", "PATCH", "VERIFY"],
                        "description": "The phase to transition to"
                    }
                },
                "required": ["target_phase"]
            }),
            ToolType::Utility,
        ),
    ]
}

/// Shared read/exploration tools available in ALL phases
fn read_explore_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            "get_skeleton",
            "Returns only imports, class names, function signatures, and docstrings from a file. Strips all inner logic using Tree-sitter.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to analyze"
                    }
                },
                "required": ["filepath"]
            }),
            ToolType::Read,
        ),
        ToolDefinition::new(
            "find_relevant_code",
            "Queries local Vector DB to return top 5 semantically relevant functions/files based on a natural language bug description.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "issue_text": {
                        "type": "string",
                        "description": "Natural language description of the bug"
                    }
                },
                "required": ["issue_text"]
            }),
            ToolType::RAG,
        ),
        ToolDefinition::new(
            "lsp_find_references",
            "Uses LSP to find all references to a symbol at the given file/line/column.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File path where the symbol is used"
                    },
                    "line": {
                        "type": "integer",
                        "description": "Line number (1-indexed)"
                    },
                    "column": {
                        "type": "integer",
                        "description": "Column number (1-indexed)"
                    },
                    "include_declaration": {
                        "type": "boolean",
                        "description": "Whether to include the declaration"
                    }
                },
                "required": ["file_path", "line", "column"]
            }),
            ToolType::LSP,
        ),
        ToolDefinition::new(
            "lsp_get_hover",
            "Gets hover documentation for a symbol at the given file/line/column.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File path"
                    },
                    "line": {
                        "type": "integer",
                        "description": "Line number (1-indexed)"
                    },
                    "column": {
                        "type": "integer",
                        "description": "Column number (1-indexed)"
                    }
                },
                "required": ["file_path", "line", "column"]
            }),
            ToolType::LSP,
        ),
        ToolDefinition::new(
            "lsp_get_definition",
            "Navigates to the definition of the symbol at the given file/line/column.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File path"
                    },
                    "line": {
                        "type": "integer",
                        "description": "Line number (1-indexed)"
                    },
                    "column": {
                        "type": "integer",
                        "description": "Column number (1-indexed)"
                    }
                },
                "required": ["file_path", "line", "column"]
            }),
            ToolType::LSP,
        ),
        ToolDefinition::new(
            "lsp_complete",
            "Gets completion suggestions at a given position.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File path"
                    },
                    "line": {
                        "type": "integer",
                        "description": "Line number"
                    },
                    "column": {
                        "type": "integer",
                        "description": "Column number"
                    }
                },
                "required": ["file_path", "line", "column"]
            }),
            ToolType::LSP,
        ),
        ToolDefinition::new(
            "read_file",
            "Reads the content of a file from the workspace.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional: Start line for partial read"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional: End line for partial read"
                    }
                },
                "required": ["path"]
            }),
            ToolType::Read,
        ),
        ToolDefinition::new(
            "list_directory",
            "Lists files and directories at a given path.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (relative to workspace)"
                    }
                },
                "required": ["path"]
            }),
            ToolType::Read,
        ),
        ToolDefinition::new(
            "search_files",
            "Searches for a pattern in files using regex.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in"
                    },
                    "file_type": {
                        "type": "string",
                        "description": "Optional: Filter by file type (e.g., 'py', 'js')"
                    }
                },
                "required": ["pattern"]
            }),
            ToolType::Read,
        ),
    ]
}

/// EXPLORE phase tools (read-only + universal)
fn explore_tools() -> Vec<ToolDefinition> {
    let mut tools = read_explore_tools();
    tools.extend(universal_tools());
    tools
}

/// REPRODUCE phase tools (read + test write/execute + universal)
fn reproduce_tools() -> Vec<ToolDefinition> {
    let mut tools = read_explore_tools();
    tools.extend(vec![
        ToolDefinition::new(
            "write_test",
            "Writes a test file to the workspace. The test must match the bug description and fail initially.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to write the test file"
                    },
                    "content": {
                        "type": "string",
                        "description": "Test file content"
                    }
                },
                "required": ["path", "content"]
            }),
            ToolType::Write,
        ),
        ToolDefinition::new(
            "run_test",
            "Runs a test file in the sandbox and returns the results.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "test_path": {
                        "type": "string",
                        "description": "Path to the test file or test command"
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Optional: Override default timeout (default: 30s)"
                    }
                },
                "required": ["test_path"]
            }),
            ToolType::Execute,
        ),
        ToolDefinition::new(
            "run_traced_script",
            "Executes a reproduction script inside the bwrap sandbox with tracing enabled to capture the hot path.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Command and arguments to execute"
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "Optional: Working directory for the command"
                    }
                },
                "required": ["command"]
            }),
            ToolType::Execute,
        ),
    ]);
    tools.extend(universal_tools());
    tools
}

/// PATCH phase tools (read + edit + why gate + universal)
fn patch_tools() -> Vec<ToolDefinition> {
    let mut tools = read_explore_tools();
    tools.extend(vec![
        ToolDefinition::new(
            "modify_ast",
            "Modifies code structurally by targeting specific AST nodes.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File path to modify (canonical parameter name)"
                    },
                    "file": {
                        "type": "string",
                        "description": "Alias for file_path"
                    },
                    "node_type": {
                        "type": "string",
                        "description": "AST node type to target (e.g., 'function_definition', 'class_def')"
                    },
                    "node_name": {
                        "type": "string",
                        "description": "Name of the node to modify"
                    },
                    "new_code": {
                        "type": "string",
                        "description": "Replacement code for the node"
                    }
                },
                "required": ["file_path", "node_type", "node_name", "new_code"]
            }),
            ToolType::Write,
        ),
        ToolDefinition::new(
            "apply_unified_diff",
            "Applies a strictly formatted unified diff to the overlayfs upperdir. REQUIRES Why Gate explanation first.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "Unified diff content"
                    }
                },
                "required": ["patch"]
            }),
            ToolType::Write,
        ),
        ToolDefinition::new(
            "provide_why_explanation",
            "Provides the 1-sentence JSON explanation for why the change fixes the bug. Required before editing tools work in PATCH phase.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": "1-sentence explanation of why the change fixes the bug"
                    }
                },
                "required": ["explanation"]
            }),
            ToolType::Utility,
        ),
    ]);
    tools.extend(universal_tools());
    tools
}

/// VERIFY phase tools (read + verify + universal)
fn verify_tools() -> Vec<ToolDefinition> {
    let mut tools = read_explore_tools();
    tools.extend(vec![
        ToolDefinition::new(
            "run_lint",
            "Runs the lint check on modified files.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Optional: Specific files to lint"
                    }
                }
            }),
            ToolType::Execute,
        ),
        ToolDefinition::new(
            "run_compile",
            "Runs syntax/compile check on modified files.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Optional: Specific files to compile check"
                    }
                }
            }),
            ToolType::Execute,
        ),
        ToolDefinition::new(
            "run_tests",
            "Runs the full test suite to verify the fix.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "test_filter": {
                        "type": "string",
                        "description": "Optional: Filter for specific tests"
                    }
                }
            }),
            ToolType::Execute,
        ),
    ]);
    tools.extend(universal_tools());
    tools
}
