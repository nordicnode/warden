//! State Machine - Enforces strict phase-based tool access control
//!
//! PHASE 1: EXPLORE - Only Read, RAG, and LSP tools allowed
//! PHASE 2: REPRODUCE - Only test-creation and execution tools allowed (Red/Green Lock)
//! PHASE 3: PATCH - Edit tools unlocked (requires 1-sentence explanation via Why Gate)
//! PHASE 4: VERIFY - Automatic Compile-Lint-Test hook runs

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Explore,
    Reproduce,
    Patch,
    Verify,
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Phase::Explore => write!(f, "EXPLORE"),
            Phase::Reproduce => write!(f, "REPRODUCE"),
            Phase::Patch => write!(f, "PATCH"),
            Phase::Verify => write!(f, "VERIFY"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolCategory {
    // EXPLORE phase tools
    Read,
    RAG,
    LSP,
    Skeleton,

    // REPRODUCE phase tools
    TestWrite,
    TestExecute,

    // PATCH phase tools
    Edit,
    DiffApply,
    WhyGate,

    // VERIFY phase tools
    Verify,

    // Universal (always allowed)
    Scratchpad,
    Status,
    PhaseTransition,
}

#[derive(Debug, Error)]
pub enum StateMachineError {
    #[error("Tool '{0}' is not allowed in {1} phase. Allowed tools: {2}")]
    ToolNotAllowedInPhase(String, Phase, String),

    #[error("Unknown tool: '{0}'. Use get_status to see available tools.")]
    UnknownTool(String),

    #[error("Invalid state transition from {0} to {1}")]
    InvalidTransition(Phase, Phase),

    #[error("Why Gate not satisfied: {0}")]
    WhyGateNotSatisfied(String),

    #[error("Red/Green Lock active: source files are locked until test is written and fails")]
    RedGreenLockActive(String),

    #[error("Syntax validation failed: {0}")]
    SyntaxValidationFailed(String),

    #[error("Test validation failed: {0}")]
    TestValidationFailed(String),
}

pub struct StateMachine {
    phase: Phase,
    why_gate_explanation: Option<String>,
    red_green_locked: bool,
    test_written: bool,
    test_fails_matching_bug: bool,
}

impl StateMachine {
    pub fn new() -> Self {
        Self {
            phase: Phase::Explore,
            why_gate_explanation: None,
            red_green_locked: false,
            test_written: false,
            test_fails_matching_bug: false,
        }
    }

    /// Get the current phase
    pub fn current_phase(&self) -> Phase {
        self.phase
    }

    /// Get all tools allowed in the current phase
    pub fn allowed_tools(&self) -> Vec<&'static str> {
        match self.phase {
            Phase::Explore => vec![
                "get_skeleton",
                "find_relevant_code",
                "lsp_find_references",
                "lsp_get_hover",
                "lsp_get_definition",
                "lsp_complete",
                "read_file",
                "list_directory",
                "search_files",
                "get_scratchpad",
                "set_scratchpad",
                "get_status",
                "transition_phase",
            ],
            Phase::Reproduce => vec![
                "write_test",
                "run_test",
                "run_traced_script",
                "read_file",
                "list_directory",
                "search_files",
                "get_skeleton",
                "find_relevant_code",
                "lsp_find_references",
                "lsp_get_hover",
                "lsp_get_definition",
                "lsp_complete",
                "get_scratchpad",
                "set_scratchpad",
                "get_status",
                "transition_phase",
            ],
            Phase::Patch => vec![
                "modify_ast",
                "apply_unified_diff",
                "provide_why_explanation",
                "read_file",
                "list_directory",
                "search_files",
                "get_skeleton",
                "find_relevant_code",
                "lsp_find_references",
                "lsp_get_hover",
                "lsp_get_definition",
                "lsp_complete",
                "get_scratchpad",
                "set_scratchpad",
                "get_status",
                "transition_phase",
            ],
            Phase::Verify => vec![
                "run_lint",
                "run_compile",
                "run_tests",
                "read_file",
                "list_directory",
                "search_files",
                "get_skeleton",
                "find_relevant_code",
                "lsp_find_references",
                "lsp_get_hover",
                "lsp_get_definition",
                "lsp_complete",
                "get_scratchpad",
                "set_scratchpad",
                "get_status",
                "transition_phase",
            ],
        }
    }

    /// Check if a tool is allowed in the current phase
    pub fn is_tool_allowed(&self, tool_name: &str) -> Result<ToolCategory, StateMachineError> {
        let category = categorize_tool(tool_name)?;

        match self.phase {
            Phase::Explore => {
                match category {
                    ToolCategory::Read | ToolCategory::RAG | ToolCategory::LSP 
                    | ToolCategory::Skeleton | ToolCategory::Scratchpad | ToolCategory::Status
                    | ToolCategory::PhaseTransition => {
                        Ok(category)
                    }
                    _ => Err(StateMachineError::ToolNotAllowedInPhase(
                        tool_name.to_string(),
                        self.phase,
                        self.allowed_tools().join(", "),
                    )),
                }
            }
            Phase::Reproduce => {
                match category {
                    ToolCategory::TestWrite | ToolCategory::TestExecute 
                    | ToolCategory::Read | ToolCategory::RAG | ToolCategory::LSP | ToolCategory::Skeleton
                    | ToolCategory::Scratchpad | ToolCategory::Status
                    | ToolCategory::PhaseTransition => {
                        Ok(category)
                    }
                    _ => Err(StateMachineError::ToolNotAllowedInPhase(
                        tool_name.to_string(),
                        self.phase,
                        self.allowed_tools().join(", "),
                    )),
                }
            }
            Phase::Patch => {
                match category {
                    ToolCategory::Edit | ToolCategory::DiffApply | ToolCategory::WhyGate
                    | ToolCategory::Read | ToolCategory::RAG | ToolCategory::LSP | ToolCategory::Skeleton
                    | ToolCategory::Scratchpad | ToolCategory::Status
                    | ToolCategory::PhaseTransition => {
                        // Check Why Gate before allowing edits (but NOT for WhyGate itself)
                        if matches!(category, ToolCategory::Edit | ToolCategory::DiffApply) {
                            if self.why_gate_explanation.is_none() {
                                return Err(StateMachineError::WhyGateNotSatisfied(
                                    "Must provide why explanation before editing".to_string(),
                                ));
                            }
                        }
                        Ok(category)
                    }
                    _ => Err(StateMachineError::ToolNotAllowedInPhase(
                        tool_name.to_string(),
                        self.phase,
                        self.allowed_tools().join(", "),
                    )),
                }
            }
            Phase::Verify => {
                match category {
                    ToolCategory::Verify
                    | ToolCategory::Read | ToolCategory::RAG | ToolCategory::LSP | ToolCategory::Skeleton
                    | ToolCategory::Scratchpad | ToolCategory::Status
                    | ToolCategory::PhaseTransition => {
                        Ok(category)
                    }
                    _ => Err(StateMachineError::ToolNotAllowedInPhase(
                        tool_name.to_string(),
                        self.phase,
                        self.allowed_tools().join(", "),
                    )),
                }
            }
        }
    }

    /// Attempt to transition to the next phase
    pub fn transition(&mut self, target: Phase) -> Result<(), StateMachineError> {
        let valid_transitions = match self.phase {
            Phase::Explore => vec![Phase::Reproduce],
            Phase::Reproduce => vec![Phase::Explore, Phase::Patch],
            Phase::Patch => vec![Phase::Verify, Phase::Explore],
            Phase::Verify => vec![Phase::Explore, Phase::Patch],
        };

        if !valid_transitions.contains(&target) {
            return Err(StateMachineError::InvalidTransition(self.phase, target));
        }

        // Special validation for transitions
        match target {
            Phase::Reproduce => {
                // Reset test state when entering reproduce
                self.test_written = false;
                self.test_fails_matching_bug = false;
                self.red_green_locked = true;
            }
            Phase::Patch => {
                // Can only enter patch if test is written AND fails
                if !self.test_written {
                    return Err(StateMachineError::RedGreenLockActive(
                        "Must write a failing test before patching".to_string(),
                    ));
                }
                if !self.test_fails_matching_bug {
                    return Err(StateMachineError::RedGreenLockActive(
                        "Test must fail and match bug description before patching".to_string(),
                    ));
                }
                self.red_green_locked = false;
            }
            Phase::Verify => {
                // Reset why gate for next cycle
                self.why_gate_explanation = None;
            }
            _ => {}
        }

        self.phase = target;
        tracing::info!("State machine transitioned to {:?}", self.phase);
        Ok(())
    }

    /// Set the Why Gate explanation (required before PATCH edits)
    pub fn set_why_explanation(&mut self, explanation: String) -> Result<(), StateMachineError> {
        if self.phase != Phase::Patch {
            return Err(StateMachineError::WhyGateNotSatisfied(
                format!("Why explanation only required in PATCH phase, currently in {:?}", self.phase)
            ));
        }
        // Validate that the explanation is not empty or whitespace-only
        let trimmed = explanation.trim();
        if trimmed.is_empty() {
            return Err(StateMachineError::WhyGateNotSatisfied(
                "Why Gate explanation cannot be empty. Provide a meaningful 1-sentence explanation of why the change fixes the bug.".to_string(),
            ));
        }
        if trimmed.len() < 10 {
            return Err(StateMachineError::WhyGateNotSatisfied(
                format!("Why Gate explanation too short ({} chars). Provide at least 10 characters explaining why the change fixes the bug.", trimmed.len()),
            ));
        }
        tracing::info!("Why Gate satisfied: {}", explanation);
        self.why_gate_explanation = Some(trimmed.to_string());
        Ok(())
    }

    /// Mark test as written
    pub fn mark_test_written(&mut self) {
        self.test_written = true;
        tracing::info!("Test written and marked\nRed/Green Lock: test_written = true");
    }

    /// Mark test as failing and matching bug description
    pub fn mark_test_failing_matching_bug(&mut self) {
        self.test_fails_matching_bug = true;
        tracing::info!("Test fails and matches bug description\nRed/Green Lock: test_fails_matching_bug = true");
    }

    /// Check if Red/Green lock is active (source files locked)
    pub fn is_red_green_locked(&self) -> bool {
        self.red_green_locked && !self.test_fails_matching_bug
    }

    /// Getters for phase-specific status checks
    pub fn is_test_written(&self) -> bool {
        self.test_written
    }

    pub fn is_test_failing(&self) -> bool {
        self.test_fails_matching_bug
    }

    /// Get the Why Gate explanation if set
    pub fn get_why_explanation(&self) -> Option<&String> {
        self.why_gate_explanation.as_ref()
    }

    /// Rollback changes (used when VERIFY fails)
    pub fn rollback(&mut self) {
        tracing::warn!("Rolling back state machine to EXPLORE phase due to verification failure");
        self.phase = Phase::Explore;
        self.why_gate_explanation = None;
        self.red_green_locked = false;
        self.test_written = false;
        self.test_fails_matching_bug = false;
    }

    /// Get full state as JSON for debugging/status
    pub fn get_state_json(&self) -> serde_json::Value {
        serde_json::json!({
            "phase": self.phase.to_string(),
            "why_gate_satisfied": self.why_gate_explanation.is_some(),
            "why_gate_explanation": self.why_gate_explanation,
            "red_green_locked": self.red_green_locked,
            "test_written": self.test_written,
            "test_fails_matching_bug": self.test_fails_matching_bug,
            "allowed_tools": self.allowed_tools(),
        })
    }
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

fn categorize_tool(tool_name: &str) -> Result<ToolCategory, StateMachineError> {
    match tool_name {
        // EXPLORE tools
        "get_skeleton" => Ok(ToolCategory::Skeleton),
        "find_relevant_code" => Ok(ToolCategory::RAG),
        "lsp_find_references" | "lsp_get_hover" | "lsp_get_definition" | "lsp_complete" => {
            Ok(ToolCategory::LSP)
        }
        "read_file" | "list_directory" | "search_files" => Ok(ToolCategory::Read),

        // REPRODUCE tools
        "write_test" => Ok(ToolCategory::TestWrite),
        "run_test" | "run_traced_script" => Ok(ToolCategory::TestExecute),

        // PATCH tools
        "modify_ast" | "apply_unified_diff" | "apply_patch" => Ok(ToolCategory::Edit),
        "provide_why_explanation" => Ok(ToolCategory::WhyGate),

        // VERIFY tools
        "run_lint" | "run_compile" | "run_tests" | "verify" => Ok(ToolCategory::Verify),

        // Universal
        "get_scratchpad" | "set_scratchpad" => Ok(ToolCategory::Scratchpad),
        "get_status" => Ok(ToolCategory::Status),
        "transition_phase" => Ok(ToolCategory::PhaseTransition),

        _ => Err(StateMachineError::UnknownTool(
            tool_name.to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explore_phase_allows_read_tools() {
        let sm = StateMachine::new();
        assert!(sm.is_tool_allowed("read_file").is_ok());
        assert!(sm.is_tool_allowed("find_relevant_code").is_ok());
        assert!(sm.is_tool_allowed("lsp_get_hover").is_ok());
    }

    #[test]
    fn test_explore_phase_blocks_edit_tools() {
        let sm = StateMachine::new();
        assert!(sm.is_tool_allowed("apply_unified_diff").is_err());
    }

    #[test]
    fn test_reproduce_phase_requires_test_first() {
        let mut sm = StateMachine::new();
        sm.transition(Phase::Reproduce).unwrap();
        assert!(sm.is_tool_allowed("write_test").is_ok());
        assert!(sm.is_tool_allowed("apply_unified_diff").is_err());
    }

    #[test]
    fn test_why_gate_required_before_patch() {
        let mut sm = StateMachine::new();
        sm.transition(Phase::Reproduce).unwrap();
        sm.mark_test_written();
        sm.mark_test_failing_matching_bug();
        sm.transition(Phase::Patch).unwrap();

        // Should fail without why explanation
        assert!(sm.is_tool_allowed("apply_unified_diff").is_err());

        // Should succeed with why explanation
        sm.set_why_explanation("Fixes null pointer by adding null check".to_string()).unwrap();
        assert!(sm.is_tool_allowed("apply_unified_diff").is_ok());
    }

    #[test]
    fn test_state_json() {
        let sm = StateMachine::new();
        let json = sm.get_state_json();
        assert_eq!(json["phase"], "EXPLORE");
        assert!(json["red_green_locked"].is_boolean());
    }
}