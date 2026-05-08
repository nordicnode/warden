//! State Scratchpad - Persistent working memory for the LLM
//!
//! The LLM writes its current goal and tracked files here.
//! Auto-prunes files not touched in 3 turns.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedFile {
    pub path: String,
    pub last_touched_turn: u32,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ScratchpadEntry {
    pub content: String,
    pub timestamp: u64,
    pub turn_number: u32,
}

pub struct StateScratchpad {
    content: String,
    current_goal: Option<String>,
    tracked_files: HashMap<String, TrackedFile>,
    turn_count: u32,
    max_age_turns: u32,
    timestamp: u64,
}

impl StateScratchpad {
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            content: String::new(),
            current_goal: None,
            tracked_files: HashMap::new(),
            turn_count: 0,
            max_age_turns: 3,
            timestamp: now,
        }
    }

    /// Get current scratchpad content
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Set scratchpad content
    pub fn set_content(&mut self, content: String) {
        self.content = content;
        self.update_timestamp();
    }

    /// Get current goal
    pub fn current_goal(&self) -> Option<&String> {
        self.current_goal.as_ref()
    }

    /// Set current goal
    pub fn set_current_goal(&mut self, goal: String) {
        self.current_goal = Some(goal);
    }

    /// Get all tracked files
    pub fn tracked_files(&self) -> Vec<&TrackedFile> {
        self.tracked_files.values().collect()
    }

    /// Track a file (marks it as touched)
    pub fn track_file(&mut self, path: &str, reason: &str) {
        self.tracked_files.insert(
            path.to_string(),
            TrackedFile {
                path: path.to_string(),
                last_touched_turn: self.turn_count,
                reason: reason.to_string(),
            },
        );
    }

    /// Touch a file (update its last touched turn)
    pub fn touch_file(&mut self, path: &str) {
        if let Some(tracked) = self.tracked_files.get_mut(path) {
            tracked.last_touched_turn = self.turn_count;
        }
    }

    /// Update timestamp
    fn update_timestamp(&mut self) {
        self.timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    /// Get timestamp of last update
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Increment turn counter
    pub fn increment_turn(&mut self) {
        self.turn_count += 1;
    }

    /// Get current turn number
    pub fn turn_count(&self) -> u32 {
        self.turn_count
    }

    /// Prune old tracked files (not touched in max_age_turns)
    pub fn prune_old_entries(&mut self, max_age: Option<u32>) {
        let max_age = max_age.unwrap_or(self.max_age_turns);
        let cutoff_turn = self.turn_count.saturating_sub(max_age);

        self.tracked_files.retain(|_, tracked| {
            tracked.last_touched_turn >= cutoff_turn
        });
    }

    /// Parse goal from content (looks for # Goal: or similar markers)
    pub fn parse_goal(&self) -> Option<String> {
        for line in self.content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("# Goal:") || trimmed.starts_with("Goal:") || trimmed.starts_with("**Goal:**") {
                return Some(
                    trimmed
                        .replace("# Goal: ", "")
                        .replace("Goal: ", "")
                        .replace("**Goal:** ", "")
                        .to_string(),
                );
            }
        }
        self.current_goal.clone()
    }

    /// Parse tracked files from content (looks for paths mentioned)
    pub fn parse_tracked_from_content(&mut self) {
        // Look for patterns like `src/foo.py` or `lib/bar.js` in content.
        // Uses a code-extension heuristic: any word containing / and ending
        // with a known code extension is likely a file path.
        let words: Vec<String> = self.content
            .split_whitespace()
            .map(|w| w.trim_end_matches(".,;:!").to_string())
            .collect();

        let code_extensions = [".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go",
            ".java", ".cpp", ".c", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt"];

        for word in words {
            if word.contains('/') && code_extensions.iter().any(|ext| word.ends_with(ext)) {
                if !self.tracked_files.contains_key(&word) {
                    self.track_file(&word, "Mentioned in scratchpad");
                }
            }
        }
    }

    /// Generate summary for injection into LLM context
    pub fn generate_context_summary(&self) -> String {
        let mut summary = String::new();

        if !self.content.is_empty() {
            summary.push_str("## Scratchpad\n");
            summary.push_str(&self.content);
            summary.push_str("\n\n");
        }

        if let Some(goal) = &self.current_goal {
            summary.push_str(&format!("**Current Goal:** {}\n\n", goal));
        }

        if !self.tracked_files.is_empty() {
            summary.push_str("**Tracked Files:**\n");
            for tracked in self.tracked_files.values() {
                summary.push_str(&format!(
                    "- {} (touched {} turns ago): {}\n",
                    tracked.path,
                    self.turn_count.saturating_sub(tracked.last_touched_turn),
                    tracked.reason
                ));
            }
            summary.push_str("\n");
        }

        summary
    }

    /// Clear all scratchpad state
    pub fn clear(&mut self) {
        self.content.clear();
        self.current_goal = None;
        self.tracked_files.clear();
        self.turn_count = 0;
    }
}

impl Default for StateScratchpad {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scratchpad_creation() {
        let scratchpad = StateScratchpad::new();
        assert!(scratchpad.content().is_empty());
        assert!(scratchpad.tracked_files().is_empty());
    }

    #[test]
    fn test_track_file() {
        let mut scratchpad = StateScratchpad::new();
        scratchpad.track_file("src/main.rs", "Main entry point");
        assert_eq!(scratchpad.tracked_files().len(), 1);
        assert_eq!(scratchpad.tracked_files()[0].path, "src/main.rs");
    }

    #[test]
    fn test_prune_old_entries() {
        let mut scratchpad = StateScratchpad::new();
        scratchpad.track_file("src/a.rs", "Old file");

        scratchpad.turn_count = 10;
        scratchpad.touch_file("src/a.rs"); // Touch at turn 10

        scratchpad.turn_count = 15;
        scratchpad.prune_old_entries(Some(3)); // Keep files touched in last 3 turns

        // a.rs was touched at turn 10, now at turn 15, so 5 turns ago - should be pruned
        assert!(scratchpad.tracked_files().is_empty());
    }

    #[test]
    fn test_context_summary() {
        let mut scratchpad = StateScratchpad::new();
        scratchpad.set_content("Working on the bug fix\nFile: src/main.rs".to_string());
        scratchpad.set_current_goal("Fix null pointer exception".to_string());
        scratchpad.track_file("src/main.rs", "Main file");

        let summary = scratchpad.generate_context_summary();
        assert!(summary.contains("Scratchpad"));
        assert!(summary.contains("null pointer"));
        assert!(summary.contains("src/main.rs"));
    }
}