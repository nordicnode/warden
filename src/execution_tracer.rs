//! Execution Tracer - Wraps execution in bwrap sandbox with hot path tracing
//!
//! - run_traced_script: Executes reproduction scripts with tracing
//! - warden_tracer.py: Generated wrapper for Python sys.settrace tracing
//! - Error truncation: Parses stack traces, strips noise, returns summarized traces

use anyhow::{Result};
use std::path::{Path, PathBuf};
use std::fs;
use tracing::info;

use crate::sandbox::{SandboxExecutor, ExecutionResult, SandboxConfig};

/// The warden_tracer.py template that wraps Python scripts with sys.settrace
const WARDEN_TRACER_TEMPLATE: &str = r##"#!/usr/bin/env python3
"""
Warden Execution Tracer - Auto-generated wrapper for Python hot path tracing
"""
import sys
import traceback

# Global tracer state
_hot_path = []
_enabled = True

def tracer(frame, event, arg):
    """sys.settrace handler that captures function call events."""
    global _hot_path, _enabled
    
    if not _enabled:
        return tracer
    
    if event == 'call':
        # Get function info
        filename = frame.f_code.co_filename
        func_name = frame.f_code.co_name
        lineno = frame.f_lineno
        
        # Skip standard library and framework code
        skip_prefixes = (
            '/usr/lib/python',
            '/usr/local/lib/python',
            '<frozen ',
            '<string>',
        )
        if any(filename.startswith(p) for p in skip_prefixes):
            return tracer
        
        # Record the call
        _hot_path.append({
            'event': 'call',
            'filename': filename,
            'func_name': func_name,
            'lineno': lineno,
        })
        
        # Limit size to prevent memory issues
        if len(_hot_path) > 10000:
            _hot_path.clear()
            _enabled = False
            print("# Tracer: Hot path overflow, truncating", file=sys.stderr)
            return tracer
        
    elif event == 'return':
        # Optionally record returns for deeper analysis
        pass
    
    return tracer

def run_traced(script_path, script_args):
    """Run a Python script with tracing enabled."""
    global _hot_path, _enabled
    
    _hot_path = []
    _enabled = True
    
    # Enable the tracer before running
    sys.settrace(tracer)
    
    try:
        # Execute the script
        with open(script_path, 'r') as f:
            script_code = compile(f.read(), script_path, 'exec')
        
        # Run with provided args
        globals_dict = {'__name__': '__main__', '__file__': script_path}
        globals_dict['__argvb__'] = script_args
        
        exec(script_code, globals_dict)
        
    except SystemExit as e:
        # Scripts can call sys.exit()
        pass
    except Exception as e:
        # Print the traceback to stderr
        traceback.print_exc()
    finally:
        # Disable the tracer
        sys.settrace(None)
        
        # Output the hot path as structured data (to stderr so stdout stays clean)
        print("\n# ===== WARDEN HOT PATH TRACING =====", file=sys.stderr)
        print(f"# Total events recorded: {len(_hot_path)}", file=sys.stderr)
        print("# Format: FILE:LINE FUNC_NAME", file=sys.stderr)
        print("# " + "=" * 40, file=sys.stderr)
        
        for entry in _hot_path:
            print(f"{entry['filename']}:{entry['lineno']} {entry['func_name']}", file=sys.stderr)
        
        print("# ===== END WARDEN HOT PATH =====", file=sys.stderr)
    
    return 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: warden_tracer.py <script.py> [args...]", file=sys.stderr)
        sys.exit(1)
    
    script_path = sys.argv[1]
    script_args = sys.argv[2:]
    
    sys.exit(run_traced(script_path, script_args))
"##;

pub struct ExecutionTracer {
    sandbox_executor: Option<SandboxExecutor>,
    workspace_path: PathBuf,
}

impl Clone for ExecutionTracer {
    fn clone(&self) -> Self {
        Self {
            sandbox_executor: self.sandbox_executor.clone(),
            workspace_path: self.workspace_path.clone(),
        }
    }
}

impl ExecutionTracer {
    pub fn new() -> Self {
        Self {
            sandbox_executor: None,
            workspace_path: PathBuf::from("/tmp"),
        }
    }

    pub fn with_workspace(workspace_path: PathBuf) -> Self {
        let executor = SandboxExecutor::new(workspace_path.clone());
        Self {
            sandbox_executor: Some(executor),
            workspace_path,
        }
    }

    /// Generate the warden_tracer.py wrapper file in the workspace
    fn ensure_tracer_script(&self) -> Result<PathBuf> {
        let tracer_path = self.workspace_path.join("warden_tracer.py");
        
        // Only write if it doesn't exist or content differs
        let should_write = if tracer_path.exists() {
            if let Ok(existing) = fs::read_to_string(&tracer_path) {
                existing != WARDEN_TRACER_TEMPLATE
            } else {
                true
            }
        } else {
            true
        };
        
        if should_write {
            fs::write(&tracer_path, WARDEN_TRACER_TEMPLATE)?;
            info!(path = %tracer_path.display(), "Generated warden_tracer.py");
        }
        
        Ok(tracer_path)
    }

    /// Run a test with tracing - uses language-specific correct commands
    pub async fn run_test(&self, test_path: &str, timeout_secs: u64) -> Result<ExecutionResult> {
        let sandbox = self.sandbox_executor.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No sandbox configured"))?;

        let mut config = SandboxConfig::default();
        config.timeout_seconds = timeout_secs;
        config.working_directory = Some(self.workspace_path.clone());

        let test_file = Path::new(test_path);
        let ext = test_file.extension().and_then(|e| e.to_str()).unwrap_or("");

        let command = match ext {
            "py" => vec!["python3".to_string(), test_path.to_string()],
            "js" => {
                // Check if jest is configured
                if self.workspace_path.join("package.json").exists() {
                    vec!["npx".to_string(), "jest".to_string(), test_path.to_string()]
                } else {
                    vec!["node".to_string(), test_path.to_string()]
                }
            }
            "ts" => {
                // Try tsx/ts-node first, fall back to jest
                vec!["npx".to_string(), "tsx".to_string(), test_path.to_string()]
            }
            "rs" => {
                // Cargo test uses test name filter, not file path.
                // Extract test name from path as best-effort.
                let test_name = test_file.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or(test_path);
                vec!["cargo".to_string(), "test".to_string(), "--".to_string(), test_name.to_string()]
            }
            _ => vec![test_path.to_string()],
        };

        sandbox.execute_traced(&command, Some(config)).await
    }

    /// Run a traced script with full sandbox isolation using warden_tracer.py for Python
    pub async fn run_traced_script(&self, command: &[String], working_dir: Option<PathBuf>) -> Result<ExecutionResult> {
        let sandbox = self.sandbox_executor.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No sandbox configured"))?;

        let mut config = SandboxConfig::default();
        config.working_directory = working_dir.or(Some(self.workspace_path.clone()));

        // Check if this is a Python script
        let is_python = command.first().and_then(|p| {
            Path::new(p).extension().and_then(|e| e.to_str())
        }).map(|ext| ext == "py").unwrap_or(false);

        let final_command = if is_python {
            // For Python, we need to use our tracer wrapper
            // Ensure the tracer script exists
            let _tracer_path = self.ensure_tracer_script()?;
            
            // Build the command to run the script through warden_tracer.py
            let script_path = command.get(0).map(String::from).unwrap_or_default();
            let script_args: Vec<String> = command.get(1..).unwrap_or(&[]).to_vec();
            
            let mut traced_cmd = vec!["python3".to_string(), "warden_tracer.py".to_string()];
            traced_cmd.push(script_path);
            traced_cmd.extend(script_args);
            
            traced_cmd
        } else {
            command.to_vec()
        };

        let result = sandbox.execute_traced(&final_command, Some(config)).await?;

        // trace_output is already set by execute_traced in sandbox.rs;
        // no need to duplicate extraction here.

        Ok(result)
    }

    /// Run lint check. When specific files are provided, lint only those files
    /// (by passing them as arguments to the lint command). Otherwise lints the entire project.
    pub async fn run_lint(&self, files: Option<&[String]>) -> Result<ExecutionResult> {
        // Detect lint command based on project files
        let mut lint_cmd = if self.workspace_path.join("package.json").exists() {
            vec!["npm".to_string(), "run".to_string(), "lint".to_string()]
        } else if self.workspace_path.join("pyproject.toml").exists() || self.workspace_path.join("setup.py").exists() {
            vec!["python3".to_string(), "-m".to_string(), "flake8".to_string()]
        } else if self.workspace_path.join("Cargo.toml").exists() {
            vec!["cargo".to_string(), "clippy".to_string()]
        } else {
            return Err(anyhow::anyhow!("Cannot detect lint command for project type"));
        };

        // Append specific files when provided
        if let Some(file_list) = files {
            lint_cmd.extend(file_list.iter().map(String::from));
        }

        let sandbox = self.sandbox_executor.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No sandbox configured"))?;

        let mut config = SandboxConfig::default();
        config.timeout_seconds = 120;

        sandbox.execute(&lint_cmd, Some(config)).await
    }

    /// Run compile/syntax check with language-specific correct commands
    pub async fn run_compile(&self, files: Option<&[String]>) -> Result<ExecutionResult> {
        let file_list = files.unwrap_or(&[]);

        if file_list.is_empty() {
            return Err(anyhow::anyhow!("No files specified for compilation check"));
        }

        let ext = Path::new(&file_list[0])
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let verify_cmd = match ext {
            "py" => vec!["python3".to_string(), "-m".to_string(), "py_compile".to_string(), file_list[0].clone()],
            "js" => vec!["node".to_string(), "--check".to_string(), file_list[0].clone()],
            "ts" => vec!["npx".to_string(), "tsc".to_string(), "--noEmit".to_string()],
            "rs" => vec!["cargo".to_string(), "check".to_string()],
            _ => return Err(anyhow::anyhow!("Unknown file type for compilation check: {}", ext)),
        };

        let sandbox = self.sandbox_executor.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No sandbox configured"))?;

        sandbox.execute(&verify_cmd, Some(SandboxConfig::default())).await
    }

    /// Run all tests
    pub async fn run_all_tests(&self, filter: Option<&str>) -> Result<ExecutionResult> {
        // Detect test command based on project files
        let test_cmd = if self.workspace_path.join("package.json").exists() {
            let mut cmd = vec!["npm".to_string(), "test".to_string()];
            if let Some(f) = filter {
                cmd.extend(["--".to_string(), "--testPathPattern".to_string(), f.to_string()]);
            }
            cmd
        } else if self.workspace_path.join("pyproject.toml").exists() || self.workspace_path.join("setup.py").exists() {
            let mut cmd = vec!["python3".to_string(), "-m".to_string(), "pytest".to_string()];
            if let Some(f) = filter {
                cmd.push(f.to_string());
            }
            cmd
        } else if self.workspace_path.join("Cargo.toml").exists() {
            let mut cmd = vec!["cargo".to_string(), "test".to_string()];
            if let Some(f) = filter {
                cmd.extend(["--".to_string(), f.to_string()]);
            }
            cmd
        } else {
            return Err(anyhow::anyhow!("Cannot detect test command for project type"));
        };

        let sandbox = self.sandbox_executor.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No sandbox configured"))?;

        let mut config = SandboxConfig::default();
        config.timeout_seconds = 300; // 5 minutes for full test suite

        sandbox.execute(&test_cmd, Some(config)).await
    }

    /// Get the error summary from a failed execution
    #[allow(dead_code)]
    pub fn summarize_error(stderr: &str, language: &str) -> String {
        crate::sandbox::summarize_error(stderr, language)
    }
}

impl Default for ExecutionTracer {
    fn default() -> Self {
        Self::new()
    }
}