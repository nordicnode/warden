//! Sandbox Executor - Bubblewrap-based code execution sandbox
//!
//! Provides secure code execution using Linux bubblewrap with overlayfs isolation.
//!
//! FIXES APPLIED:
//! - Bind-mounts /usr/bin, /lib, /lib64 (read-only) so interpreters/tools are available.
//! - Removed duplicate `environment_vars` field; only `env_vars` remains.
//! - Made SandboxExecutor Clone.
//! - build_bwrap_command mounts host binaries/libs when allow_host_tools is set.

use anyhow::{Result, Context};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;
use tracing::info;

// NOTE: std::os::unix::fs::MetadataExt is only used by is_root() which is gated on cfg(target_os = "linux")

/// Sandbox executor using bubblewrap
#[derive(Clone)]
pub struct SandboxExecutor {
    pub bwrap_path: PathBuf,
    config: SandboxConfig,
    workspace_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    #[allow(dead_code)]
    pub readonly_rootfs: bool,
    pub disable_network: bool,
    pub disable_mount_home: bool,
    pub allowed_paths: Vec<String>,
    pub env_vars: Vec<(String, String)>,
    pub timeout_seconds: u64,
    pub working_directory: Option<PathBuf>,
    /// When true, bind-mount host /usr/bin, /lib, /lib64 (read-only) so that
    /// interpreters (python3, node, etc.) and tools (patch, cargo) are available.
    pub allow_host_tools: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            readonly_rootfs: true,
            disable_network: true,
            disable_mount_home: true,
            allowed_paths: vec!["/tmp".to_string()],
            env_vars: vec![],
            timeout_seconds: 30,
            working_directory: None,
            allow_host_tools: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub execution_time_ms: u64,
    pub trace_output: Option<String>,
}

impl SandboxExecutor {
    pub fn new(workspace_path: PathBuf) -> Self {
        let bwrap_path = PathBuf::from("/usr/bin/bwrap");
        Self {
            bwrap_path,
            config: SandboxConfig::default(),
            workspace_path,
        }
    }

    #[allow(dead_code)]
    pub fn with_bwrap(bwrap_path: PathBuf, workspace_path: PathBuf) -> Self {
        Self {
            bwrap_path,
            config: SandboxConfig::default(),
            workspace_path,
        }
    }

    #[allow(dead_code)]
    pub fn with_config(workspace_path: PathBuf, config: SandboxConfig) -> Self {
        let bwrap_path = PathBuf::from("/usr/bin/bwrap");
        Self {
            bwrap_path,
            config,
            workspace_path,
        }
    }

    pub fn is_bwrap_available(&self) -> bool {
        Command::new(&self.bwrap_path)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn build_bwrap_command(&self, program: &[String], args: &[String], config: &SandboxConfig) -> Command {
        let mut cmd = Command::new(&self.bwrap_path);

        // NOTE: --readonly-filesystem is not a valid bwrap flag.
        // Read-only rootfs is enforced via --ro-bind in the allow_host_tools section below.

        if config.disable_network {
            cmd.arg("--unshare-net");
        }

        if config.disable_mount_home {
            cmd.arg("--lock-file=/tmp/warden.lock");
        }

        // Bind-mount host tools (read-only) so interpreters/compilers are available
        if config.allow_host_tools {
            // Mount /usr (contains /usr/bin, /usr/lib, etc.)
            if std::path::Path::new("/usr").exists() {
                cmd.arg("--ro-bind").arg("/usr").arg("/usr");
            }
            // Mount standard library paths
            if std::path::Path::new("/lib").exists() {
                cmd.arg("--ro-bind").arg("/lib").arg("/lib");
            }
            if std::path::Path::new("/lib64").exists() {
                cmd.arg("--ro-bind").arg("/lib64").arg("/lib64");
            }
            // Mount /bin (symlink to /usr/bin on many systems)
            if std::path::Path::new("/bin").exists() && !std::path::Path::new("/bin").is_symlink() {
                cmd.arg("--ro-bind").arg("/bin").arg("/bin");
            }
            // Mount /etc for resolver config, ca-certificates, etc.
            if std::path::Path::new("/etc").exists() {
                cmd.arg("--ro-bind").arg("/etc").arg("/etc");
            }
        }

        cmd.arg("--bind").arg(&self.workspace_path).arg("/workspace");
        
        let chdir_path = config.working_directory.as_ref()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "/workspace".to_string());
        
        cmd.arg("--chdir").arg(chdir_path);

        for path in &config.allowed_paths {
            cmd.arg("--bind").arg(path).arg(path);
        }

        // Set environment variables
        for (k, v) in &config.env_vars {
            cmd.arg("--setenv").arg(k).arg(v);
        }

        // Set default PATH so interpreters can be found
        cmd.arg("--setenv").arg("PATH").arg("/usr/bin:/usr/local/bin:/bin");
        cmd.arg("--setenv").arg("HOME").arg("/tmp");

        cmd.arg("--proc").arg("/proc");
        cmd.arg("--dev").arg("/dev");

        cmd.arg("--").args(program).args(args);

        cmd
    }

    pub async fn execute(&self, program: &[String], config: Option<SandboxConfig>) -> Result<ExecutionResult> {
        if !self.is_bwrap_available() {
            return Err(anyhow::anyhow!("bwrap not available at {}", self.bwrap_path.display()));
        }

        let actual_config = config.unwrap_or_else(|| self.config.clone());
        let start = std::time::Instant::now();

        let (bin, rest) = if program.is_empty() {
            (vec!["/bin/sh".to_string()], vec![])
        } else {
            (vec![program[0].clone()], program[1..].to_vec())
        };

        let mut cmd = self.build_bwrap_command(&bin, &rest, &actual_config);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        info!(program = ?program, "Executing in sandbox");

        let mut tokio_cmd = tokio::process::Command::from(cmd);
        tokio_cmd.kill_on_drop(true);
        
        let child = tokio_cmd.spawn()
            .context("Failed to spawn sandboxed process")?;
        
        let timeout = Duration::from_secs(actual_config.timeout_seconds);
        
        let result = tokio::time::timeout(timeout, child.wait_with_output()).await;
        
        match result {
            Ok(Ok(output)) => {
                Ok(ExecutionResult {
                    exit_code: output.status.code().unwrap_or(-1),
                    stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                    stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                    execution_time_ms: start.elapsed().as_millis() as u64,
                    trace_output: None,
                })
            }
            Ok(Err(e)) => Err(anyhow::anyhow!("Process execution failed: {}", e)),
            Err(_) => {
                Err(anyhow::anyhow!("Process timed out after {}s", actual_config.timeout_seconds))
            }
        }
    }

    pub async fn execute_traced(&self, program: &[String], config: Option<SandboxConfig>) -> Result<ExecutionResult> {
        let mut result = self.execute(program, config).await?;
        
        // Always extract the warden trace from stderr regardless of exit code.
        // Successful Python executions also produce trace data that callers need.
        result.trace_output = Some(extract_warden_trace(&result.stderr));
        
        Ok(result)
    }

    #[allow(dead_code)]
    pub fn is_root(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::MetadataExt;
            std::fs::metadata("/").map(|m| m.uid() == 0).unwrap_or(false)
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }
}

pub fn summarize_error(error_output: &str, language: &str) -> String {
    let trace_lines: Vec<&str> = error_output.lines().collect();
    let mut summary = String::new();
    let mut found_specific = false;

    match language {
        "python" => {
            for line in &trace_lines {
                let trimmed = line.trim();
                if trimmed.starts_with("File ") && trimmed.contains(", line ") {
                    let parts: Vec<&str> = trimmed.split('"').collect();
                    if parts.len() >= 2 {
                        let file_part = parts[1].trim();
                        if let Some(line_part) = trimmed.split(", line ").nth(1) {
                            let line_num = line_part.split(',').next().unwrap_or("?");
                            summary.push_str(&format!("Crash at Line {} in {}\n", line_num, file_part));
                            found_specific = true;
                        }
                    }
                } else if trimmed.starts_with("Error:") 
                    || trimmed.starts_with("Exception:") 
                    || trimmed.ends_with("Error") 
                    || trimmed.ends_with("Exception")
                    || trimmed.starts_with("Traceback")
                    || trimmed.starts_with("AssertionError")
                    || trimmed.starts_with("AttributeError")
                    || trimmed.starts_with("TypeError")
                    || trimmed.starts_with("ValueError")
                    || trimmed.starts_with("KeyError")
                    || trimmed.starts_with("IndexError")
                    || trimmed.starts_with("ImportError")
                    || trimmed.starts_with("ModuleNotFoundError")
                    || trimmed.starts_with("NameError")
                    || trimmed.starts_with("SyntaxError") {
                    summary.push_str(&format!("Error: {}\n", trimmed));
                    found_specific = true;
                }
            }
            if !found_specific && !summary.is_empty() {
                // Keep what we found
            } else if summary.is_empty() {
                summary = trace_lines.iter().take(3).fold(String::new(), |mut s, l| {
                    s.push_str(l);
                    s.push('\n');
                    s
                });
            }
        }
        "javascript" | "typescript" => {
            for line in &trace_lines {
                let trimmed = line.trim();
                if trimmed.contains("Error:") 
                    || trimmed.contains("ReferenceError") 
                    || trimmed.contains("TypeError") 
                    || trimmed.contains("SyntaxError")
                    || trimmed.contains("RangeError")
                    || trimmed.contains("exception")
                    || trimmed.starts_with("at ")
                    || trimmed.contains("AssertionError") {
                    summary.push_str(trimmed);
                    summary.push('\n');
                    found_specific = true;
                }
            }
            if !found_specific {
                summary = error_output.lines().take(5).collect::<Vec<_>>().join("\n");
            }
        }
        "rust" => {
            for line in &trace_lines {
                let trimmed = line.trim();
                if trimmed.starts_with("error[") 
                    || trimmed.starts_with("error:")
                    || trimmed.starts_with("warning[")
                    || trimmed.contains("thread '") 
                    || trimmed.contains("panicked") {
                    summary.push_str(trimmed);
                    summary.push('\n');
                    found_specific = true;
                }
            }
            if !found_specific {
                summary = error_output.lines().take(5).collect::<Vec<_>>().join("\n");
            }
        }
        _ => {
            // Generic: capture lines that look like errors
            for line in &trace_lines {
                let trimmed = line.trim();
                let lower = trimmed.to_lowercase();
                if lower.contains("error:")
                    || lower.contains("exception")
                    || lower.contains("fail")
                    || lower.contains("panic")
                    || lower.contains("traceback")
                    || lower.contains("syntaxerror")
                    || lower.contains("typeerror") {
                    summary.push_str(trimmed);
                    summary.push('\n');
                    found_specific = true;
                }
            }
            if !found_specific {
                summary = error_output.lines().take(5).collect::<Vec<_>>().join("\n");
            }
        }
    }

    summary
}

/// Extract the warden trace markers from stderr output.
/// Looks for the `# ===== WARDEN HOT PATH TRACING =====` section.
pub fn extract_warden_trace(stderr: &str) -> String {
    let mut trace_lines = Vec::new();
    let mut in_warden_section = false;

    for line in stderr.lines() {
        if line.contains("# ===== WARDEN HOT PATH TRACING =====") {
            in_warden_section = true;
            trace_lines.push(line.to_string());
            continue;
        }
        if in_warden_section {
            if line.contains("# ===== END WARDEN HOT PATH =====") {
                trace_lines.push(line.to_string());
                break;
            }
            trace_lines.push(line.to_string());
        }
    }

    trace_lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summarize_error_python() {
        let trace = "Traceback (most recent call last):\n  File \"/app/myapp.py\", line 42, in process_data\n    result = data.get(key)\nAttributeError: 'NoneType' object has no attribute 'get'\n";
        let summary = summarize_error(trace, "python");
        assert!(summary.contains("myapp.py"));
        assert!(summary.contains("42"));
    }

    #[test]
    fn test_executor_default_config() {
        let executor = SandboxExecutor::new(
            PathBuf::from("/tmp/workspace"),
        );
        assert_eq!(executor.bwrap_path, PathBuf::from("/usr/bin/bwrap"));
        assert!(executor.config.readonly_rootfs);
        assert!(executor.config.disable_network);
    }
}
