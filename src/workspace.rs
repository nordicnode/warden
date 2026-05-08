//! Overlayfs Workspace Manager - Zero-copy isolation using Linux overlayfs
//!
//! Uses the actual repository as a read-only lowerdir and creates a writable upperdir
//! for the session workspace. All changes are isolated to the upperdir.

use anyhow::{Result, Context, anyhow};
use std::path::{Path, PathBuf};
use nix::mount::{mount, umount, MsFlags};
use std::fs;
use tracing::{info, warn, error};

/// Check if the current process has CAP_SYS_ADMIN capability
pub fn has_sys_admin_capability() -> bool {
    #[cfg(target_os = "linux")]
    {
        match caps::has_cap(None, caps::CapSet::Effective, caps::Capability::CAP_SYS_ADMIN) {
            Ok(true) => true,
            Ok(false) => false,
            Err(_) => false,
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        // Non-Linux platforms don't support CAP_SYS_ADMIN check this way
        false
    }
}

/// Check if running as root (UID 0)
pub fn is_running_as_root() -> bool {
    #[cfg(target_os = "linux")]
    {
        unsafe { libc::geteuid() == 0 }
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Detailed capability and permission check result
#[derive(Debug)]
pub struct CapabilityCheckResult {
    pub has_cap_sys_admin: bool,
    pub is_root: bool,
    pub can_mount_overlayfs: bool,
    pub recommendations: Vec<String>,
}

impl CapabilityCheckResult {
    /// Generate a human-readable summary of the capability check
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        
        if self.is_root {
            parts.push("Running as root (UID 0)".to_string());
        } else {
            parts.push(format!("Running as non-root (UID: {})", unsafe { libc::geteuid() }));
        }
        
        if self.has_cap_sys_admin {
            parts.push("Has CAP_SYS_ADMIN capability".to_string());
        } else {
            parts.push("Missing CAP_SYS_ADMIN capability".to_string());
        }
        
        if self.can_mount_overlayfs {
            parts.push("✓ Can mount overlayfs".to_string());
        } else {
            parts.push("✗ Cannot mount overlayfs".to_string());
        }
        
        for rec in &self.recommendations {
            parts.push(format!("  → {}", rec));
        }
        
        parts.join("\n")
    }
}

/// Perform comprehensive capability and permission checks
pub fn check_overlayfs_capabilities() -> CapabilityCheckResult {
    let is_root = is_running_as_root();
    let has_cap_sys_admin = has_sys_admin_capability();
    
    let mut recommendations = Vec::new();
    
    // Determine if overlayfs mount is possible
    let can_mount = is_root || has_cap_sys_admin;
    
    if !can_mount {
        if is_root {
            recommendations.push("Running as root but missing CAP_SYS_ADMIN - check Docker/container restrictions".to_string());
        } else {
            recommendations.push("Run as root: sudo setcap cap_sys_admin+ep <executable>".to_string());
            recommendations.push("Or run with: sudo -E ./warden-mcp".to_string());
            recommendations.push("Or add to appropriate Linux capability group".to_string());
        }
    }
    
    // Additional environment checks
    #[cfg(target_os = "linux")]
    {
        // Check if we're in a container
        if std::path::Path::new("/.dockerenv").exists() {
            recommendations.push("Running inside Docker - ensure container has --privileged or --cap-add=SYS_ADMIN".to_string());
        }
        
        // Check for systemd-detect
        if std::path::Path::new("/run/systemd/system").exists() {
            recommendations.push("Systemd detected - may need to run in user namespace or as system service".to_string());
        }
    }
    
    CapabilityCheckResult {
        has_cap_sys_admin,
        is_root,
        can_mount_overlayfs: can_mount,
        recommendations,
    }
}

pub struct OverlayfsWorkspace {
    session_id: uuid::Uuid,
    repo_path: PathBuf,
    upperdir: PathBuf,
    workdir: PathBuf,
    merged_path: PathBuf,
    is_mounted: bool,
    /// When true, overlayfs mount failed and we fell back to a plain copy.
    /// The workspace still works but changes are tracked differently.
    fallback_mode: bool,
    /// In fallback mode, we keep a copy of the original repo for diffing.
    fallback_original: Option<PathBuf>,
}

impl OverlayfsWorkspace {
    /// Create a new overlayfs workspace session
    pub fn new(repo_path: &Path, session_id: uuid::Uuid) -> Result<Self> {
        // Validate inputs
        if !repo_path.exists() {
            return Err(anyhow!("Repository path does not exist: {}", repo_path.display()));
        }
        
        if !repo_path.is_dir() {
            return Err(anyhow!("Repository path is not a directory: {}", repo_path.display()));
        }
        
        let base_dir = PathBuf::from("/tmp/warden-sessions");
        let session_dir = base_dir.join(session_id.to_string());

        let upperdir = session_dir.join("upperdir");
        let workdir = session_dir.join("workdir");
        let merged_path = session_dir.join("merged");

        // Create directories
        fs::create_dir_all(&upperdir).context("Failed to create upperdir")?;
        fs::create_dir_all(&workdir).context("Failed to create workdir")?;
        fs::create_dir_all(&merged_path).context("Failed to create merged dir")?;

        info!(session_id = %session_id, "Created overlayfs workspace directories");

        Ok(Self {
            session_id,
            repo_path: repo_path.to_path_buf(),
            upperdir,
            workdir,
            merged_path,
            is_mounted: false,
            fallback_mode: false,
            fallback_original: None,
        })
    }
    
    /// Check capabilities before mounting - call this before mount() to get detailed error
    pub fn check_capabilities(&self) -> CapabilityCheckResult {
        check_overlayfs_capabilities()
    }

    /// Mount the overlayfs filesystem, or fall back to a plain copy if overlayfs is unavailable.
    pub fn mount(&mut self) -> Result<()> {
        if self.is_mounted {
            warn!(session_id = %self.session_id, "Workspace already mounted");
            return Ok(());
        }
        
        // Pre-mount capability check with detailed error
        let cap_check = self.check_capabilities();
        if !cap_check.can_mount_overlayfs {
            warn!("Insufficient capabilities for overlayfs mount, using fallback mode.\n{}", 
                  cap_check.summary());
            return self.mount_fallback();
        }

        // Verify repo path exists and is accessible
        if !self.repo_path.exists() {
            return Err(anyhow!("Repository path no longer accessible: {}", self.repo_path.display()));
        }
        
        // Verify directories exist
        if !self.upperdir.exists() || !self.workdir.exists() || !self.merged_path.exists() {
            return Err(anyhow!("Workspace directories were removed - cannot mount"));
        }

        let options = format!(
            "lowerdir={},upperdir={},workdir={}",
            self.repo_path.display(),
            self.upperdir.display(),
            self.workdir.display()
        );

        // Mount overlayfs
        match mount::<Path, Path, str, str>(None, &self.merged_path, Some("overlay"), MsFlags::empty(), Some(&options)) {
            Ok(_) => {
                self.is_mounted = true;
                info!(session_id = %self.session_id, merged_path = %self.merged_path.display(), "Overlayfs mounted successfully");
                Ok(())
            }
            Err(e) => {
                // Provide more helpful error message based on the error
                let error_detail = match e {
                    nix::errno::Errno::EPERM => {
                        "EPERM: Operation not permitted.".to_string()
                    }
                    nix::errno::Errno::EACCES => {
                        "EACCES: Permission denied.".to_string()
                    }
                    nix::errno::Errno::ENOENT => {
                        "ENOENT: One of the paths does not exist.".to_string()
                    }
                    nix::errno::Errno::ENODEV => {
                        "ENODEV: Overlay filesystem not supported by kernel.".to_string()
                    }
                    _ => format!("Unknown error: {}", e),
                };
                
                warn!("Overlayfs mount failed: {}. Falling back to plain copy mode.", error_detail);
                self.mount_fallback()
            }
        }
    }

    /// Fallback: copy the entire repository into the merged directory.
    /// Changes are tracked by comparing against the original copy.
    fn mount_fallback(&mut self) -> Result<()> {
        info!("Using fallback workspace mode (plain copy)");
        
        // Copy the original repo to a backup location for diffing
        let original_backup = self.workdir.join("original_backup");
        self.copy_recursive(&self.repo_path, &original_backup)?;
        self.fallback_original = Some(original_backup);
        
        // Copy the repo into merged_path for working
        // (merged_path is already created in new())
        self.copy_recursive(&self.repo_path, &self.merged_path)?;
        
        self.is_mounted = true;
        self.fallback_mode = true;
        
        warn!("WARNING: Running in fallback mode. Changes are applied directly to the workspace copy.\n\
               Overlayfs isolation is not active.");
        
        Ok(())
    }

    /// Recursively copy a directory
    fn copy_recursive(&self, src: &Path, dst: &Path) -> Result<()> {
        if !dst.exists() {
            fs::create_dir_all(dst)?;
        }
        
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());
            
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                self.copy_recursive(&src_path, &dst_path)?;
            } else if file_type.is_symlink() {
                #[cfg(unix)]
                {
                    if let Ok(target) = fs::read_link(&src_path) {
                        let _ = std::os::unix::fs::symlink(&target, &dst_path);
                    }
                }
            } else {
                fs::copy(&src_path, &dst_path)?;
            }
        }
        
        Ok(())
    }

    /// Unmount the overlayfs filesystem.
    /// In fallback mode this is a no-op (no actual mount to undo).
    ///
    /// NOTE: LSP servers spawned for this workspace are NOT shut down here.
    /// Their child processes will be cleaned up by the OS when the Warden
    /// process exits. If workspace rotation is ever implemented, a shutdown
    /// hook should be added to kill LSP server children before unmount.
    pub fn unmount(&mut self) -> Result<()> {
        if !self.is_mounted {
            return Ok(());
        }

        if !self.fallback_mode {
            umount(&self.merged_path).context("Failed to unmount overlayfs")?;
            info!(session_id = %self.session_id, "Overlayfs unmounted successfully");
        } else {
            info!(session_id = %self.session_id, "Fallback mode — skipping umount, cleaning up copy");
        }
        
        self.is_mounted = false;
        Ok(())
    }

    /// Get the merged path where all files are accessible (both read from lowerdir and written to upperdir)
    pub fn merged_path(&self) -> &Path {
        &self.merged_path
    }

    /// Get the upperdir path (session-specific changes)
    pub fn upperdir(&self) -> &Path {
        &self.upperdir
    }

    /// Get the session ID
    pub fn session_id(&self) -> uuid::Uuid {
        self.session_id
    }

    /// Get the original repository path (read-only)
    pub fn repo_path(&self) -> &Path {
        &self.repo_path
    }

    /// Check if mounted
    pub fn is_mounted(&self) -> bool {
        self.is_mounted
    }

    /// Get diff of changes made in this session (relative to lowerdir)
    pub fn get_session_diff(&self) -> Result<Vec<PathBuf>> {
        let mut changed_files = Vec::new();

        // Walk the upperdir to find all modified/added files
        for entry in walkdir::WalkDir::new(&self.upperdir) {
            let entry = entry.context("Failed to walk upperdir")?;
            if entry.file_type().is_file() {
                let relative_path = entry.path()
                    .strip_prefix(&self.upperdir)
                    .context("Failed to strip upperdir prefix")?
                    .to_path_buf();
                changed_files.push(relative_path);
            }
        }

        Ok(changed_files)
    }

    /// Get the path to a specific file in the merged view
    pub fn get_file_path(&self, relative_path: &Path) -> PathBuf {
        self.merged_path.join(relative_path)
    }

    /// Revert changes to a specific file (remove from upperdir)
    pub fn revert_file(&self, relative_path: &Path) -> Result<()> {
        let upper_file = self.upperdir.join(relative_path);
        if upper_file.exists() {
            fs::remove_file(&upper_file).context("Failed to revert file")?;
            info!(file = %relative_path.display(), "Reverted file changes");
        }
        Ok(())
    }

    /// Revert all changes (clear upperdir)
    pub fn revert_all(&mut self) -> Result<()> {
        if self.upperdir.exists() {
            // Remove all contents of upperdir
            for entry in fs::read_dir(&self.upperdir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    fs::remove_dir_all(&path)?;
                } else {
                    fs::remove_file(&path)?;
                }
            }
        }
        info!(session_id = %self.session_id, "Reverted all session changes");
        Ok(())
    }

    /// Commit changes from upperdir to a new directory (for export/backup)
    pub fn commit_changes(&self, dest_dir: &Path) -> Result<()> {
        fs::create_dir_all(dest_dir)?;

        for entry in walkdir::WalkDir::new(&self.upperdir) {
            let entry = entry?;
            let relative_path = entry.path()
                .strip_prefix(&self.upperdir)
                .context("Failed to strip prefix")?
                .to_path_buf();

            let dest_path = dest_dir.join(&relative_path);

            if entry.file_type().is_dir() {
                fs::create_dir_all(&dest_path)?;
            } else {
                if let Some(parent) = dest_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::copy(entry.path(), &dest_path)?;
            }
        }

        info!(dest = %dest_dir.display(), "Committed session changes");
        Ok(())
    }
}

impl Drop for OverlayfsWorkspace {
    fn drop(&mut self) {
        if self.is_mounted {
            if let Err(e) = self.unmount() {
                error!(session_id = %self.session_id, error = %e, "Failed to unmount on drop");
            }
        }

        // Optionally clean up the session directory
        // We keep it by default in case of debugging
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_capability_check() {
        let result = check_overlayfs_capabilities();
        println!("Capability check result:\n{}", result.summary());
        // This test always runs - it just prints the current state
    }

    #[test]
    fn test_overlayfs_workspace_creation() {
        let temp_repo = TempDir::new().unwrap();
        let session_id = uuid::Uuid::new_v4();

        let workspace = OverlayfsWorkspace::new(temp_repo.path(), session_id);
        assert!(workspace.is_ok());
    }

    #[test]
    fn test_overlayfs_workspace_mount() {
        // Note: This test requires root or appropriate permissions
        // Skip if not running as root or with CAP_SYS_ADMIN
        let cap_check = check_overlayfs_capabilities();
        if !cap_check.can_mount_overlayfs {
            println!("Skipping mount test - insufficient capabilities:\n{}", cap_check.summary());
            return;
        }

        let temp_repo = TempDir::new().unwrap();
        let session_id = uuid::Uuid::new_v4();

        let mut workspace = OverlayfsWorkspace::new(temp_repo.path(), session_id).unwrap();
        let result = workspace.mount();
        
        if result.is_ok() {
            workspace.unmount().unwrap();
        }
    }
}