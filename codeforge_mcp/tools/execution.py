"""Execution tools — bash_run, test_run.

bash_run is sandboxed via bubblewrap (bwrap) when available, with a configurable timeout.
Dangerous commands (rm, sudo, chmod, etc.) trigger a confirmation prompt.
test_run auto-detects the test runner (pytest/vitest/cargo test) and parses failures.
"""

from __future__ import annotations

import asyncio
import json as _json
import os
import re
import shlex
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from codeforge_mcp.logging import audit_dangerous_execution

# Patterns for commands that require confirmation
_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"\brm(?!dir)\b", "rm (remove files)"),
    (r"\bsudo\b", "sudo (superuser)"),
    (r"\bchmod\b", "chmod (change permissions)"),
    (r"\bchown\b", "chown (change ownership)"),
    (r"\bmkfs\b", "mkfs (format filesystem)"),
    (r"\bdd\b", "dd (disk copy)"),
    (r"\bkill\b", "kill (terminate process)"),
    (r"\bshutdown\b", "shutdown"),
    (r"\breboot\b", "reboot"),
    (r"\bfork\s*bomb\b", "fork bomb"),
    (r">\s*/dev/", "write to device"),
    (r"\bmv\b.*\s+/(etc|bin|sbin|usr|lib|lib64|boot|dev|proc|sys|root|run|srv|var|opt)(/|$)", "mv (move to system directory)"),
    (r"\bmv\b.*\s+/$", "mv (move to root directory)"),
    (r"\bgit\s+push\b.*--force", "git push --force"),
    (r"\bgit\s+reset\b.*--hard", "git reset --hard"),
    (r"\bdocker\s+(rm|prune|system\s+prune)\b", "destructive docker command"),
    (r"\bnpm\s+(unpublish|deprecate)\b", "npm unpublish/deprecate"),
    (r"\bpip\s+uninstall\b", "pip uninstall"),
    (r"\bpacman\s+-R", "pacman remove"),
    (r"curl.*\|\s*(ba)?sh", "curl-pipe-bash"),
    (r"wget.*\|\s*(ba)?sh", "wget-pipe-bash"),
]

def _is_dangerous(cmd: str) -> list[str]:
    """Check if a command matches dangerous patterns. Returns list of reasons."""
    reasons: list[str] = []
    for pattern, reason in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd):
            reasons.append(reason)
    return reasons


async def bash_run(
    cmd: str,
    cwd: str | None = None,
    timeout: int = 30,
    project_root: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Run a bash command, optionally sandboxed with bubblewrap.

    Args:
        cmd: The shell command to run.
        cwd: Working directory (default: project_root).
        timeout: Maximum execution time in seconds.
        project_root: Project root for sandboxing.
        confirmed: Set to True to confirm dangerous commands.

    Returns:
        {exit_code, stdout, stderr, timed_out, sandboxed, dangerous, confirmation_required}
    """
    work_dir = Path(cwd or project_root or ".")

    # Check for dangerous commands
    dangers = _is_dangerous(cmd)
    if dangers and not confirmed:
        audit_dangerous_execution(
            cmd=cmd,
            dangers=dangers,
            cwd=str(work_dir),
            outcome="blocked",
        )
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"DANGEROUS COMMAND DETECTED. Reasons: {', '.join(dangers)}. "
                       f"Set confirmed=true to execute anyway, or use a safer alternative.",
            "timed_out": False,
            "sandboxed": False,
            "dangerous": True,
            "confirmation_required": True,
            "danger_reasons": dangers,
        }

    # Try bwrap sandbox if available
    sandbox_cmd = _build_sandbox_command(cmd, work_dir, project_root)
    actual_cmd = sandbox_cmd if sandbox_cmd else ["bash", "-c", cmd]
    sandboxed = sandbox_cmd is not None
    t0 = time.time()

    env = os.environ.copy()
    venv_bin = str(work_dir / ".venv" / "bin")
    node_bin = str(work_dir / "node_modules" / ".bin")
    env["PATH"] = f"{venv_bin}:{node_bin}:{env.get('PATH', '')}"

    # Generate a global .ripgreprc to protect raw LLM ripgrep calls.
    # Use mkstemp with O_EXCL semantics so the filename is unpredictable
    # and the call fails (rather than overwrites) if the file already exists.
    rc_fd = None
    rc_path = None
    try:
        rc_fd, rc_path = tempfile.mkstemp(prefix=".codeforge_ripgreprc_")
        os.write(rc_fd, b"\n".join(
            f"--glob=!{d}/\n--glob=!**/{d}/**\n".encode()
            for d in [".venv", "venv", "env", "node_modules", "dist", "build", ".git", ".codeforge"]
        ))
        os.close(rc_fd)
        rc_fd = None
        env["RIPGREP_CONFIG_PATH"] = rc_path
    finally:
        if rc_fd is not None:
            try:
                os.close(rc_fd)
            except OSError:
                pass
        if rc_path is not None:
            try:
                os.unlink(rc_path)
            except OSError:
                pass

    try:
        proc = await asyncio.create_subprocess_exec(
            *actual_cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(work_dir),
            env=env,
            start_new_session=True,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            timed_out = False
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout_bytes, stderr_bytes = await proc.communicate()
            timed_out = True

        await proc.wait()

        duration_ms = (time.time() - t0) * 1000
        exit_code = proc.returncode or 0

        if dangers:
            audit_dangerous_execution(
                cmd=cmd,
                dangers=dangers,
                cwd=str(work_dir),
                outcome="completed" if exit_code == 0 else "error",
                exit_code=exit_code,
                duration_ms=duration_ms,
            )

        return {
            "exit_code": exit_code,
            "stdout": stdout_bytes.decode(errors="replace")[:10000],
            "stderr": stderr_bytes.decode(errors="replace")[:10000],
            "timed_out": timed_out,
            "sandboxed": sandboxed,
            "dangerous": bool(dangers),
            "confirmation_required": False,
        }
    except FileNotFoundError:
        duration_ms = (time.time() - t0) * 1000
        if dangers:
            audit_dangerous_execution(
                cmd=cmd,
                dangers=dangers,
                cwd=str(work_dir),
                outcome="error",
                exit_code=127,
                duration_ms=duration_ms,
            )
        return {
            "exit_code": 127,
            "stdout": "",
            "stderr": f"Command not found: {actual_cmd[0]}",
            "timed_out": False,
            "sandboxed": sandboxed,
            "dangerous": bool(dangers),
            "confirmation_required": False,
        }


def _build_sandbox_command(cmd: str, cwd: Path, project_root: str | None) -> list[str] | None:
    """Build a bwrap sandbox command when CODEFORGE_SANDBOX=1 and bwrap is available."""
    if os.environ.get("CODEFORGE_SANDBOX") != "1":
        return None
    import shutil
    if shutil.which("bwrap") is None:
        return None
    # Build sandbox: ro-bind root, rw-bind project, isolate network, private /tmp
    return [
        "bwrap",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/sbin", "/sbin",
        "--ro-bind", "/etc", "/etc",
        "--bind", str(cwd), str(cwd),
        "--chdir", str(cwd),
        "--tmpfs", "/tmp",
        "--unshare-net",
        "--die-with-parent",
        "--new-session",
        "bash", "-c", cmd,
    ]


def test_run(
    selector: str = "",
    project_root: str | None = None,
    summary_only: bool = False,
) -> dict[str, Any]:
    """Run tests using auto-detected test runner.

    Args:
        selector: Optional test selector (file, function, or filter).
        project_root: Project directory.
        summary_only: When True, the response omits per-test PASSED lines and
            keeps only the runner header, the failure section, and the final
            summary line (e.g. ``269 passed in 4.5s``). Use this for large
            test suites to keep the response well under the 10kB stdout cap.

    Returns:
        {runner, exit_code, stdout, stderr, failures: [{test, message}]}
        When ``summary_only`` is set, also includes ``summary`` (the
        condensed runner output) and ``stdout_truncated`` flags.
    """
    root = Path(project_root or ".")
    runner, runner_cmd = _detect_test_runner(root)

    if runner is None or runner_cmd is None:
        return {
            "runner": "unknown",
            "exit_code": 1,
            "stdout": "",
            "stderr": "No test runner detected. Supported: pytest, vitest, cargo test",
            "failures": [],
        }

    # Individual quoting for selector parts to allow multiple files
    safe_selector = " ".join(shlex.quote(p) for p in selector.split()) if selector else ""
    cmd = _build_test_command(runner, runner_cmd, safe_selector)

    # Add venv/bin and node_modules/.bin to PATH
    env = os.environ.copy()
    venv_bin = str(root / ".venv" / "bin")
    node_bin = str(root / "node_modules" / ".bin")
    env["PATH"] = f"{venv_bin}:{node_bin}:{env.get('PATH', '')}"

    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(root), env=env, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True
        )
        stdout, stderr = proc.communicate(timeout=60)
        failures = _parse_test_failures(runner, stdout + stderr)
        return _format_test_response(
            runner=runner, cmd=cmd, exit_code=proc.returncode,
            stdout=stdout, stderr=stderr, failures=failures,
            summary_only=summary_only,
        )
    except subprocess.TimeoutExpired:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        else:
            stdout, stderr = "", ""
        return {
            "runner": runner,
            "exit_code": -1,
            "stdout": stdout[:10000],
            "stderr": "Timeout after 60s\n" + stderr[:10000],
            "failures": [],
        }
    except FileNotFoundError:
        return {
            "runner": runner,
            "exit_code": 127,
            "stdout": "",
            "stderr": f"Command not found: {runner_cmd}",
            "failures": [],
        }


def _format_test_response(
    *,
    runner: str,
    cmd: str,
    exit_code: int,
    stdout: str,
    stderr: str,
    failures: list[dict[str, Any]],
    summary_only: bool,
) -> dict[str, Any]:
    """Build the test_run response, optionally condensing stdout."""
    if not summary_only:
        return {
            "runner": runner,
            "command": cmd,
            "exit_code": exit_code,
            "stdout": stdout[:10000],
            "stderr": stderr[:10000],
            "failures": failures,
        }

    # Condense: keep the session header, any FAILED/ERROR lines, and the
    # final summary lines. Drop the per-test PASSED markers that dominate
    # large suites.
    summary_lines: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if "PASSED" in line and "FAILED" not in line and "ERROR" not in line:
            continue
        if line.startswith("collecting") or line.startswith("collected"):
            summary_lines.append(line)
            continue
        # Header lines, separators, failure breakdown, final summary
        if (
            line.startswith("=")
            or line.startswith("_")
            or "FAILED" in line
            or "ERROR" in line
            or line.startswith("rootdir:")
            or line.startswith("plugins:")
            or line.startswith("platform")
            or "passed" in line
            or "failed" in line
            or "skipped" in line
        ):
            summary_lines.append(line)

    summary_text = "\n".join(summary_lines)[:8000]
    return {
        "runner": runner,
        "command": cmd,
        "exit_code": exit_code,
        "summary": summary_text,
        "stdout": summary_text,  # callers expecting `stdout` still work
        "stderr": stderr[:5000],
        "failures": failures,
        "summary_only": True,
        "stdout_lines_dropped": stdout.count("PASSED"),
    }


def _detect_test_runner(root: Path) -> tuple[str | None, str | None]:
    """Detect the test runner and its command from project config files.

    Returns (runner_name, executable_command) or (None, None).
    """
    # Prefer .venv/bin/python if it exists
    venv_python = root / ".venv" / "bin" / "python"
    python_cmd = str(venv_python) if venv_python.exists() else "python"

    # pytest
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text()
            if "pytest" in content or "tool.pytest" in content:
                return ("pytest", f"{python_cmd} -m pytest")
        except (OSError, PermissionError):
            pass
    if (root / "pytest.ini").exists():
        return ("pytest", f"{python_cmd} -m pytest")
    
    # Check for test files
    if pyproject.exists() or list(root.rglob("test_*.py")) or list(root.rglob("*_test.py")):
        if list(root.rglob("test_*.py")) or list(root.rglob("*_test.py")):
            return ("pytest", f"{python_cmd} -m pytest")

    # vitest — check local node_modules first
    if (root / "vitest.config.ts").exists() or (root / "vitest.config.js").exists():
        local_vitest = root / "node_modules" / ".bin" / "vitest"
        if local_vitest.exists():
            return ("vitest", str(local_vitest))
        return ("vitest", "npx vitest")

    if (root / "package.json").exists():
        try:
            pkg = _json.loads((root / "package.json").read_text())
            if "vitest" in str(pkg.get("devDependencies", {})):
                local_vitest = root / "node_modules" / ".bin" / "vitest"
                if local_vitest.exists():
                    return ("vitest", str(local_vitest))
                return ("vitest", "npx vitest")
            # jest
            if "jest" in str(pkg.get("devDependencies", {})):
                local_jest = root / "node_modules" / ".bin" / "jest"
                if local_jest.exists():
                    return ("jest", str(local_jest))
                return ("jest", "npx jest")
        except (OSError, PermissionError):
            pass

    # cargo test
    if (root / "Cargo.toml").exists():
        return ("cargo test", "cargo test")

    return (None, None)


def _build_test_command(runner: str, runner_cmd: str, selector: str) -> str:
    """Build the test command from runner, executable, and selector."""
    if runner == "pytest":
        cmd = f"{runner_cmd} -v"
        if selector:
            cmd += f" {selector}"
        return cmd
    elif runner in ("vitest", "jest"):
        cmd = f"{runner_cmd} run" if "vitest" in runner_cmd else f"{runner_cmd} --no-cache"
        if selector:
            cmd += f" {selector}"
        return cmd
    elif runner == "cargo test":
        cmd = runner_cmd
        if selector:
            cmd += f" {selector}"
        return cmd
    return runner_cmd


def _parse_test_failures(runner: str, output: str) -> list[dict[str, Any]]:
    """Parse test output for failure information."""
    failures: list[dict] = []
    lines = output.split("\n")

    if runner == "pytest":
        for line in lines:
            if "FAILED" in line:
                failures.append({"test": line.strip(), "message": ""})
    elif runner in ("vitest", "jest"):
        for i, line in enumerate(lines):
            if "FAIL" in line and ("test" in line.lower() or "●" in line):
                msg = lines[i + 1].strip() if i + 1 < len(lines) else ""
                failures.append({"test": line.strip(), "message": msg})
    elif runner == "cargo test":
        for i, line in enumerate(lines):
            if "FAILED" in line:
                failures.append({"test": line.strip(), "message": ""})

    return failures
