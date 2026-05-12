"""Structured response models for MCP tool results.

Every tool response conforms to ToolResponse, which provides:
- success: whether the operation completed without error
- error_code: machine-readable error classification (for weak-model retry logic)
- error_message: human-readable error description
- data: the actual payload (tool-specific)

Error codes follow a hierarchical scheme:
  FILE_*    — filesystem operations
  GRAPH_*   — knowledge graph queries
  LSP_*     — language server protocol
  EXEC_*    — command execution
  SUBAGENT_*— subagent orchestration
  PARSE_*   — AST/tree-sitter
  INTERNAL  — unexpected server error
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    """Machine-readable error codes for retry/dispatch logic."""
    NONE = "NONE"

    # File operations
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_TRAVERSAL_DENIED = "FILE_TRAVERSAL_DENIED"
    FILE_PROTECTED = "FILE_PROTECTED"
    FILE_READ_ERROR = "FILE_READ_ERROR"
    FILE_WRITE_ERROR = "FILE_WRITE_ERROR"

    # Patch operations
    PATCH_HASH_MISMATCH = "PATCH_HASH_MISMATCH"
    PATCH_LINE_MISMATCH = "PATCH_LINE_MISMATCH"
    PATCH_PARSE_ERROR = "PATCH_PARSE_ERROR"

    # Graph
    GRAPH_SYMBOL_NOT_FOUND = "GRAPH_SYMBOL_NOT_FOUND"
    GRAPH_QUERY_ERROR = "GRAPH_QUERY_ERROR"

    # LSP
    LSP_NOT_AVAILABLE = "LSP_NOT_AVAILABLE"
    LSP_TIMEOUT = "LSP_TIMEOUT"
    LSP_SERVER_ERROR = "LSP_SERVER_ERROR"

    # Execution
    EXEC_DANGEROUS = "EXEC_DANGEROUS"
    EXEC_TIMEOUT = "EXEC_TIMEOUT"
    EXEC_NOT_FOUND = "EXEC_NOT_FOUND"
    EXEC_FAILED = "EXEC_FAILED"

    # Subagent
    SUBAGENT_UNKNOWN_ROLE = "SUBAGENT_UNKNOWN_ROLE"
    SUBAGENT_FAILED = "SUBAGENT_FAILED"

    # Parse / AST
    PARSE_UNSUPPORTED_LANG = "PARSE_UNSUPPORTED_LANG"
    PARSE_SYNTAX_ERROR = "PARSE_SYNTAX_ERROR"

    # Generic
    INTERNAL = "INTERNAL"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class ToolResponse(BaseModel):
    """Uniform envelope for all tool responses.

    Weak models can branch on `success` and `error_code` without
    parsing free-text error messages.
    """
    success: bool = True
    error_code: ErrorCode = ErrorCode.NONE
    error_message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(cls, **data: Any) -> "ToolResponse":
        """Convenience constructor for successful responses."""
        return cls(success=True, data=data)

    @classmethod
    def error(cls, code: ErrorCode, message: str, **extra: Any) -> "ToolResponse":
        """Convenience constructor for error responses."""
        return cls(
            success=False,
            error_code=code,
            error_message=message,
            data=extra,
        )
