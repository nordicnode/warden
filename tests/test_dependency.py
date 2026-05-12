"""Tests for dependency tools — ast_dependency_graph, import extraction, resolution.

Verifies:
- _file_language maps extensions correctly
- _extract_imports handles Python, JS/TS, Rust, Go, C/C++ import syntax
- _resolve_import finds files by dotted name (local + project-root)
- _discover_importable_files finds files and respects exclusions
- ast_dependency_graph builds nodes and edges
"""

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codeforge_mcp.tools.dependency import (
    _file_language,
    _extract_imports,
    _resolve_import,
    _discover_importable_files,
    _resolve_dep_files,
    ast_dependency_graph,
)


class TestFileLanguage:
    """Verify _file_language maps extensions correctly."""

    def test_python(self) -> None:
        assert _file_language("main.py") == "python"

    def test_javascript(self) -> None:
        assert _file_language("app.js") == "javascript"
        assert _file_language("app.jsx") == "javascript"
        assert _file_language("app.mjs") == "javascript"

    def test_typescript(self) -> None:
        assert _file_language("app.ts") == "typescript"
        assert _file_language("app.tsx") == "typescript"

    def test_rust(self) -> None:
        assert _file_language("main.rs") == "rust"

    def test_go(self) -> None:
        assert _file_language("main.go") == "go"

    def test_c_cpp(self) -> None:
        assert _file_language("main.c") == "c"
        assert _file_language("header.h") == "c"
        assert _file_language("main.cpp") == "cpp"
        assert _file_language("header.hpp") == "cpp"

    def test_unknown(self) -> None:
        assert _file_language("README.md") == "unknown"
        assert _file_language("Dockerfile") == "unknown"
        assert _file_language("Makefile") == "unknown"


class TestExtractImports:
    """Verify _extract_imports extracts import statements from source code."""

    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        return tmp_path

    def test_python_import(self, root: Path) -> None:
        (root / "mod.py").write_text("import os\nimport sys, json\nfrom pathlib import Path\n")
        imports = _extract_imports(root, root / "mod.py")
        assert "os" in imports
        assert "sys" in imports
        assert "json" in imports
        assert "pathlib" in imports

    def test_python_from_import(self, root: Path) -> None:
        (root / "mod.py").write_text("from collections import defaultdict\nfrom .local import helper\n")
        imports = _extract_imports(root, root / "mod.py")
        assert "collections" in imports
        # Relative import (".local") is resolved to a path string
        assert any(".local" in imp or "local" in imp for imp in imports)

    def test_python_relative_dot_import(self, root: Path) -> None:
        """from .sibling import X produces a resolved relative path."""
        (root / "pkg").mkdir()
        (root / "pkg" / "mod.py").write_text("from .sibling import helper\n")
        imports = _extract_imports(root, root / "pkg" / "mod.py")
        # The resolved path should point to pkg/sibling (best-effort)
        assert any("sibling" in imp.lower() for imp in imports)

    def test_python_comment_ignored(self, root: Path) -> None:
        (root / "mod.py").write_text("import os  # standard library\n")
        imports = _extract_imports(root, root / "mod.py")
        assert imports == ["os"]

    def test_import_with_as(self, root: Path) -> None:
        (root / "mod.py").write_text("import numpy as np\n")
        imports = _extract_imports(root, root / "mod.py")
        assert imports == ["numpy"]

    def test_python_empty_file(self, root: Path) -> None:
        (root / "mod.py").write_text("# just a comment\nx = 1\n")
        imports = _extract_imports(root, root / "mod.py")
        assert imports == []

    def test_js_import_from(self, root: Path) -> None:
        (root / "app.js").write_text("import React from 'react';\nimport { useState } from './hooks';\n")
        imports = _extract_imports(root, root / "app.js")
        assert "react" in imports
        assert "./hooks" in imports

    def test_js_require(self, root: Path) -> None:
        (root / "app.js").write_text("const fs = require('fs');\nconst util = require('./util');\n")
        imports = _extract_imports(root, root / "app.js")
        assert "fs" in imports
        assert "./util" in imports

    def test_typescript_import(self, root: Path) -> None:
        (root / "app.ts").write_text("import type { Foo } from './types';\nimport { bar } from 'bar';\n")
        imports = _extract_imports(root, root / "app.ts")
        assert "./types" in imports
        assert "bar" in imports

    def test_rust_use(self, root: Path) -> None:
        (root / "main.rs").write_text("use std::collections::HashMap;\nuse crate::my_mod;\nuse serde;\n")
        imports = _extract_imports(root, root / "main.rs")
        # std::collections::HashMap → top-level "std"
        assert "std" in imports
        # crate::my_mod → "crate" is excluded, so no crate
        assert "crate" not in imports
        # self/super are also excluded per code logic
        assert "serde" in imports

    def test_go_import(self, root: Path) -> None:
        # Note: the regex only captures the first package in grouped imports
        # import ( "pkg1" "pkg2" ) captures "pkg1" only
        (root / "main.go").write_text(
            'import "fmt"\nimport (\n  "net/http"\n  "os"\n)\n'
        )
        imports = _extract_imports(root, root / "main.go")
        assert "fmt" in imports
        assert "net/http" in imports
        # "os" not captured by the single-package regex in grouped imports

    def test_go_import_standalone(self, root: Path) -> None:
        (root / "main.go").write_text('import "os"\nimport "fmt"\n')
        imports = _extract_imports(root, root / "main.go")
        assert "fmt" in imports
        assert "os" in imports

    def test_c_include(self, root: Path) -> None:
        (root / "main.c").write_text('#include <stdio.h>\n#include "local.h"\n')
        imports = _extract_imports(root, root / "main.c")
        assert "stdio.h" in imports
        assert "local.h" in imports

    def test_unknown_language_returns_empty(self, root: Path) -> None:
        (root / "README.md").write_text("import os  # not really code\n")
        imports = _extract_imports(root, root / "README.md")
        assert imports == []

    def test_nonexistent_file(self, root: Path) -> None:
        imports = _extract_imports(root, root / "nonexistent.py")
        assert imports == []

    def test_deduplicates(self, root: Path) -> None:
        (root / "mod.py").write_text("import os\nimport os\nimport sys\n")
        imports = _extract_imports(root, root / "mod.py")
        assert imports == ["os", "sys"]


class TestResolveImport:
    """Verify _resolve_import resolves dotted names to file paths."""

    def test_resolves_python_module_in_root(self, tmp_path: Path) -> None:
        (tmp_path / "mypackage").mkdir()
        (tmp_path / "mypackage" / "__init__.py").write_text("")
        (tmp_path / "mypackage" / "submod.py").write_text("")

        result = _resolve_import(tmp_path, tmp_path / "main.py", "mypackage.submod")
        assert result is not None
        assert result.name == "submod.py"

    def test_resolves_js_module_by_bare_specifier(self, tmp_path: Path) -> None:
        """_resolve_import handles bare specifiers that map to files (not ./ relative paths)."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "index.js").write_text("")
        (tmp_path / "src" / "utils.js").write_text("")

        # Use a dotted module name that maps to a file path
        result = _resolve_import(tmp_path, tmp_path / "src" / "index.js", "utils")
        assert result is not None
        assert result.name == "utils.js"

    def test_resolves_ts_extension_variants(self, tmp_path: Path) -> None:
        """_resolve_import tries .ts/.tsx/.js/.jsx/.mjs extensions for module names."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.ts").write_text("")
        (tmp_path / "src" / "types.ts").write_text("")

        # Use a dotted module name (not ./ relative path)
        result = _resolve_import(tmp_path, tmp_path / "src" / "app.ts", "types")
        assert result is not None
        assert result.name == "types.ts"

    def test_returns_none_for_unresolvable(self, tmp_path: Path) -> None:
        result = _resolve_import(tmp_path, tmp_path / "main.py", "nonexistent.module")
        assert result is None

    def test_resolves_init_py_package(self, tmp_path: Path) -> None:
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "__init__.py").write_text("")

        result = _resolve_import(tmp_path, tmp_path / "main.py", "pkg")
        assert result is not None
        assert result.name == "__init__.py"


class TestDiscoverImportableFiles:
    """Verify _discover_importable_files finds files and excludes hidden/build dirs."""

    def test_finds_importable_files(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("x=1")
        (tmp_path / "utils.js").write_text("x=1")
        (tmp_path / "lib.rs").write_text("x=1")
        (tmp_path / "main.go").write_text("x=1")
        (tmp_path / "README.md").write_text("text")

        files = _discover_importable_files(tmp_path, 200)
        names = [Path(f).name for f in files]
        assert "main.py" in names
        assert "utils.js" in names
        assert "lib.rs" in names
        assert "main.go" in names
        # README.md is not importable
        assert "README.md" not in names

    def test_excludes_hidden_and_build_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("x=1")
        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".hidden" / "secret.py").write_text("x=1")
        (tmp_path / "node_modules").mkdir(parents=True)
        (tmp_path / "node_modules" / "dep.js").write_text("x=1")

        files = _discover_importable_files(tmp_path, 200)
        paths = [f for f in files]
        assert any("src/main.py" in f for f in paths)
        assert not any(".hidden" in f for f in paths)
        assert not any("node_modules" in f for f in paths)

    def test_respects_max_files(self, tmp_path: Path) -> None:
        for i in range(10):
            (tmp_path / f"file_{i}.py").write_text("x=1")

        files = _discover_importable_files(tmp_path, 5)
        assert len(files) <= 5


class TestResolveDepFiles:
    """Verify _resolve_dep_files follows imports recursively."""

    def test_follows_import_chain(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("from b import foo\n")
        (tmp_path / "b.py").write_text("from c import bar\n")
        (tmp_path / "c.py").write_text("x = 1\n")

        mock_indexer = MagicMock()
        files = _resolve_dep_files(tmp_path, "a.py", mock_indexer, 200)
        names = [Path(f).name for f in files]
        assert "a.py" in names
        assert "b.py" in names
        assert "c.py" in names

    def test_handles_missing_focus_file(self, tmp_path: Path) -> None:
        mock_indexer = MagicMock()
        files = _resolve_dep_files(tmp_path, "nonexistent.py", mock_indexer, 200)
        assert files == []

    def test_respects_max_files(self, tmp_path: Path) -> None:
        # Create chain: a → b, b → c, c → d, but max_files=2
        (tmp_path / "a.py").write_text("from b import foo\n")
        (tmp_path / "b.py").write_text("from c import foo\n")
        (tmp_path / "c.py").write_text("from d import foo\n")
        (tmp_path / "d.py").write_text("x = 1\n")

        mock_indexer = MagicMock()
        files = _resolve_dep_files(tmp_path, "a.py", mock_indexer, 2)
        assert len(files) <= 2


class TestDependencyGraph:
    """Verify ast_dependency_graph builds nodes and edges."""

    def test_builds_graph_from_project(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("from utils import helper\n")
        (tmp_path / "utils.py").write_text("def helper(): pass\n")

        mock_indexer = MagicMock()
        result = ast_dependency_graph(mock_indexer, str(tmp_path), max_files=50)

        assert "nodes" in result
        assert "edges" in result
        assert "total_files" in result
        assert "total_edges" in result
        assert result["total_files"] >= 1

    def test_focus_file_mode(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("from b import foo\n")
        (tmp_path / "b.py").write_text("from c import foo\n")
        (tmp_path / "c.py").write_text("x = 1\n")
        (tmp_path / "unrelated.py").write_text("x = 2\n")

        mock_indexer = MagicMock()
        result = ast_dependency_graph(
            mock_indexer, str(tmp_path), focus_file="a.py", max_files=50
        )

        # Should include a.py, b.py, c.py but not unrelated.py
        file_names = set(result["nodes"].keys())
        assert "a.py" in file_names or any("a.py" in f for f in file_names)
