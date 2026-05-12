# Codeforge MCP Server

**Persistent 5-layer knowledge graph for AI agents navigating large codebases.**

Built for CachyOS / Arch Linux. Provides a persistent memory backbone,
parallel deterministic subagents, and tools that return facts — not guesses.
The MCP server itself does **not** call any external LLM; reasoning is left
to the calling MCP client's model, which composes the building blocks below.

## Architecture

```
┌─────────────────────────────────────────────┐
│  MCP Client (Claude, Codebuff, etc.)        │
├─────────────────────────────────────────────┤
│  Atlas:// Resources  │  20+ MCP Tools       │
├─────────────────────────────────────────────┤
│  Navigation │ Understanding │ Execution │ Memory │
├─────────────────────────────────────────────┤
│  LSP Multiplexer  │  AST Indexer (tree-sitter)  │
├─────────────────────────────────────────────┤
│  Knowledge Graph (SQLite + BM25)             │
└─────────────────────────────────────────────┘
```

### Why it works

- **No hallucinated search**: every tool returns line numbers + hashes, not summaries
- **Stateful sessions**: the graph persists between turns, so the agent never re-discovers the repo
- **Parallel subagents**: the main agent delegates; it doesn't do everything serially
- **LSP + AST hybrid**: tree-sitter gives structure in 20ms, LSP gives types and references

## Installation

### 1. System dependencies

```bash
sudo pacman -Syu --needed \
  python uv ripgrep fd bat tree-sitter tree-sitter-cli \
  base-devel clang llvm nodejs npm go rust \
  pyright typescript-language-server rust-analyzer clang gopls \
  bubblewrap sqlite
```

### 2. Install the package

```bash
cd /path/to/codeforge-mcp
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 3. Verify installation

```bash
python -m codeforge_mcp.server --help
```

## Usage

### Direct CLI

```bash
cd /path/to/your/project
python -m codeforge_mcp.server  # Auto-detects project root from CWD
```

### With an MCP client (e.g., Gemini CLI, Claude Desktop)

#### Gemini CLI
Add to your `.gemini/settings.json` (project-specific) or `~/.gemini/settings.json` (global):

```json
{
  "mcpServers": {
    "codeforge": {
      "command": "/path/to/mcp_server/.venv/bin/python",
      "args": ["-m", "codeforge_mcp.server"],
      "env": {
        "CODEFORGE_SANDBOX": "0"
      }
    }
  }
}
```

#### Claude Desktop
Add to your MCP client config:

```json
{
  "mcpServers": {
    "codeforge": {
      "command": "/path/to/mcp_server/.venv/bin/python",
      "args": ["-m", "codeforge_mcp.server", "--project", "/path/to/project"],
      "env": {
        "CODEFORGE_SANDBOX": "0"
      }
    }
  }
}
```

The server uses no API keys and does not contact any external LLM.
The MCP client's own model handles all reasoning and tool composition.

### Sandboxed execution

Set `CODEFORGE_SANDBOX=1` to sandbox all `bash_run` commands via bubblewrap:

```bash
export CODEFORGE_SANDBOX=1
```

## Tools

### Navigation
| Tool | Description |
|------|-------------|
| `code_find_files(pattern)` | Find files by name with fd/glob |
| `code_search(query)` | Search code with ripgrep + BM25 re-rank |
| `symbol_lookup(name)` | LSP workspace/symbol across all servers |
| `lsp_workspace_symbols(query)` | Full workspace symbol search |
| `read_file(path)` | Read file with optional line range |
| `list_directory(path)` | List files and directories |
| `git_diff(base, head)` | Git diff with separate staged/unstaged |

### Understanding
| Tool | Description |
|------|-------------|
| `ast_query(file, xpath)` | Tree-sitter query (keywords + S-expressions) |
| `call_graph(function)` | Traverse call graph (upstream/downstream) |
| `impact_analysis(target)` | Blast radius: callers, tests, risk score |
| `ast_dependency_graph()` | Module-level import dependency graph |
| `lsp_goto_definition(file, line)` | Go to definition via LSP |
| `lsp_find_references(file, line)` | Find all references via LSP |
| `lsp_hover(file, line)` | Type info and docs via LSP |
| `lsp_diagnostics(file)` | Errors/warnings via publishDiagnostics |

### Execution
| Tool | Description |
|------|-------------|
| `bash_run(cmd)` | Run shell commands (bwrap sandbox optional) |
| `test_run(selector)` | Auto-detect pytest/vitest/jest/cargo test |
| `test_run_affected(target)` | Run only tests inferred to depend on a symbol |

### Memory
| Tool | Description |
|------|-------------|
| `decision_record(title, why)` | Write to .codeforge/decisions.md |
| `brief()` | Codebase summary: files, symbols, knowledge score |
| `context_budget()` | Current subagent token budget usage |

### Subagents
| Tool | Description |
|------|-------------|
| `spawn_subagent(role, task)` | Roles: file_finder, code_searcher, reviewer, test_impact, decompose |
| `spawn_subagents(specs)` | Run multiple subagents in parallel |

### Safe Editing
| Tool | Description |
|------|-------------|
| `patch_file_tool(path, ...)` | Apply a line-range patch with hash verification |
| `safe_edit_tool(path, ...)` | Patch plus reindex/diagnostic/test validation |

### File Watcher
| Tool | Description |
|------|-------------|
| `start_watcher()` | Start inotify-based auto-reindexing |
| `stop_watcher()` | Stop the file watcher |
| `watcher_status()` | Check watcher state and stats |

## Atlas:// Resources

Streamable snapshots available without tool calls:

| URI | Description |
|-----|-------------|
| `atlas://workspace/structure` | Full file tree |
| `atlas://symbols/{language}` | All symbols for a language (python, typescript, rust, etc.) |
| `atlas://dependencies` | Full module dependency graph |
| `atlas://decisions` | Recent design decisions |
| `atlas://brief` | Codebase summary with language breakdown |

## Example Session

```
Agent: "Fix rate limiting in auth"

Server does:
  1. code_search("rate limit", context=2)    → returns matches including TODO in login.rs line 47
  2. symbol_lookup("verify_token")           → LSP returns definition + 12 references
  3. impact_analysis("verify_token")         → HIGH risk, 3 modules affected
  4. spawn_subagent("test_impact", ...)      → finds 4 tests to run
  5. Agent edits
  6. spawn_subagent("reviewer", ...)         → runs diagnostics, returns checklist

Total tokens used by main agent: ~800
```

## Project Structure

```
codeforge_mcp/
├── server.py              # FastMCP server, 20+ tools, 5 resources, CLI
├── indexer.py             # File discovery (pathspec .gitignore), full/incremental indexing
├── logging.py             # Structured JSON logging to stderr
├── watcher.py             # File watcher (watchfiles/inotify) for cache invalidation
├── graph/
│   └── store.py           # SQLite knowledge graph, BM25 search, call graph traversal
├── ast/
│   └── indexer.py         # Tree-sitter parsing, XXH3 checksums, S-expression queries
├── lsp/
│   └── multiplexer.py     # Lazy-start LSP servers, single-reader dispatch, diagnostics
├── tools/
│   ├── navigation.py      # code_find_files, code_search, symbol_lookup
│   ├── understanding.py   # ast_query, call_graph, impact_analysis
│   ├── execution.py       # bash_run (bwrap sandbox), test_run
│   ├── memory.py          # decision_record, brief
│   ├── file_ops.py        # read_file, write_file, list_directory, git_diff
│   └── dependency.py      # AST dependency graph, import extraction
└── subagents/
    └── orchestrator.py    # spawn_subagent, spawn_subagents, atlas_delegate_task, ReAct loop
```

## Why CachyOS / Arch

CachyOS provides native packages — no containers needed:

- **btrfs snapshots** let you rollback after `bash_run`
- The **kernel scheduler** handles many LSP processes efficiently
- **pacman** keeps tree-sitter grammars up to date
- **bubblewrap** in the repos enables secure sandboxing

## License

MIT
