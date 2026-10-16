# Astronaut AI

> Local-first agentic CLI with native tool calling, MCP support, RAG, subagents, and a self-verifying loop.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/ollama-compatible-cyan)](https://ollama.com)
[![MCP](https://img.shields.io/badge/MCP-supported-purple)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/license-MIT-green)]()

Astronaut AI is an autonomous agent that lives on your machine and actually **does** things —
reads and edits files, runs commands, searches the web, manages Git, calls MCP servers,
and verifies its own work before showing you the answer.

It doesn't depend on cloud APIs. By default it talks to a local Ollama model.
Point it at OpenAI, OpenRouter, Groq, or Together and it works the same way.

---

## Features

### Core
- **Native tool calling** through Ollama's `tools=` API — no prompt hacks for supported models
- **Leaked tool-call parser** — catches `<tool_call>` and JSON blocks when a model drops them in plain text
- **Self-verifier loop** — a second model call checks the answer before it reaches you. Push-back on incomplete tasks.
- **Duplicate guard** — stops the model from calling the same tool with the same arguments twice
- **Fake-code detector** — flags when the model writes `import context7` instead of calling a function
- **Loop breaker** — kills tool-call storms

### Tools (30+)
- **Files**: `list_directory`, `open_file`, `create_file`, `delete_file`, `rename_file`, `append_to_file`, `file_info` — all with **unified diffs** on writes
- **Web**: `ddg_search` (JSON results), `fetch_url` (readability-extracted text), `get_weather`, `get_time`
- **Git**: `git_status`, `git_log`, `git_diff`, `git_add`, `git_commit`, `git_push`, `git_pull`, `git_checkout`, `git_branch_new`
- **System**: `run_command` (whitelist), `execute_python` (AST-sandboxed)
- **Math**: `calculate` (safe AST eval, no `eval()`)
- **RAG**: `rag_index`, `search_workspace` (local file index)
- **Agents**: `run_subagent`, `run_parallel_subagents`, `delegate_file_task`
- **Skills**: `list_skills_tool`, `load_skill`

### MCP integration
- **Auto-start** — MCP servers spawn as child processes at launch, no manual `npx` calls
- **Parallel warm-up** — all servers connect simultaneously, not sequentially
- **Health monitor** — background ping every 30s; dead servers are removed from the tool map
- **Hot reload** — `/mcphealth` shows status, toggle servers on/off without restarting
- Tested with: **Context7**, **Firecrawl**, **Tavily**, **sequential-thinking**

### Modes

| Mode | Description |
|------|-------------|
| **Standard** | All tools, full autonomy |
| **Code** | Programming focus, `create_file` + Git heavy |
| **Minimal** | Only `run_command`, `list_directory`, `open_file`, `create_file`, `file_info` |
| **Creator** | Standard + experimental tools |

Switch at any time with `/mode <name>`.

### Interfaces
- **CLI** — Rich-based TUI with Markdown streaming, colored diffs, spinner for thinking/verifying
- **Web UI** — Flask + Tailwind, multi-chat sidebar, live SSE streaming, inline confirmation widgets
- **GUI** — native window via `pywebview` (reuses the web UI, ~5 MB instead of Electron's 150 MB)

### Persistence
- **Append-only JSONL history** — every session is a `sessions/<id>.jsonl` file
- **AI-generated chat titles** — after the first message, a short title is generated and cached
- **Settings persistence** — `.astronaut_settings.json` (temperature, max steps, disabled tools, verifier config)
- **Provider persistence** — `.astronaut_providers.json` (API keys, base URLs, active model per provider)
- **MCP persistence** — `.astronaut_mcp.json` (server commands, env, enabled flag)

---

## Architecture

```
astronaut/
├── main.py                 # entry point, arg parsing, plugin loader
├── cli.py                  # Rich CLI: prompts, streaming, verifier, commands
├── web.py                  # Flask app + SSE streaming + confirmation queue
├── desktop.py              # pywebview wrapper for native window
├── config.py               # CONFIG, BASE_RULES, PROVIDER_PRESETS, LIBRARY_HINTS
├── providers.py            # OllamaProvider, OpenAIProvider, unified ChatResponse
├── providers_config.py     # load/save provider settings
├── history.py              # append-only JSONL, session management, truncation
├── settings.py             # global runtime settings
├── mcp_client.py           # MCP manager: parallel load, health monitor, tool wrappers
├── repomap.py              # workspace symbol map + import graph
├── rag.py                  # workspace file index (embeddings optional)
├── sandbox.py              # AST-validated subprocess Python execution
├── skills.py               # skills/*.md loader
├── planning.py             # plan-first execution
├── utils.py                # @tool / @requires_confirmation decorators
├── _shared.py              # safe_path, workspace root, context budget
├── plugins/
│   └── toolbox.py          # all native tools
├── skills/                 # *.md skill files (user-supplied)
├── templates/
│   └── index.html          # Web UI (Tailwind)
├── sessions/               # per-chat JSONL logs (runtime)
└── workspace/              # sandboxed file area (runtime)
```

### The agent loop

```
┌─────────────────────────────────────────────────────┐
│  user message                                       │
└──────────────────────┬──────────────────────────────┘
                       ▼
       ┌───────────────────────────────┐
       │  provider.chat(tools=[...])   │◄──────┐
       └───────────────┬───────────────┘       │
                       ▼                       │
             ┌──────────────────┐              │
             │  tool_calls?     │              │
             └────┬─────────┬───┘              │
                  │         │                  │
              yes │         │ no               │
                  ▼         ▼                  │
        ┌──────────────┐  ┌──────────────────┐ │
        │ execute tools│  │   VERIFIER       │ │
        │ append result│  │  (pass 1 + 2)    │ │
        └──────┬───────┘  └────────┬─────────┘ │
               │                   │           │
               │              OK   │  INCOMPLETE
               │                   │           │
               └───────────────────┼───────────┘
                                   │ push-back
                                   └───────────┘
                                   ▼
                         final answer to user
```

---

## Installation

### 1. Requirements

- **Python 3.10+**
- **Ollama** running locally: [ollama.com](https://ollama.com)
- **Node.js** (only if you want MCP servers via `npx`)

### 2. Clone

```bash
git clone https://github.com/DcorpProj/astronaut
cd astronaut
```

### 3. Install dependencies

**Core (CLI):**
```bash
pip install ollama rich requests beautifulsoup4 transliterate \
            prompt_toolkit pyyaml mcp
```

**Web UI:**
```bash
pip install flask flask-socketio pyjwt werkzeug
```

**Optional:**
```bash
pip install openai     # OpenAI / OpenRouter / Groq / Together
pip install pywebview  # native GUI window (default mode)
```

Or from a file (if you keep one):
```bash
pip install -r requirements.txt
```

### 4. Pull a model

```bash
ollama pull qwen2.5:7b        # default — fast, good tool calling
ollama pull qwen3:8b          # newer, supports thinking traces
ollama pull granite4.1:8b     # stricter instruction following
```

### 5. Run

```bash
python main.py                          # default: native GUI window
python main.py --cli                    # terminal UI
python main.py --web                    # browser UI on http://127.0.0.1:5000
```

If `pywebview` isn't installed, `python main.py` falls back to the CLI automatically.

---

## Quick start

### CLI

```bash
python main.py --cli                     # default model, Standard mode
python main.py --cli --model qwen3:8b    # different model
python main.py --cli --mode Code         # Code mode
python main.py --cli --workspace ./my-project
```

Inside the CLI:

```
❯ find 3 cake recipes and save to cakes.md

  ✻ Thinking…
  ⏺ ddg_search({"query": "cake recipes", "num_results": 3})
      (0.7s)
    ⎿ {"count": 3, "results": [...]}
  ⏺ create_file({"filename": "cakes.md", "content": "..."})
    ⎿ File cakes.md created (12 lines):
        + # Cake Recipes
        + 1. **Chocolate Cake** — https://...
  ✻ Verifying…
  ✓ verifier ok (2 passes)

> Готово. Нашёл 3 рецепта и сохранил в cakes.md.
```

### Web UI

```bash
python main.py --web
```

Opens `http://127.0.0.1:5000` in your browser. Multi-chat sidebar, live streaming, inline confirmation widgets, tool toggles, provider switcher, MCP settings.

### Native window (default)

```bash
python main.py
```

Same UI, wrapped in a real desktop window via `pywebview`. ~5 MB instead of 150 MB (Electron).

---

## Configuration files

### `.astronaut_settings.json`

```json
{
  "system_prompt": "",
  "temperature": 0.4,
  "max_steps": 15,
  "keep_alive": "30m",
  "history_max_messages": 40,
  "ai_titles": true,
  "disabled_tools": [],
  "verification_enabled": true,
  "max_verification_retries": 3,
  "min_verification_attempts": 2,
  "verify_trivial_skip": true,
  "thinking_mode": "auto",
  "planning_enabled": false
}
```

### `.astronaut_providers.json`

```json
{
  "_active": "ollama",
  "ollama": { "model": "qwen2.5:7b" },
  "openai": {
    "api_key": "sk-...",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini"
  }
}
```

### `.astronaut_mcp.json`

```json
{
  "servers": {
    "context7": {
      "command": "cmd",
      "args": ["/c", "npx", "-y", "@upstash/context7-mcp"],
      "env": {},
      "enabled": true
    },
    "sequential-thinking": {
      "command": "cmd",
      "args": ["/c", "npx", "-y",
               "@modelcontextprotocol/server-sequential-thinking"],
      "env": {},
      "enabled": false
    }
  }
}
```

> On Windows, MCP servers launch via `cmd /c npx …`.
> On macOS/Linux, replace `"command": "cmd"` with `"command": "npx"` and drop the `/c` argument.

---

## Commands

### Session
| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/status` | Current configuration |
| `/settings` | Runtime settings |
| `/sessions` | List all chats |
| `/history` | Show current chat history |
| `/switch <n>` | Switch to another chat |
| `/new` | Start a new chat |
| `/rename <title>` | Rename current chat |
| `/clear` | Clear current chat |
| `/undo` | Undo last user+assistant exchange |
| `/retry` | Regenerate last response |
| `/find <text>` | Search current chat |
| `/stats` | Session statistics |
| `/context` | Approximate context size |
| `/system` | Show effective system prompt |
| `/export <file.md>` | Export chat as Markdown |
| `/save <file.md>` | Alias for `/export` |
| `/yank <file>` | Save last reply to a file |

### Model & providers
| Command | Description |
|---------|-------------|
| `/model <name>` | Switch model |
| `/provider <name>` | `ollama`, `openai`, `openrouter`, `groq`, `together`, `custom` |
| `/think on\|off\|auto` | Toggle thinking traces (Qwen3, Nemotron, Gemma 4) |
| `/verify on\|off\|trivial\|passes N` | Control self-verification |
| `/verifier-model <name\|off>` | Set a dedicated verifier model |
| `/planning on\|off` | Toggle plan-first execution |

### Modes & tools
| Command | Description |
|---------|-------------|
| `/mode <mode>` | `Standard`, `Code`, `Minimal`, `Creator` |
| `/tools` | List active tools |
| `/enable <name>` | Enable a tool |
| `/disable <name>` | Disable a tool |
| `/scope <text>` | Preview which tools are allowed for a given prompt |

### Workspace & MCP
| Command | Description |
|---------|-------------|
| `/workspace <path>` | Change workspace |
| `/attach <path>` | Attach a file (or use `!- path -!` inline) |
| `/repomap` | Show repo map for the current workspace |
| `/repomap-gen` | Rebuild the repo map from disk |
| `/mcp` | List MCP servers |
| `/mcphealth` | MCP health dashboard |

### Other
| Command | Description |
|---------|-------------|
| `/debug` | Test `create_file` round-trip & print signature |
| `/kill` | Interrupt the current agent turn |
| `/exit` | Quit |
| `/quit` | Alias for `/exit` |

---

## Adding a plugin

Create any `.py` in `plugins/` with one or more functions decorated with `@tool`:

```python
# plugins/my_plugin.py
from utils import tool, requires_confirmation
from _shared import safe_path

@tool("Count lines in a file", category="file")
def count_lines(filename: str) -> str:
    """Count the number of lines in a text file."""
    target = safe_path(filename)
    with open(target, encoding="utf-8") as f:
        n = sum(1 for _ in f)
    return f"{filename}: {n} lines"

@tool("Delete a file", category="file")
@requires_confirmation
def safe_delete(filename: str) -> str:
    """Delete a file (asks for confirmation)."""
    ...
```

Restart the CLI — the tool is now in the inventory. `@requires_confirmation` triggers a `y/N` prompt before execution.

---

## Adding an MCP server

### Via the web UI

`⚙️ Settings → MCP Servers → + add`

### Via config file

Add an entry to `.astronaut_mcp.json`:

```json
{
  "servers": {
    "filesystem": {
      "command": "cmd",
      "args": ["/c", "npx", "-y",
               "@modelcontextprotocol/server-filesystem", "D:/myproject"],
      "env": {},
      "enabled": true
    }
  }
}
```

Restart. The tools appear in the inventory with an `[MCP]` badge.

---

## Model recommendations

### Local

| Size | Model | Notes |
|------|-------|-------|
| 7B | `qwen2.5:7b` | Best balance for tool calling |
| 7B | `qwen2.5-coder:7b` | Better for code, tool calling is decent |
| 8B | `qwen3:8b` | Thinking traces; good agentic behaviour |
| 8B | `granite4.1:8b` | Stricter instruction following, denser |
| 14B | `qwen2.5:14b` | **Big jump** in multi-step chains |
| 32B | `qwen2.5:32b` | Nearly no hallucinations (needs 24 GB VRAM) |

### Cloud (via Ollama Cloud)

| Model | Notes |
|-------|-------|
| `nemotron-3-nano:30b-cloud` | MoE, 3.5B active — efficient agentic model |
| `gemma4:31b-cloud` | Well-rounded, strong reasoning |
| `gpt-oss:120b-cloud` | Heavy; burns free credits fast |

**7B caveats**: forgets data mid-chain, invents tool arguments,
loops on meta-tools. Astronaut AI has guards against all of this, but
14B+ eliminates most of them at the source.

---

## Design notes

- **Append-only history** — never rewritten, only extended. Enables undo, audit, replay.
- **Leaked tool-call parser** — real-world models (Granite, DeepSeek-R1) drop `tool_calls` in text; we catch them.
- **Verifier with push-back** — the answer isn't shown until it passes verification (configurable).
- **Sandboxed Python** — AST validation blocks `os.system`, `subprocess`, `eval`, `exec`.
- **Safe paths** — every file operation is confined to `workspace/` unless the tool is explicitly system-level.
- **Parallel MCP warm-up** — N servers start at once instead of sequentially.

---

## Status

Active development. Not production-ready. Expect bugs, especially with
smaller models and non-OpenAI-compatible providers.

**Feedback, PRs, and issues welcome — I read every one.**

---

## Credits

Built on the shoulders of:

- [Ollama](https://ollama.com) — local model runtime
- [Rich](https://github.com/Textualize/rich) — terminal UI
- [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) — input handling
- [MCP](https://modelcontextprotocol.io/) — standard for tool servers
- [Flask](https://flask.palletsprojects.com/) + [Tailwind](https://tailwindcss.com/) — web UI
- [D3](https://d3js.org/) — repo map graph rendering
- [marked](https://marked.js.org/) — Markdown rendering in the web UI
- [highlight.js](https://highlightjs.org/) — syntax highlighting

Named after the people who actually do this for a living. Yes, this thing
mostly stays on the ground.

---

## License

MIT — see [LICENSE](LICENSE).