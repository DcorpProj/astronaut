# plugins/toolbox.py
import ast as _ast
import datetime
import difflib
import json
import operator
import os
import platform
import subprocess

from utils import requires_confirmation, tool
from _shared import (
    safe_path, ws, context_budget, truncate_for_model,
    get_context_window, invalidate_context_cache,
)


# ============================================================
# DIFF HELPER
# ============================================================

def _make_diff(old_text: str, new_text: str, filename: str,
               context: int = 2, max_lines: int = 40) -> str:
    old_text = (old_text or "").replace("\r\n", "\n")
    new_text = (new_text or "").replace("\r\n", "\n")

    if not old_text:
        new_lines = new_text.splitlines()
        shown = new_lines[:max_lines]
        out = "\n".join("+ " + line for line in shown)
        if len(new_lines) > max_lines:
            out += f"\n+ ... ({len(new_lines) - max_lines} more lines)"
        return out or "(empty file)"

    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        n=context,
    ))

    if not diff:
        return "(no changes)"

    if len(diff) > max_lines:
        diff = diff[:max_lines]
        diff.append(f"... (diff truncated at {max_lines} lines)\n")

    return "".join(
        line if line.endswith("\n") else line + "\n" for line in diff
    ).rstrip()


# ============================================================
# FILES
# ============================================================

@tool("List contents of a directory", category="file")
def list_directory(path: str = "") -> str:
    """List contents of a directory (relative to workspace)."""
    try:
        target = safe_path(path) if path else ws()
        if not target.exists():
            return f"Directory {path} not found"
        if not target.is_dir():
            return f"{path} is not a directory"

        items = sorted(target.iterdir())
        total = len(items)
        if total == 0:
            return "Directory is empty"

        HARD_CAP = 500
        truncated = total > HARD_CAP
        if truncated:
            items = items[:HARD_CAP]

        lines = []
        for i in items:
            prefix = "[D] " if i.is_dir() else "[F] "
            size = i.stat().st_size if i.is_file() else ""
            lines.append(f"{prefix}{i.name}{'  (' + str(size) + ' bytes)' if size else ''}")

        out = "\n".join(lines)
        if truncated:
            out += f"\n... (showed {HARD_CAP} of {total} entries)"
        return out
    except Exception as e:
        return f"Error: {e}"


@tool("Read a text file", category="file")
def open_file(filename: str) -> str:
    """Read and return contents of a text file. Truncated to the model's
    context budget so a single read doesn't blow the window."""
    try:
        t = safe_path(filename)
        if not t.exists():
            return f"File '{filename}' not found."
        if t.is_dir():
            return f"'{filename}' is a directory."
        size = t.stat().st_size

        if size > 20_000_000:
            return (f"File '{filename}' is {size} bytes — too large to read "
                    f"(20 MB cap). Use run_command with head/tail or grep to "
                    f"inspect specific parts.")

        with open(t, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        content = truncate_for_model(content, label="file truncated")
        return f"File '{filename}' ({size} bytes):\n{content}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool("Show file details", category="file")
def file_info(filename: str) -> str:
    """Show file details."""
    try:
        t = safe_path(filename)
        if not t.exists():
            return f"File '{filename}' not found."
        s = t.stat()
        mtime = datetime.datetime.fromtimestamp(s.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        return (f"Name: {filename}\nType: {'Directory' if t.is_dir() else 'File'}\n"
                f"Size: {s.st_size} bytes\nModified: {mtime}")
    except Exception as e:
        return f"Error: {e}"


@tool("Create or overwrite a file", category="file")
@requires_confirmation
def create_file(filename: str, content: str) -> str:
    """Create or overwrite a file. Returns a unified diff."""
    try:
        t = safe_path(filename)
        t.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, list):
            content = "\n".join(content)
        elif isinstance(content, str):
            content = content.replace("\\n", "\n").replace("\\t", "\t")

        existed = t.exists() and t.is_file()
        old_text = ""
        if existed:
            try:
                old_text = t.read_text(encoding="utf-8", errors="replace")
            except Exception:
                old_text = ""

        with open(t, "w", encoding="utf-8") as f:
            f.write(content)

        verb = "updated" if existed else "created"
        n_lines = len(content.splitlines())
        diff = _make_diff(old_text, content, filename)
        return f"File {filename} {verb} ({n_lines} lines):\n{diff}"
    except Exception as e:
        return f"Error: {e}"


@tool("Delete a file or empty directory", category="file")
@requires_confirmation
def delete_file(filename: str) -> str:
    """Delete a file or empty directory."""
    try:
        t = safe_path(filename)
        if not t.exists():
            return f"{filename} does not exist"
        if t.is_file():
            size = t.stat().st_size
            os.remove(t)
            return f"- {filename}  ({size} bytes, deleted)"
        if any(t.iterdir()):
            return f"Directory {filename} not empty – not deleted"
        t.rmdir()
        return f"- {filename}/  (empty dir, deleted)"
    except Exception as e:
        return f"Error: {e}"


@tool("Rename or move a file/folder", category="file")
@requires_confirmation
def rename_file(old_name: str, new_name: str) -> str:
    """Rename or move a file/folder."""
    try:
        o = safe_path(old_name)
        n = safe_path(new_name)
        if not o.exists():
            return f"Source '{old_name}' not found."
        if n.exists():
            return f"Target '{new_name}' already exists."
        n.parent.mkdir(parents=True, exist_ok=True)
        o.rename(n)
        return f"- {old_name}\n+ {new_name}"
    except Exception as e:
        return f"Error: {e}"


@tool("Append text to a file", category="file")
@requires_confirmation
def append_to_file(filename: str, content: str) -> str:
    """Append text to an existing file. Returns a diff of the change."""
    try:
        t = safe_path(filename)
        if not t.exists():
            return f"File '{filename}' not found."
        if t.is_dir():
            return f"'{filename}' is a directory."
        if isinstance(content, str):
            content = content.replace("\\n", "\n").replace("\\t", "\t")

        old_text = t.read_text(encoding="utf-8", errors="replace")
        new_text = old_text + content

        with open(t, "a", encoding="utf-8") as f:
            f.write(content)

        diff = _make_diff(old_text, new_text, filename)
        return f"Appended to {filename} ({len(content)} chars):\n{diff}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================
# GIT
# ============================================================

def _git(args):
    try:
        r = subprocess.run(["git"] + args, cwd=str(ws()), capture_output=True,
                           text=True, check=False, timeout=30)
        if r.returncode != 0:
            return f"Git error: {r.stderr.strip()}"
        return r.stdout.strip() or "(no output)"
    except Exception as e:
        return f"Git error: {e}"


@tool("Show git status", category="git")
def git_status() -> str:
    """Show git status."""
    return _git(["status", "--porcelain"])


@tool("Show current branch", category="git")
def git_branch() -> str:
    """Show current branch and list all branches."""
    cur = _git(["branch", "--show-current"])
    if cur.startswith("Git error"):
        return cur
    return f"Current: {cur}\n\nAll:\n{_git(['branch', '--list'])}"


@tool("Show git log", category="git")
def git_log(n: int = 5) -> str:
    """Show last n commits."""
    return _git(["log", f"-n {n}", "--oneline"])


@tool("Show diff", category="git")
def git_diff(staged: bool = False) -> str:
    """Show diff of working tree (or staged if staged=True)."""
    out = _git(["diff", "--cached"] if staged else ["diff"])
    return truncate_for_model(out, label="diff truncated")


@tool("List git remotes", category="git")
def git_remote() -> str:
    """List git remotes."""
    return _git(["remote", "-v"])


@tool("Stage files for commit", category="git")
@requires_confirmation
def git_add(files: str = ".") -> str:
    """Stage files for commit ('.' for all)."""
    return _git(["add", "-A"]) if files == "." else _git(["add", files])


@tool("Commit staged changes", category="git")
@requires_confirmation
def git_commit(message: str) -> str:
    """Commit staged changes."""
    return _git(["commit", "-m", message])


@tool("Push commits", category="git")
@requires_confirmation
def git_push(remote: str = "origin", branch: str = "") -> str:
    """Push commits to remote."""
    if not branch:
        branch = _git(["branch", "--show-current"]) or "HEAD"
        if branch.startswith("Git error"):
            return branch
    return _git(["push", remote, branch])


@tool("Pull changes", category="git")
@requires_confirmation
def git_pull(remote: str = "origin", branch: str = "") -> str:
    """Pull changes from remote."""
    if not branch:
        branch = _git(["branch", "--show-current"]) or "HEAD"
        if branch.startswith("Git error"):
            return branch
    return _git(["pull", remote, branch])


@tool("Checkout branch", category="git")
@requires_confirmation
def git_checkout(branch: str) -> str:
    """Switch to a branch."""
    return _git(["checkout", branch])


@tool("Create new branch", category="git")
@requires_confirmation
def git_branch_new(name: str) -> str:
    """Create and switch to a new branch."""
    return _git(["checkout", "-b", name])


# ============================================================
# WEB
# ============================================================

@tool("Search the web (DuckDuckGo HTML) - returns a JSON list of results", category="web")
def ddg_search(query: str, num_results: int = 5, region: str = "us-en") -> str:
    """Search the web via DuckDuckGo HTML endpoint.

    Returns JSON: {"query": "...", "count": N, "results": [
        {"title": "...", "url": "...", "snippet": "..."}, ...
    ]}
    """
    import requests
    from bs4 import BeautifulSoup

    if not query:
        return json.dumps({"error": "empty query"})

    num_results = max(3, min(int(num_results or 5), 15))

    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
    }
    params = {"q": query, "kl": region}

    resp = None
    last_err = None
    for attempt in range(2):
        try:
            resp = requests.post(url, data=params, headers=headers, timeout=120)
            resp.raise_for_status()
            break
        except Exception as e:
            last_err = str(e)
            resp = None
            if attempt == 0:
                import time as _t
                _t.sleep(1)

    if resp is None:
        return json.dumps({"error": f"search failed after 2 attempts: {last_err}", "query": query})

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for item in soup.select(".result"):
        title_el = item.select_one(".result__title a")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        link = title_el.get("href")
        if link and "duckduckgo.com/y.js" in link:
            continue
        if link and "uddg=" in link:
            link = link.split("uddg=")[1].split("&")[0]
            link = requests.utils.unquote(link)
        snippet_el = item.select_one(".result__snippet")
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        results.append({
            "title": title,
            "url": link or "",
            "snippet": snippet,
        })
        if len(results) >= num_results:
            break

    if not results:
        return json.dumps({"error": "no results found", "query": query, "count": 0, "results": []})

    return json.dumps({
        "query": query,
        "count": len(results),
        "results": results,
    }, ensure_ascii=False)


@tool("Fetch a URL and return its main text content", category="web")
def fetch_url(url: str, max_chars: int = 5000) -> str:
    """Fetch a URL and return its main readable text (truncated)."""
    import requests
    from bs4 import BeautifulSoup

    if not url:
        return "Error: empty URL"

    model_budget = context_budget()
    max_chars = max(500, min(int(max_chars or 5000), model_budget))

    try:
        r = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
        })
        r.raise_for_status()
    except Exception as e:
        return f"Fetch error: {e}"

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "noscript", "iframe", "svg", "form", "button"]):
        tag.decompose()

    main = (soup.find("main") or soup.find("article")
            or soup.find(attrs={"role": "main"})
            or soup.find(class_=lambda c: c and any(
                k in " ".join(c if isinstance(c, list) else [c]).lower()
                for k in ("content", "article", "post", "recipe", "main")))
            or soup.body
            or soup)

    text = main.get_text(separator="\n", strip=True)
    text = "\n".join(line for line in text.splitlines() if line.strip())

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... (truncated, total {len(text)} chars)"

    return text or "(no readable content)"


@tool("Get weather for a city", category="web")
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    from transliterate import translit
    import requests
    c = city.strip()
    if not all(ch.isascii() or ch.isspace() for ch in c):
        try:
            c = translit(c, "ru", reversed=True)
        except Exception:
            pass
    try:
        r = requests.get(f"https://wttr.in/{c}?format=j1", timeout=10)
        r.raise_for_status()
        data = r.json()
        cur = data.get("current_condition", [{}])[0]
        if not cur:
            return f"No weather for {city}"
        return (f"Temperature {cur.get('temp_C', '?')}C, humidity {cur.get('humidity', '?')}%, "
                f"wind {cur.get('wind_speed', '?')} km/h, "
                f"{cur.get('weatherDesc', [{'value': 'unknown'}])[0]['value']}")
    except Exception as e:
        return f"Weather error: {e}"


@tool("Get current local time", category="web")
def get_time(city: str = "local") -> str:
    """Get current local time."""
    return f"Time in {city}: {datetime.datetime.now().strftime('%H:%M:%S')}"


# ============================================================
# MATH
# ============================================================

_OPS = {_ast.Add: operator.add, _ast.Sub: operator.sub, _ast.Mult: operator.mul,
        _ast.Div: operator.truediv, _ast.Mod: operator.mod, _ast.Pow: operator.pow,
        _ast.USub: operator.neg, _ast.UAdd: operator.pos, _ast.FloorDiv: operator.floordiv}


def _seval(node):
    if isinstance(node, _ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, _ast.BinOp):
        return _OPS[type(node.op)](_seval(node.left), _seval(node.right))
    if isinstance(node, _ast.UnaryOp):
        return _OPS[type(node.op)](_seval(node.operand))
    raise ValueError(f"Unsupported: {type(node).__name__}")


@tool("Evaluate a math expression safely", category="math")
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        tree = _ast.parse(expression, mode="eval")
        return str(_seval(tree.body))
    except Exception as e:
        return f"Calculation error: {e}"


# ============================================================
# SYSTEM
# ============================================================

_WIN = ("dir", "cd", "type", "echo", "whoami", "date", "hostname", "where",
        "cargo", "rustc", "rustup", "python", "pip", "git", "npm", "node", "npx", "yarn",
        "java", "javac", "mvn", "gradle", "go", "dotnet", "cl", "clang", "gcc", "g++", "make", "cmake")
_UNIX = ("ls", "pwd", "cat", "echo", "whoami", "date", "uname", "which",
         "cargo", "rustc", "rustup", "python", "python3", "pip", "pip3", "git", "npm", "node", "npx", "yarn",
         "java", "javac", "mvn", "gradle", "go", "dotnet", "gcc", "g++", "clang", "make", "cmake")


@tool("Execute a shell command (dev tools only)", category="system")
@requires_confirmation
def run_command(command: str) -> str:
    """Execute a shell command."""
    prefixes = _WIN if platform.system() == "Windows" else _UNIX
    c = command.strip()
    if not any(c.startswith(p) for p in prefixes):
        return f"Not allowed. Prefixes: {', '.join(prefixes)}"
    if any(ch in command for ch in "|&;><$`\n\r"):
        return "Forbidden shell control characters (including newlines)"
    try:
        r = subprocess.run(command, shell=True, text=True, timeout=60,
                           cwd=str(ws()), capture_output=True, check=False)
        out = ((r.stdout or "") + (r.stderr or "")).strip() or "(no output)"
        out = truncate_for_model(out, label="output truncated")
        return f"Exit {r.returncode}:\n{out}" if r.returncode != 0 else out
    except subprocess.TimeoutExpired:
        return "Timeout (60s)"
    except Exception as e:
        return f"Error: {e}"


@tool("Execute Python code in a sandbox", category="system")
@requires_confirmation
def execute_python(code: str) -> str:
    """Execute Python code in a sandboxed subprocess."""
    from sandbox import execute_python as _ex
    return _ex(code, ws())


# ============================================================
# RAG + SKILLS
# ============================================================

@tool("Index workspace files for RAG", category="rag")
@requires_confirmation
def rag_index(force: bool = False) -> str:
    """Build or refresh the RAG index for the current workspace."""
    from rag import index_workspace, index_stats
    res = index_workspace(ws(), force=force)
    st = index_stats(ws())
    return f"Indexed {res['files_indexed']} files, {res['chunks']} chunks. Total: {st['files']} files, {st['chunks']} chunks."


@tool("Search indexed workspace files (code, docs) - NOT web search", category="rag")
def search_workspace(question: str, top_k: int = 5) -> str:
    """Search the LOCAL workspace file index. For web search use ddg_search."""
    from rag import query as _q
    results = _q(ws(), question, top_k=top_k)
    if not results:
        return "No relevant chunks found in workspace. Try rag_index first, or use ddg_search for web."
    lines = []
    for r in results:
        snippet = r["text"][:400] + ("..." if len(r["text"]) > 400 else "")
        lines.append(f"[{r['score']}] {r['file']} (chunk {r['chunk']}):\n{snippet}")
    return "\n\n".join(lines)


@tool("Show RAG index stats", category="rag")
def rag_stats() -> str:
    """Show how many files and chunks are in the RAG index."""
    from rag import index_stats
    st = index_stats(ws())
    return f"RAG index: {st['files']} files, {st['chunks']} chunks."


@tool("List available skills", category="agent")
def list_skills_tool() -> str:
    """List all available skills the agent can load."""
    from skills import list_skills
    skills = list_skills()
    if not skills:
        return "No skills found."
    return "\n".join(f"- {s['name']}: {s['title']} - {s['description']}" for s in skills)


@tool("Load a skill's full instructions", category="agent")
def load_skill(name: str) -> str:
    """Load the full text of a skill by name. Use list_skills to see options."""
    from skills import read_skill
    text = read_skill(name)
    if text is None:
        return f"Skill '{name}' not found."
    return text


# ============================================================
# AGENT
# ============================================================

def _build_subagent_schema(tools: dict) -> list:
    import inspect
    tmap = {int: "integer", float: "number", str: "string",
            bool: "boolean", list: "array", dict: "object"}
    schema = []
    for name, fn in tools.items():
        mcp_schema = getattr(fn, "_mcp_schema", None)
        if mcp_schema:
            schema.append(mcp_schema)
            continue
        sig = inspect.signature(fn)
        props, required = {}, []
        for pname, p in sig.parameters.items():
            ann = p.annotation
            props[pname] = {"type": tmap.get(ann, "string") if ann is not inspect.Parameter.empty else "string"}
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        meta = getattr(fn, "_tool_meta", {})
        schema.append({
            "type": "function",
            "function": {
                "name": name,
                "description": meta.get("description", fn.__doc__ or ""),
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })
    return schema


def _subagent_toolset(include_confirm_required: bool = True) -> dict:
    blocked = {"run_subagent", "run_parallel_subagents", "create_plan", "delegate_file_task"}
    out = {}
    for fn in ALL_TOOLS:
        if fn.__name__ in blocked:
            continue
        if not include_confirm_required and getattr(fn, "_requires_confirmation", False):
            continue
        out[fn.__name__] = fn
    return out


def _resolve_subagent_provider():
    from providers import get_provider
    from providers_config import get_active, get_provider_settings
    from config import CONFIG, PROVIDER_PRESETS

    active = get_active()
    preset = PROVIDER_PRESETS.get(active, {})
    settings = get_provider_settings(active)

    if active == "ollama":
        cfg = {
            "provider": "ollama",
            "keep_alive": CONFIG.get("keep_alive", "30m"),
            "temperature": 0.4,
        }
        model = settings.get("model") or CONFIG.get("model") or "qwen2.5:7b"
    else:
        cfg = {
            "provider": "openai",
            "api_key": settings.get("api_key") or CONFIG.get("api_key"),
            "base_url": settings.get("base_url") or preset.get("default_base"),
            "temperature": 0.4,
        }
        model = settings.get("model") or CONFIG.get("model") or "gpt-4o-mini"

    return get_provider(cfg), model


def _fallback_from_tool_results(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "tool":
            content = (m.get("content") or "").strip()
            if content:
                return truncate_for_model(content, label="sub-agent fallback")
    return ""


def _run_subagent_core(instruction: str, max_steps: int = 8,
                       tools: dict = None, quiet: bool = True) -> str:
    if tools is None:
        tools = _subagent_toolset(include_confirm_required=True)

    schema = _build_subagent_schema(tools)

    try:
        provider, model = _resolve_subagent_provider()
    except Exception as e:
        return f"(sub-agent init failed: {e})"

    messages = [
        {"role": "system", "content": (
            "You are a focused autonomous sub-agent. "
            "Complete the given task using the available tools. "
            "Do NOT ask for confirmation — just execute tools directly. "
            "Work step by step; if a tool fails, try another approach. "
            "When done, reply with ONLY the final result. No meta-commentary, "
            "no 'I will now...', no explanations."
        )},
        {"role": "user", "content": instruction},
    ]

    recent_calls = []
    final = None

    for _ in range(max(1, int(max_steps))):
        try:
            resp = provider.chat(messages, tools=schema, model=model)
        except Exception as e:
            return f"(sub-agent call failed: {e})"

        if not resp.tool_calls:
            final = (resp.content or "").strip() or None
            break

        messages.append({
            "role": "assistant",
            "content": resp.content or "",
            "tool_calls": [
                {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in resp.tool_calls
            ],
        })

        for tc in resp.tool_calls:
            try:
                key = (tc["name"], json.dumps(tc["arguments"], sort_keys=True, default=str))
            except Exception:
                key = (tc["name"], str(tc["arguments"]))

            if key in recent_calls:
                result = "[HARNESS] Duplicate call blocked. The tool was NOT executed."
            else:
                recent_calls.append(key)
                fn = tools.get(tc["name"])
                if not fn:
                    result = f"Unknown tool: {tc['name']}"
                else:
                    try:
                        result = fn(**tc["arguments"])
                    except Exception as e:
                        result = f"Tool error: {e}"

            messages.append({"role": "tool", "tool_name": tc["name"], "content": str(result)})

    if final is None:
        try:
            r = provider.chat(messages, tools=None, model=model)
            content = (r.content or "").strip()
            if content:
                final = content
            else:
                fallback = _fallback_from_tool_results(messages)
                if fallback:
                    final = (f"(sub-agent returned no summary; last tool "
                             f"result follows)\n\n{fallback}")
                else:
                    final = "(sub-agent produced no answer)"
        except Exception as e:
            fallback = _fallback_from_tool_results(messages)
            if fallback:
                final = (f"(sub-agent finalize failed: {e}; last tool "
                         f"result follows)\n\n{fallback}")
            else:
                final = f"(sub-agent failed to finalize: {e})"

    return final or "(empty)"


def _run_file_agent(task: str, max_steps: int = 5) -> str:
    allowed = {
        "list_directory", "open_file", "file_info",
        "create_file", "append_to_file", "delete_file", "rename_file",
        "run_command",
    }
    tools = {}
    for fn in ALL_TOOLS:
        if fn.__name__ in allowed:
            tools[fn.__name__] = fn

    schema = _build_subagent_schema(tools)

    try:
        provider, model = _resolve_subagent_provider()
    except Exception as e:
        return f"(file-agent init failed: {e})"

    messages = [
        {"role": "system", "content": (
            "You are a file-system specialist agent. "
            "Complete the given file or shell task using ONLY the provided tools. "
            "Work step by step. Execute tools directly. "
            "Never perform destructive operations (delete, overwrite) unless "
            "the task EXPLICITLY asks for them. "
            "When done, reply with a ONE-LINE summary of what you did."
        )},
        {"role": "user", "content": task},
    ]

    recent_calls = []
    final = None

    for _ in range(max(1, int(max_steps))):
        try:
            resp = provider.chat(messages, tools=schema, model=model)
        except Exception as e:
            return f"(file-agent call failed: {e})"

        if not resp.tool_calls:
            final = (resp.content or "").strip() or None
            break

        messages.append({
            "role": "assistant",
            "content": resp.content or "",
            "tool_calls": [
                {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in resp.tool_calls
            ],
        })

        for tc in resp.tool_calls:
            try:
                key = (tc["name"], json.dumps(tc["arguments"], sort_keys=True, default=str))
            except Exception:
                key = (tc["name"], str(tc["arguments"]))
            if key in recent_calls:
                result = "[HARNESS] Duplicate call blocked. The tool was NOT executed."
            else:
                recent_calls.append(key)
                fn = tools.get(tc["name"])
                if not fn:
                    result = f"Unknown tool: {tc['name']}"
                else:
                    try:
                        result = fn(**tc["arguments"])
                    except Exception as e:
                        result = f"Tool error: {e}"

            messages.append({"role": "tool", "tool_name": tc["name"], "content": str(result)})

    if final is None:
        try:
            r = provider.chat(messages, tools=None, model=model)
            content = (r.content or "").strip()
            if content:
                final = content
            else:
                fallback = _fallback_from_tool_results(messages)
                final = fallback or "(file-agent produced no answer)"
        except Exception as e:
            fallback = _fallback_from_tool_results(messages)
            final = fallback or f"(file-agent failed to finalize: {e})"

    return final or "(empty)"


@tool("Run an autonomous sub-agent to complete a task", category="agent")
def run_subagent(instruction: str, max_steps: int = 8) -> str:
    """Run an autonomous sub-agent with its own isolated context and full toolset.

    The sub-agent can call ANY tool (files, web, git, search, code, etc.)
    except itself, iterating up to `max_steps` rounds until it produces
    a final answer. Its entire conversation is isolated from the parent —
    only the final text result comes back.

    Use for:
      - multi-step research: "find X, summarize, save to file Y"
      - isolated exploration: "check A and B, compare results"
      - heavy tasks that would clutter the main context

    Args:
        instruction: The task description, as clear and self-contained as possible.
        max_steps: Max tool-call rounds before forcing a final answer (default 8).

    Returns:
        The sub-agent's final answer as plain text.
    """
    return _run_subagent_core(instruction, max_steps=max_steps)


@tool("Run multiple sub-agents in parallel and combine their answers", category="agent")
def run_parallel_subagents(tasks_json: str, max_steps: int = 8) -> str:
    """Run multiple sub-agents in parallel, each with its own isolated context."""
    import threading

    if not tasks_json or not tasks_json.strip():
        return "Error: tasks_json is empty."

    try:
        tasks = json.loads(tasks_json)
    except json.JSONDecodeError as e:
        return f"Error: invalid JSON ({e})"

    if not isinstance(tasks, list) or not tasks:
        return "Error: tasks_json must be a non-empty JSON array."

    norm = []
    for i, t in enumerate(tasks):
        if isinstance(t, str):
            norm.append({"name": f"task-{i+1}", "instruction": t})
        elif isinstance(t, dict):
            name = str(t.get("name") or f"task-{i+1}")
            instr = str(t.get("instruction") or "").strip()
            if instr:
                norm.append({"name": name, "instruction": instr})
    if not norm:
        return "Error: no valid tasks with instructions."

    tools = _subagent_toolset(include_confirm_required=True)
    results = {}
    errors = {}

    def _worker(name, instr):
        try:
            results[name] = _run_subagent_core(instr, max_steps=max_steps, tools=tools)
        except Exception as e:
            errors[name] = str(e)

    threads = []
    for t in norm:
        th = threading.Thread(target=_worker, args=(t["name"], t["instruction"]), daemon=True)
        th.start()
        threads.append(th)

    for th in threads:
        th.join(timeout=600)

    parts = []
    for t in norm:
        name = t["name"]
        if name in results:
            parts.append(f"### {name}\n{results[name]}")
        elif name in errors:
            parts.append(f"### {name}\nERROR: {errors[name]}")
        else:
            parts.append(f"### {name}\n(timed out)")

    return "\n\n".join(parts)


@tool("Delegate a file/command task to a specialized file-system agent", category="agent")
def delegate_file_task(task: str) -> str:
    """Delegate a file or shell task to a focused file-system agent."""
    return _run_file_agent(task)


@tool("Record a plan (informational)", category="agent")
def create_plan(plan: str) -> str:
    """Acknowledge a plan."""
    return "Plan acknowledged."


@tool("Set the session goal", category="agent")
def set_goal(goal: str) -> str:
    """Set a high-level goal."""
    (ws() / ".astronaut_goal").write_text(goal, encoding="utf-8")
    return f"Goal set: {goal}"


@tool("Get the current goal", category="agent")
def get_goal() -> str:
    """Retrieve the current session goal."""
    f = ws() / ".astronaut_goal"
    return f"Current goal: {f.read_text(encoding='utf-8')}" if f.exists() else "No goal set."


# ============================================================
# TOOL REGISTRY
# ============================================================

ALL_TOOLS = [
    list_directory,
    open_file,
    file_info,
    create_file,
    delete_file,
    rename_file,
    append_to_file,
    git_status,
    git_branch,
    git_log,
    git_diff,
    git_remote,
    git_add,
    git_commit,
    git_push,
    git_pull,
    git_checkout,
    git_branch_new,
    ddg_search,
    fetch_url,
    get_weather,
    get_time,
    calculate,
    run_command,
    execute_python,
    rag_index,
    search_workspace,
    rag_stats,
    list_skills_tool,
    load_skill,
    run_subagent,
    run_parallel_subagents,
    delegate_file_task,
    create_plan,
    set_goal,
    get_goal,
]


def filter_tools(mode: str, tools: dict = None) -> dict:
    base = {f.__name__: f for f in ALL_TOOLS}
    if mode == "Minimal":
        keep = {"run_command", "list_directory", "open_file", "create_file", "file_info"}
        return {k: v for k, v in base.items() if k in keep}
    return base


def build_tool_schema(tools: dict = None) -> list:
    if tools is None:
        tools = {f.__name__: f for f in ALL_TOOLS}
    return _build_subagent_schema(tools)