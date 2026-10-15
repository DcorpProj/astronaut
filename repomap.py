# repomap.py
"""Advanced repository map: files, symbols, and the edges between them.

Produces a ranked, budget-aware map that an LLM can use to navigate a
codebase without reading every file. Layered:

  1. scan      — find candidate files, filter by extension and size
  2. parse     — extract symbols + imports via ast (Python) or regex
  3. graph     — build import graph, invert to get importers
  4. rank      — PageRank over the import graph
  5. summarize — optional: cache per-file one-line summaries
  6. render    — budget-aware, ranked output

Cache is workspace-local (.astronaut_repomap.json next to the project).
Set ASTRONAUT_REPOMAP_DEBUG=1 to log scan / cache decisions.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import time
from pathlib import Path


# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════

CACHE_NAME   = ".astronaut_repomap.json"
SUMMARY_NAME = ".astronaut_repomap_summaries.json"

MAX_FILES     = 2000
MAX_FILE_SIZE = 1_000_000        # 1 MB — the old 200_000 was silently
                                 # dropping ordinary source files
MAX_SYMBOLS   = 40
MAX_DEPTH     = 12

# Extensions we both index and attempt to parse for symbols.
CODE_EXTS = {
    # python
    ".py", ".pyi",
    # js / ts
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    # other languages
    ".go", ".rs", ".java", ".kt", ".swift", ".cs",
    ".c", ".cc", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".lua", ".sh", ".bash", ".zsh",
    # web
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".vue", ".svelte", ".astro",
    # docs / config / data
    ".sql", ".md", ".mdx", ".rst", ".txt",
    ".json", ".jsonc", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".env", ".xml",
    ".graphql", ".gql", ".proto",
}

# Well-known extensionless source files.
_EXTENSIONLESS_NAMES = {
    "dockerfile", "makefile", "cmakelists.txt",
    "gemfile", "procfile", "rakefile",
    ".gitignore", ".env", ".editorconfig", ".dockerignore",
}

# Directory names to skip. IMPORTANT: matched against path segments
# RELATIVE to the workspace root, not absolute ones — see _scan_workspace.
IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".rag_index", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", "out", "target",
    ".next", ".nuxt", ".svelte-kit", ".turbo", ".cache",
    ".idea", ".vscode", ".vs",
    "sessions", "skills", ".astronaut",
    "coverage", ".nyc_output",
}

DEBUG = os.environ.get("ASTRONAUT_REPOMAP_DEBUG", "0") == "1"


def _log(*a):
    if DEBUG:
        print("[repomap]", *a, flush=True)


# ════════════════════════════════════════════════════════════
# CACHE PATHS (workspace-local)
# ════════════════════════════════════════════════════════════

def _cache_path(workspace) -> Path:
    """Where the repomap cache lives for this workspace."""
    return Path(workspace).resolve() / CACHE_NAME


def _summary_path(workspace) -> Path:
    return Path(workspace).resolve() / SUMMARY_NAME


# ════════════════════════════════════════════════════════════
# SCAN
# ════════════════════════════════════════════════════════════

def _is_source_file(p: Path) -> bool:
    if p.suffix.lower() in CODE_EXTS:
        return True
    return p.name.lower() in _EXTENSIONLESS_NAMES


def _scan_workspace(root: Path):
    """Return the list of candidate source files under `root`.

    The IGNORE_DIRS check uses the RELATIVE tail of each path, never
    the absolute one. Otherwise a workspace living under, say, `env/`
    or `build/` would be entirely filtered out.
    """
    root = root.resolve()
    entries = []

    if not root.exists():
        _log(f"workspace does not exist: {root}")
        return entries

    skipped = {"ignore": 0, "not_file": 0, "ext": 0, "size": 0, "err": 0}
    truncated = False

    for p in root.rglob("*"):
        try:
            try:
                rel_parts = p.relative_to(root).parts
            except ValueError:
                continue

            if any(part in IGNORE_DIRS for part in rel_parts):
                skipped["ignore"] += 1
                continue

            if not p.is_file():
                skipped["not_file"] += 1
                continue

            if not _is_source_file(p):
                skipped["ext"] += 1
                continue

            try:
                size = p.stat().st_size
            except OSError:
                skipped["err"] += 1
                continue

            if size > MAX_FILE_SIZE:
                skipped["size"] += 1
                _log(f"skip (>{MAX_FILE_SIZE} B): "
                     f"{p.relative_to(root)}  {size} B")
                continue

            entries.append(p)

            if len(entries) >= MAX_FILES:
                truncated = True
                break

        except (OSError, PermissionError):
            skipped["err"] += 1
            continue

    _log(
        f"scan {root} → {len(entries)} files"
        + (f" (capped at {MAX_FILES})" if truncated else "")
        + f"; skipped={skipped}"
    )
    return entries


# ════════════════════════════════════════════════════════════
# PARSE
# ════════════════════════════════════════════════════════════

def _fmt_args(args: ast.arguments) -> str:
    parts = []
    for a in args.posonlyargs:
        parts.append(a.arg)
    if args.posonlyargs:
        parts.append("/")
    for a in args.args:
        parts.append(a.arg)
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")
    for a in args.kwonlyargs:
        parts.append(a.arg)
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)
    return ", ".join(parts)


def _parse_python(source: str, rel_path: str):
    """Return (symbols, imports, calls) for a Python source file."""
    symbols = []
    imports = []
    calls = []

    try:
        tree = ast.parse(source)
    except Exception:
        return symbols, imports, calls

    # imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    # calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                calls.append(fn.id)
            elif isinstance(fn, ast.Attribute):
                calls.append(fn.attr)

    # top-level symbols
    def _walk_body(body, prefix=""):
        out = []
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                out.append({
                    "kind": "func",
                    "name": (prefix + node.name) if prefix else node.name,
                    "sig": f"{kind} {node.name}({_fmt_args(node.args)})",
                })
            elif isinstance(node, ast.ClassDef):
                out.append({
                    "kind": "class",
                    "name": (prefix + node.name) if prefix else node.name,
                    "sig": f"class {node.name}",
                })
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        kind = ("async def" if isinstance(sub, ast.AsyncFunctionDef)
                                else "def")
                        out.append({
                            "kind": "method",
                            "name": f"{node.name}.{sub.name}",
                            "sig": f"    {kind} {sub.name}({_fmt_args(sub.args)})",
                        })
                        if len(out) >= MAX_SYMBOLS:
                            return out
            if len(out) >= MAX_SYMBOLS:
                return out
        return out

    symbols = _walk_body(tree.body)
    return symbols, imports, calls


_RE_DEFS = [
    re.compile(r'^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\('),
    re.compile(r'^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)'),
    re.compile(r'^\s*(?:pub\s+)?fn\s+([A-Za-z_][\w]*)\s*\('),
    re.compile(r'^\s*(?:pub\s+)?(?:struct|enum|trait)\s+([A-Za-z_][\w]*)'),
    re.compile(r'^\s*func\s+(?:\([^)]+\)\s+)?([A-Za-z_][\w]*)\s*\('),
    re.compile(r'^\s*def\s+([A-Za-z_][\w]*)\s*\('),
]


def _parse_generic(source: str):
    """Regex fallback for non-Python languages."""
    symbols = []
    for line in source.splitlines():
        if len(symbols) >= MAX_SYMBOLS:
            break
        for pat in _RE_DEFS:
            m = pat.match(line)
            if m:
                stripped = line.strip()
                if len(stripped) > 120:
                    stripped = stripped[:117] + "…"
                symbols.append({"kind": "def", "name": m.group(1),
                                "sig": stripped})
                break
    return symbols, [], []


def _parse_file(p: Path, rel: str):
    try:
        source = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {"symbols": [], "imports": [], "calls": []}

    if p.suffix.lower() in (".py", ".pyi"):
        symbols, imports, calls = _parse_python(source, rel)
    elif p.suffix.lower() in CODE_EXTS:
        symbols, imports, calls = _parse_generic(source)
    else:
        symbols, imports, calls = [], [], []

    return {"symbols": symbols, "imports": imports, "calls": calls}


# ════════════════════════════════════════════════════════════
# IMPORT GRAPH
# ════════════════════════════════════════════════════════════

def _module_name(rel_path: str) -> str:
    """'src/app/main.py' → 'src.app.main'; 'app.py' → 'app'."""
    p = rel_path.rsplit(".", 1)[0]
    return p.replace("\\", "/").replace("/", ".")


def _build_import_graph(files: list) -> dict:
    """Return {rel_path: {"imports": [rel_path], "imported_by": [rel_path]}}."""
    by_module = {_module_name(f["path"]): f["path"] for f in files}

    def resolve(modname: str):
        if modname in by_module:
            return by_module[modname]
        parts = modname.split(".")
        for i in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in by_module:
                return by_module[candidate]
        return None

    graph = {f["path"]: {"imports": [], "imported_by": []} for f in files}

    for f in files:
        for modname in f.get("imports", []):
            target = resolve(modname)
            if target and target != f["path"]:
                graph[f["path"]]["imports"].append(target)
                graph[target]["imported_by"].append(f["path"])

    for rel in graph:
        graph[rel]["imports"] = list(dict.fromkeys(graph[rel]["imports"]))
        graph[rel]["imported_by"] = list(dict.fromkeys(graph[rel]["imported_by"]))

    return graph


# ════════════════════════════════════════════════════════════
# PAGERANK
# ════════════════════════════════════════════════════════════

def _pagerank(graph: dict, iterations: int = 20, damping: float = 0.85):
    nodes = list(graph.keys())
    n = len(nodes)
    if n == 0:
        return {}

    out_deg = {node: max(1, len(graph[node]["imports"])) for node in nodes}
    ranks = {node: 1.0 / n for node in nodes}

    for _ in range(iterations):
        new = {}
        for node in nodes:
            rank = (1 - damping) / n
            for importer in graph[node]["imported_by"]:
                rank += damping * (ranks[importer] / out_deg[importer])
            new[node] = rank
        ranks = new

    return ranks


# ════════════════════════════════════════════════════════════
# SUMMARY CACHE
# ════════════════════════════════════════════════════════════

def _load_summaries(workspace) -> dict:
    p = _summary_path(workspace)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_summaries(workspace, data):
    try:
        p = _summary_path(workspace)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(data, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    except Exception as e:
        _log(f"summary save failed: {e}")


def file_summaries(workspace, provider, model, max_files=200):
    """Ask the model for a one-line summary of each file. Cached by mtime."""
    root = Path(workspace).resolve()
    cache = _load_summaries(root)
    summaries = {}
    changed = False

    entries = _scan_workspace(root)[:max_files]
    for p in entries:
        rel = str(p.relative_to(root)).replace(os.sep, "/")
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue

        entry = cache.get(rel)
        if entry and entry.get("mtime") == mtime:
            summaries[rel] = entry["summary"]
            continue

        try:
            source = p.read_text(encoding="utf-8", errors="replace")[:6000]
        except Exception:
            continue

        prompt = (
            "Summarize this file in one short sentence (max 15 words). "
            "Say what it does, not what it contains. No preamble.\n\n"
            f"FILE: {rel}\n---\n{source}\n---"
        )
        try:
            resp = provider.chat(
                [{"role": "user", "content": prompt}],
                tools=None, model=model,
            )
            summary = (resp.content or "").strip().split("\n")[0][:200]
        except Exception:
            summary = ""

        if summary:
            summaries[rel] = summary
            cache[rel] = {"mtime": mtime, "summary": summary}
            changed = True

    if changed:
        _save_summaries(root, cache)

    return summaries


# ════════════════════════════════════════════════════════════
# BUILD
# ════════════════════════════════════════════════════════════

def build_map(workspace, with_summaries=False,
              provider=None, model=None, max_summaries: int = 200) -> dict:
    root = Path(workspace).resolve()
    entries = _scan_workspace(root)

    files = []
    for p in entries:
        try:
            rel = str(p.relative_to(root)).replace(os.sep, "/")
            size = p.stat().st_size
        except (OSError, ValueError):
            continue
        parsed = _parse_file(p, rel)
        files.append({
            "path": rel,
            "size": size,
            "symbols": parsed["symbols"],
            "imports": parsed["imports"],
            "calls": parsed["calls"],
        })

    graph = _build_import_graph(files)
    ranks = _pagerank(graph)

    for f in files:
        f["imports"] = graph[f["path"]]["imports"]
        f["imported_by"] = graph[f["path"]]["imported_by"]
        f["rank"] = round(ranks.get(f["path"], 0.0), 6)

    files.sort(key=lambda x: x["rank"], reverse=True)

    # ── Summaries ──────────────────────────────────────────────
    # Always attach whatever's in the summaries cache (keyed by mtime,
    # so stale entries are refreshed the next time /api/repomap/summaries
    # runs). Only *generate* new ones when with_summaries=True.
    summaries = {}
    try:
        cache = _load_summaries(root)
        summaries = {
            rel: entry["summary"]
            for rel, entry in cache.items()
            if isinstance(entry, dict) and "summary" in entry
        }
    except Exception:
        summaries = {}

    if with_summaries and provider and model:
        try:
            new = file_summaries(root, provider, model, max_files=200)
            summaries.update(new)
        except Exception:
            pass

    for f in files:
        if f["path"] in summaries:
            f["summary"] = summaries[f["path"]]

    if with_summaries and provider and model:
        try:
            new = file_summaries(root, provider, model,
                                 max_files=max_summaries)
            summaries.update(new)
        except Exception:
            pass

    return {
        "workspace": str(root),
        "generated_at": time.time(),
        "count": len(files),
        "files": files,
    }


# ════════════════════════════════════════════════════════════
# PERSISTENCE (workspace-local, loud failures)
# ════════════════════════════════════════════════════════════

def save_map(data, path=None):
    try:
        if path is None:
            path = _cache_path(data.get("workspace") or Path.cwd())
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    except Exception as e:
        _log(f"save_map failed ({path}): {e}")


def load_map(workspace, path=None):
    if path is None:
        path = _cache_path(workspace)
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        _log(f"load_map failed ({p}): {e}")
        return None


# ════════════════════════════════════════════════════════════
# STALENESS + ENTRY
# ════════════════════════════════════════════════════════════

def _is_stale(cached: dict, root: Path) -> bool:
    """True if the cached map should be discarded.

    Handles every case the old mtime-only check missed:
      - empty cache (the original bug: an empty `files` list made the
        mtime loop iterate zero times, so the empty result was served
        forever)
      - added files
      - deleted files
      - renamed files
    """
    files = cached.get("files") or []
    count = cached.get("count", 0)
    gen   = cached.get("generated_at", 0)

    # Empty cache is ALWAYS stale.
    if count == 0 or not files:
        return True

    try:
        live = _scan_workspace(root)
    except Exception:
        return True

    if len(live) != count:
        return True

    live_rels = {str(p.relative_to(root)).replace(os.sep, "/") for p in live}
    cached_rels = {f.get("path") for f in files}
    if live_rels != cached_rels:
        return True

    for rel in live_rels:
        try:
            if (root / rel).stat().st_mtime > gen:
                return True
        except OSError:
            return True

    return False


def get_or_build(workspace, force=False, with_summaries=False,
                 provider=None, model=None) -> dict:
    root = Path(workspace).resolve()

    if not force:
        cached = load_map(root)
        if (cached
                and cached.get("workspace") == str(root)
                and not _is_stale(cached, root)):
            _log(f"serving cached map ({cached.get('count')} files)")
            return cached
        _log("cache stale or missing — rebuilding")

    data = build_map(root, with_summaries=with_summaries,
                     provider=provider, model=model)
    save_map(data)
    _log(f"rebuilt: {data['count']} files → {_cache_path(root)}")
    return data


# ════════════════════════════════════════════════════════════
# RENDER
# ════════════════════════════════════════════════════════════

def _default_budget() -> int:
    """Ask _shared for a model-scaled budget. Fall back to 12000."""
    try:
        from _shared import context_budget
        return max(4000, int(context_budget() * 0.15))
    except Exception:
        return 12_000


def render_text(data: dict, max_chars: int = None) -> str:
    if not data:
        return "(repo map is empty)"

    if max_chars is None:
        max_chars = _default_budget()

    lines = []
    header = (f"# Repo map — {data.get('count', 0)} files "
              f"in {data.get('workspace', '?')}")
    lines.append(header)
    lines.append("")

    budget = max_chars - len(header) - 2

    for f in data.get("files", []):
        block = []
        block.append(f"{f['path']}  ({f['size']} B)")

        if f.get("summary"):
            block.append(f"  ⤷ {f['summary']}")

        if f.get("imported_by"):
            importers = f["imported_by"][:6]
            extra = "" if len(f["imported_by"]) <= 6 else \
                    f" (+{len(f['imported_by']) - 6})"
            block.append(f"  ← imported by: {', '.join(importers)}{extra}")

        if f.get("imports"):
            imports = f["imports"][:6]
            extra = "" if len(f["imports"]) <= 6 else \
                    f" (+{len(f['imports']) - 6})"
            block.append(f"  → imports:     {', '.join(imports)}{extra}")

        for sym in f.get("symbols", [])[:MAX_SYMBOLS]:
            block.append(sym["sig"] if "sig" in sym
                         else f"    {sym.get('kind', 'def')} {sym.get('name', '?')}")

        if len(f.get("symbols", [])) >= MAX_SYMBOLS:
            block.append("    ... (more symbols)")

        block_text = "\n".join(block) + "\n"
        if len(block_text) > budget:
            lines.append("... (map truncated; lowest-ranked files omitted)")
            break

        lines.append(block_text)
        budget -= len(block_text)

    return "\n".join(lines)


def summary_one_line(data) -> str:
    if not data:
        return "no map"
    n = data.get("count", 0)
    gen = data.get("generated_at", 0)
    try:
        age = int(time.time() - gen)
        if age < 60:
            age_s = f"{age}s ago"
        elif age < 3600:
            age_s = f"{age//60}m ago"
        else:
            age_s = f"{age//3600}h ago"
    except Exception:
        age_s = "?"
    return f"{n} files · {age_s}"


# ════════════════════════════════════════════════════════════
# INTENT
# ════════════════════════════════════════════════════════════

_REPO_HINTS = (
    "repo", "repository", "project", "codebase",
    "structure", "architecture", "where is", "where's",
    "which file", "what file", "find the file", "find file",
    "refactor", "rename across", "usages", "references",
    "call graph", "import", "module",
    "репозитор", "проект", "структур", "архитектур",
    "где находит", "какой файл", "в каком файле",
    "используется", "ссылки", "импорт",
)


def user_mentions_repo(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(h in low for h in _REPO_HINTS)


# ════════════════════════════════════════════════════════════
# GRAPH / TREE EXPORT
# ════════════════════════════════════════════════════════════

def as_graph(data: dict, max_nodes: int = 400) -> dict:
    """Convert the repomap into a node/edge graph for visualization."""
    if not data:
        return {"nodes": [], "edges": []}

    files = data.get("files", [])[:max_nodes]

    def _cat(path: str) -> str:
        p = path.lower()
        if "/tests/" in p or p.startswith("tests/") or p.endswith("_test.py"):
            return "test"
        if "/migrations/" in p or "/alembic/" in p:
            return "migration"
        if p.endswith((".md", ".rst", ".txt")):
            return "docs"
        if p.endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".cfg")):
            return "config"
        if p.endswith((".html", ".htm", ".css", ".scss", ".sass", ".less",
                       ".vue", ".svelte")):
            return "ui"
        if p.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")):
            return "js"
        if p.endswith((".py", ".pyi")):
            return "py"
        return "other"

    def _label(path: str) -> str:
        return path.rsplit("/", 1)[-1]

    files = sorted(files, key=lambda f: f.get("rank", 0), reverse=True)
    keep = {f["path"] for f in files}

    nodes = []
    for f in files:
        rank = f.get("rank", 0) or 0
        weight = 1.0 + math.log1p(rank * 1000)
        nodes.append({
            "id": f["path"],
            "label": _label(f["path"]),
            "full": f["path"],
            "size": f.get("size", 0),
            "rank": round(rank, 6),
            "weight": round(weight, 3),
            "category": _cat(f["path"]),
            "imports_count": len(f.get("imports", [])),
            "imported_by_count": len(f.get("imported_by", [])),
            "symbol_count": len(f.get("symbols", [])),
            "summary": f.get("summary", ""),
        })

    edges = []
    seen = set()
    for f in files:
        src = f["path"]
        for dst in f.get("imports", []):
            if dst not in keep or (src, dst) in seen:
                continue
            seen.add((src, dst))
            edges.append({"from": src, "to": dst})

    return {
        "nodes": nodes,
        "edges": edges,
        "workspace": data.get("workspace"),
        "generated_at": data.get("generated_at"),
        "count": data.get("count", len(files)),
        "shown": len(nodes),
    }


def as_tree(data: dict) -> dict:
    """Directory tree with files and their symbols."""
    if not data:
        return {"name": "workspace", "type": "dir", "children": []}

    root = {"name": "workspace", "type": "dir", "path": "",
            "children": {}, "files": []}

    for f in data.get("files", []):
        parts = f["path"].split("/")
        node = root
        for i, part in enumerate(parts[:-1]):
            child = node["children"].get(part)
            if not child:
                child = {"name": part, "type": "dir",
                         "path": "/".join(parts[:i + 1]),
                         "children": {}, "files": []}
                node["children"][part] = child
            node = child
        node["files"].append({
            "name": parts[-1],
            "path": f["path"],
            "size": f.get("size", 0),
            "rank": f.get("rank", 0),
            "summary": f.get("summary", ""),
            "imports_count": len(f.get("imports", [])),
            "imported_by_count": len(f.get("imported_by", [])),
            "symbols": f.get("symbols", []),
        })

    def _collapse(node):
        return {
            "name": node["name"],
            "type": "dir",
            "path": node.get("path", ""),
            "dirs": [_collapse(node["children"][n])
                     for n in sorted(node["children"].keys())],
            "files": sorted(node["files"], key=lambda x: x["name"]),
        }

    return _collapse(root)