#!/usr/bin/env python3
# main.py
# Windows: VT100 + UTF-8 before anything else touches stdout
import sys
import os

if sys.platform == "win32":
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        old = ctypes.c_uint32()
        if k.GetConsoleMode(h, ctypes.byref(old)):
            k.SetConsoleMode(h, old.value | 0x0004)
    except Exception:
        pass
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        import colorama
        colorama.just_fix_windows_console()
    except ImportError:
        pass

import argparse
import importlib.util
import inspect
import pathlib
import threading
import time
import webbrowser

from config import CONFIG, PROVIDER_PRESETS
import settings as settings_store
from mcp_client import get_manager as get_mcp_manager

# ============================================================
# ARGS
# ============================================================
parser = argparse.ArgumentParser(description="Astronaut AI v3.5")
parser.add_argument("--cli",       action="store_true",
                    help="run in the terminal instead of opening the GUI window")
parser.add_argument("--web",       action="store_true",
                    help="open the UI in your browser instead of a native window")
parser.add_argument("--model",     default=None)
parser.add_argument("--provider",  choices=list(PROVIDER_PRESETS.keys()), default=None)
parser.add_argument("--workspace", default=None)
parser.add_argument("--mode",      choices=["Standard", "Code", "Minimal", "Creator"], default=None)
parser.add_argument("--max-steps", type=int, default=None)
parser.add_argument("--no-history", action="store_true")
parser.add_argument("--port",      type=int, default=5000)
parser.add_argument("--host",      default="127.0.0.1")
parser.add_argument("--width",     type=int, default=1280)
parser.add_argument("--height",    type=int, default=820)
args = parser.parse_args()

if args.model:      CONFIG["model"]     = args.model
if args.provider:   CONFIG["provider"]  = args.provider
if args.workspace:  CONFIG["workspace"] = args.workspace
if args.mode:       CONFIG["mode"]      = args.mode
if args.max_steps:  CONFIG["max_steps"] = args.max_steps
if args.no_history: CONFIG["no_history"] = True

os.makedirs(CONFIG["workspace"], exist_ok=True)


# ============================================================
# PLUGIN LOADER
# ============================================================
PLUGINS_DIR = "plugins"


def load_plugins():
    tools = {}
    path = pathlib.Path(PLUGINS_DIR)
    if not path.exists():
        print(f"[!] '{PLUGINS_DIR}' not found")
        return tools

    root = str(path.resolve().parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    for py in sorted(path.glob("*.py")):
        if py.name.startswith("_"):
            continue
        modname = py.stem
        try:
            spec = importlib.util.spec_from_file_location(modname, py)
            mod  = importlib.util.module_from_spec(spec)
            sys.modules[modname] = mod
            spec.loader.exec_module(mod)
            for name, obj in inspect.getmembers(mod, inspect.isfunction):
                if name.startswith("_"):
                    continue
                if getattr(obj, "__module__", "") != modname:
                    continue
                if not obj.__doc__:
                    continue
                tools[name] = obj
        except Exception as e:
            print(f"[!] failed to load {py.name}: {e}")
    return tools


def filter_tools(mode, tools):
    if mode == "Minimal":
        keep = {"run_command", "list_directory", "open_file", "create_file", "file_info"}
        return {k: v for k, v in tools.items() if k in keep}
    return dict(tools)


# ============================================================
# BOOT
# ============================================================
print(" Loading plugins ")
all_tools = load_plugins()

print(" Loading MCP ")
mcp_mgr = get_mcp_manager()
mcp_mgr.start()
mcp_mgr.load_all()
for name, fn in mcp_mgr.get_tools().items():
    if name not in all_tools:
        all_tools[name] = fn

_rt       = settings_store.load()
_disabled = set(_rt.get("disabled_tools", []) or [])

tool_map = filter_tools(CONFIG["mode"], all_tools)
tool_map = {k: v for k, v in tool_map.items() if k not in _disabled}

for name, fn in sorted(tool_map.items()):
    meta = getattr(fn, "_tool_meta", {})
    cat  = meta.get("category", "general")
    cf   = " [confirm]" if getattr(fn, "_requires_confirmation", False) else ""
    mcp  = " [mcp]"     if getattr(fn, "_mcp_schema", None) else ""
    print(f"  [{cat}] {name}{cf}{mcp}")

print(f"\nTotal: {len(tool_map)} tools\n")


# ============================================================
# CLI MODE  —  python main.py --cli
# ============================================================
if args.cli:
    from cli import run_cli
    run_cli(all_tools, tool_map, filter_tools)
    sys.exit(0)


# ============================================================
# WEB MODE  —  python main.py --web
# ============================================================
if args.web:
    from web import create_app
    app = create_app(tool_map, all_tools, load_plugins, filter_tools)
    url = f"http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{args.port}"

    def _open():
        time.sleep(0.8)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()
    print(f"\n→ Web UI: {url}\n")
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)
    sys.exit(0)


# ============================================================
# GUI MODE (DEFAULT)  —  python main.py
# ============================================================
try:
    from web import create_app
    from desktop import run_desktop
except ImportError as e:
    print(f"[!] GUI needs pywebview + flask. Install:\n    pip install pywebview flask")
    print(f"    ({e})")
    print("\nFalling back to CLI...\n")
    from cli import run_cli
    run_cli(all_tools, tool_map, filter_tools)
    sys.exit(0)

sys.exit(run_desktop(
    create_app, tool_map, all_tools, load_plugins, filter_tools,
    title="Astronaut AI",
    width=args.width, height=args.height,
    port=args.port,
))