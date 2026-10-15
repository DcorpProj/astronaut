# mcp_client.py
"""MCP client for Astronaut AI.

- Parallel warm-up: all enabled servers start at once.
- Health monitor: background thread pings each server; dead ones are removed
  from the tool map and reported via `get_dead_servers()`.
"""

import asyncio
import json
import threading
import time
from contextlib import AsyncExitStack
from pathlib import Path

MCP_CONFIG = Path(".astronaut_mcp.json")

DEFAULTS = {"servers": {}}


def load_config() -> dict:
    if MCP_CONFIG.exists():
        try:
            data = json.loads(MCP_CONFIG.read_text(encoding="utf-8"))
            return {**DEFAULTS, **data}
        except Exception:
            pass
    return dict(DEFAULTS)


def save_config(data: dict) -> dict:
    merged = {**DEFAULTS, **(data or {})}
    MCP_CONFIG.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return merged


class MCPManager:
    """Manages MCP server subprocesses on a dedicated event loop."""

    def __init__(self, health_interval: int = 30):
        self._loop = None
        self._thread = None
        self._sessions = {}          # server_name -> {session, stack, tools}
        self._tool_index = {}         # tool_name -> server_name
        self._dead_servers = {}       # server_name -> {"reason": str, "at": float}
        self._started = False

        # health monitor
        self._health_interval = max(10, int(health_interval))
        self._health_enabled = False
        self._health_thread = None

    # ---------------- lifecycle ----------------

    def start(self):
        if self._started:
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        deadline = time.time() + 5
        while self._loop is None and time.time() < deadline:
            time.sleep(0.05)
        self._started = True

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro, timeout=120):
        if self._loop is None:
            raise RuntimeError("MCP event loop not running")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    # ---------------- health monitor ----------------

    def start_health_monitor(self):
        """Start background thread that pings all live servers."""
        if self._health_thread and self._health_thread.is_alive():
            return
        self._health_enabled = True
        self._health_thread = threading.Thread(
            target=self._health_loop, daemon=True, name="mcp-health"
        )
        self._health_thread.start()

    def stop_health_monitor(self):
        self._health_enabled = False

    def _health_loop(self):
        # initial wait so servers finish warming up
        time.sleep(self._health_interval)
        while self._health_enabled:
            try:
                self._health_check()
            except Exception as e:
                print(f"[mcp-health] error: {e}")
            # sleep in small chunks so stop() is responsive
            waited = 0
            while self._health_enabled and waited < self._health_interval:
                time.sleep(1)
                waited += 1

    def _health_check(self):
        """Ping every connected server. Drop the dead ones."""
        for name in list(self._sessions.keys()):
            try:
                self._submit(self._ping(name), timeout=8)
            except Exception as e:
                self._mark_dead(name, str(e))

    async def _ping(self, name):
        sess = self._sessions.get(name)
        if not sess:
            return
        # list_tools is cheap and works as a liveness probe
        await sess["session"].list_tools()

    def _mark_dead(self, name, reason):
        data = self._sessions.pop(name, None)
        if not data:
            return
        for tn, srv in list(self._tool_index.items()):
            if srv == name:
                self._tool_index.pop(tn, None)
        self._dead_servers[name] = {"reason": reason[:200], "at": time.time()}
        print(f"[mcp] server '{name}' died: {reason[:120]}")
        # try to close leftover session cleanly in background
        try:
            self._submit(data["stack"].aclose(), timeout=5)
        except Exception:
            pass

    def get_dead_servers(self) -> dict:
        return dict(self._dead_servers)

    def clear_dead(self):
        self._dead_servers.clear()

    # ---------------- loading (parallel) ----------------

    def load_all(self):
        cfg = load_config()
        servers = cfg.get("servers", {})

        # disconnect disabled / removed servers
        for name in list(self._sessions.keys()):
            if name not in servers or not servers[name].get("enabled", True):
                self._disconnect(name)

        # collect what to connect
        to_connect = []
        for name, server in servers.items():
            if not server.get("enabled", True):
                continue
            if name in self._sessions:
                continue
            to_connect.append((name, server))

        if not to_connect:
            self._start_health_monitor()
            return

        # fire them all at once
        async def _parallel(items):
            results = await asyncio.gather(
                *(self._connect(n, s) for n, s in items),
                return_exceptions=True,
            )
            for (n, _), r in zip(items, results):
                if isinstance(r, Exception):
                    print(f"[mcp] failed to connect '{n}': {r}")

        try:
            self._submit(_parallel(to_connect), timeout=180)
        except Exception as e:
            print(f"[mcp] parallel connect error: {e}")

        self.start_health_monitor()

    def reload(self):
        """Disconnect everything and re-read config."""
        self.clear_dead()
        for name in list(self._sessions.keys()):
            self._disconnect(name)
        self.load_all()

    async def _connect(self, name, server):
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise RuntimeError("Install MCP SDK: pip install mcp")

        command = server.get("command")
        if not command:
            raise ValueError(f"server '{name}': command is required")

        params = StdioServerParameters(
            command=command,
            args=list(server.get("args", [])),
            env=server.get("env") or None,
        )

        stack = AsyncExitStack()
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        tools = await session.list_tools()
        tool_list = getattr(tools, "tools", []) or []

        for t in tool_list:
            self._tool_index[t.name] = name

        self._sessions[name] = {
            "session": session,
            "stack": stack,
            "tools": tool_list,
        }
        print(f"[mcp] connected '{name}': {len(tool_list)} tools")

    def _disconnect(self, name):
        data = self._sessions.pop(name, None)
        if not data:
            return
        for tool_name, srv in list(self._tool_index.items()):
            if srv == name:
                self._tool_index.pop(tool_name, None)
        try:
            self._submit(data["stack"].aclose(), timeout=10)
        except Exception:
            pass
        print(f"[mcp] disconnected '{name}'")

    # ---------------- tools ----------------

    def get_tools(self) -> dict:
        out = {}
        for server_name, data in self._sessions.items():
            for t in data["tools"]:
                out[t.name] = self._make_callable(server_name, t)
        return out

    def get_tool_schemas(self) -> list:
        out = []
        for server_name, data in self._sessions.items():
            for t in data["tools"]:
                out.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema or {"type": "object", "properties": {}},
                    },
                })
        return out

    def _make_callable(self, server_name, tool):
        def _call(**kwargs):
            # if server died since this callable was captured → fail fast
            if server_name not in self._sessions:
                return f"[mcp] server '{server_name}' is not available."
            return self._submit(self._call_tool(server_name, tool.name, kwargs))
        _call.__name__ = tool.name
        _call.__doc__ = tool.description or f"MCP tool from '{server_name}'"
        _call._tool_meta = {
            "description": tool.description or f"MCP tool: {tool.name}",
            "category": "mcp",
            "name": tool.name,
            "mcp_server": server_name,
        }
        _call._requires_confirmation = False
        _call._mcp_schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            },
        }
        return _call

    async def _call_tool(self, server_name, tool_name, args):
        sess = self._sessions.get(server_name)
        if not sess:
            return f"[mcp] server '{server_name}' is not available."
        try:
            result = await sess["session"].call_tool(tool_name, args)
        except Exception as e:
            return f"MCP call error: {e}"

        parts = []
        for c in (result.content or []):
            text = getattr(c, "text", None)
            if text is not None:
                parts.append(text)
            else:
                try:
                    parts.append(json.dumps(c.model_dump(), ensure_ascii=False))
                except Exception:
                    parts.append(str(c))
        return "\n".join(parts) if parts else "(no output)"

    def status(self) -> list:
        cfg = load_config()
        out = []
        for name, server in cfg.get("servers", {}).items():
            sess = self._sessions.get(name)
            tools = []
            if sess:
                tools = [t.name for t in sess["tools"]]
            dead_info = self._dead_servers.get(name)
            out.append({
                "name": name,
                "command": server.get("command", ""),
                "args": server.get("args", []),
                "enabled": server.get("enabled", True),
                "connected": sess is not None,
                "dead": dead_info is not None,
                "dead_reason": (dead_info or {}).get("reason", ""),
                "tools": tools,
                "tool_count": len(tools),
            })
        return out

    def shutdown(self):
        self.stop_health_monitor()
        for name in list(self._sessions.keys()):
            self._disconnect(name)
        if self._loop:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass


def set_server_enabled(name: str, enabled: bool) -> bool:
    cfg = load_config()
    servers = cfg.get("servers", {})
    if name not in servers:
        return False
    servers[name]["enabled"] = bool(enabled)
    save_config(cfg)
    return True


_manager = MCPManager()


def get_manager() -> MCPManager:
    return _manager