# desktop.py
"""Native desktop window for Astronaut AI — wraps web.py in pywebview."""

import os
import socket
import sys
import threading
import time

# Silence WebView2/Chromium log spam BEFORE importing webview.
os.environ.setdefault(
    "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
    "--disable-logging --log-level=3 --disable-features=RendererCodeIntegrity",
)

import webview
from werkzeug.serving import make_server


def _find_free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.4):
                return True
        except OSError:
            time.sleep(0.08)
    return False


def run_desktop(
    create_app_fn,
    tool_map,
    all_tools,
    load_plugins_fn,
    filter_tools_fn,
    title="Astronaut AI",
    width=1280,
    height=820,
    port=5000,
):
    host = "127.0.0.1"
    port = _find_free_port(port)
    url  = f"http://{host}:{port}"

    app = create_app_fn(tool_map, all_tools, load_plugins_fn, filter_tools_fn)

    server = make_server(host, port, app, threaded=True)
    app.config["_server"] = server

    def _serve():
        try:
            server.serve_forever()
        except Exception as e:
            print(f"[desktop] server stopped: {e}")

    threading.Thread(target=_serve, daemon=True, name="flask").start()

    if not _wait_for_server(host, port):
        print(f"[desktop] server failed to come up on {url}")
        try:
            server.shutdown()
        except Exception:
            pass
        return 1

    window = webview.create_window(
        title,
        url,
        width=width,
        height=height,
        min_size=(820, 560),
        confirm_close=False,
    )

    # ── shutdown wiring ────────────────────────────────────────

    _shutdown_lock = threading.Lock()
    _shutdown_done = {"v": False}

    def _hard_exit(code: int = 0):
        with _shutdown_lock:
            if _shutdown_done["v"]:
                return
            _shutdown_done["v"] = True

        print("[desktop] shutting down…", flush=True)

        try:
            server.shutdown()
        except Exception:
            pass

        try:
            time.sleep(0.7)
        except Exception:
            pass

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        os._exit(code)

    def _on_window_closed():
        threading.Thread(target=_hard_exit, args=(0,), daemon=True).start()

    try:
        window.events.closed += _on_window_closed
    except Exception as e:
        print(f"[desktop] could not hook window.closed: {e}")

    # ── exposed JS API ────────────────────────────────────────

    def _quit():
        try:
            window.destroy()
        except Exception:
            pass
        _hard_exit(0)

    def pick_file():
        """Native file picker — returns a list of absolute paths."""
        try:
            result = window.create_file_dialog(
                webview.OPEN_DIALOG,
                allow_multiple=True,
            )
        except Exception as e:
            print(f"[desktop] pick_file failed: {e}")
            return []
        if not result:
            return []
        if isinstance(result, str):
            return [result]
        return [str(p) for p in result]

    def pick_folder():
        """Native folder picker — returns a single absolute path or ''."""
        try:
            result = window.create_file_dialog(webview.FOLDER_DIALOG)
        except Exception as e:
            print(f"[desktop] pick_folder failed: {e}")
            return ""
        if not result:
            return ""
        if isinstance(result, str):
            return result
        try:
            return str(result[0])
        except Exception:
            return ""

    try:
        window.expose(_quit)
        window.expose(pick_file)
        window.expose(pick_folder)
    except Exception as e:
        print(f"[desktop] expose failed: {e}")

    try:
        webview.start(debug=False)
    finally:
        _hard_exit(0)

    return 0