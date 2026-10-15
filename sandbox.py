# sandbox.py
import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_ALLOWED_IMPORTS = {
    "os", "sys", "json", "pathlib", "re", "datetime", "time", "math",
    "random", "collections", "itertools", "functools", "typing",
    "csv", "io", "string", "textwrap", "unicodedata", "hashlib",
    "base64", "uuid", "decimal", "statistics", "glob",
}
_FORBIDDEN = {
    "eval", "exec", "compile", "__import__",
    "os.system", "os.popen", "os.exec", "os.spawn",
    "subprocess.run", "subprocess.Popen", "subprocess.call",
}


def _check(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    return False, f"Import '{a.name}' not allowed"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] not in _ALLOWED_IMPORTS:
                return False, f"Import from '{node.module}' not allowed"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN:
                return False, f"Call to '{node.func.id}' not allowed"
            if isinstance(node.func, ast.Attribute):
                full = f"{getattr(node.func.value, 'id', '')}.{node.func.attr}"
                if full in _FORBIDDEN:
                    return False, f"Call to '{full}' not allowed"
    return True, ""


def execute_python(code, workspace, timeout=60):
    ok, err = _check(code)
    if not ok:
        return f"Sandbox rejected: {err}"
    ws = Path(workspace).resolve()
    ws.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                     dir=str(ws), encoding="utf-8") as f:
        f.write(code)
        script = f.name
    try:
        r = subprocess.run([sys.executable, script], cwd=str(ws),
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return f"Exit {r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        return r.stdout or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Timeout ({timeout}s)"
    except Exception as e:
        return f"Execution error: {e}"
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass