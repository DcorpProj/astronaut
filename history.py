# history.py
import datetime
import json
import os
import uuid
from pathlib import Path

SESSIONS_DIR = "sessions"
CURRENT_SESSION_FILE = "current_session.txt"


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _ensure():
    Path(SESSIONS_DIR).mkdir(exist_ok=True)


def _session_path(sid):
    return Path(SESSIONS_DIR) / f"{sid}.jsonl"


def _meta_path(sid):
    return Path(SESSIONS_DIR) / f"{sid}.meta.json"


def new_session_id():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def create_session(title="New chat"):
    _ensure()
    sid = new_session_id()
    _meta_path(sid).write_text(
        json.dumps({"id": sid, "title": title, "created": _now(), "updated": _now()},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    _session_path(sid).touch()
    return sid


def list_sessions():
    _ensure()
    out = []
    for f in Path(SESSIONS_DIR).glob("*.meta.json"):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    out.sort(key=lambda m: m.get("updated", ""), reverse=True)
    return out


def get_meta(sid):
    p = _meta_path(sid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def update_meta(sid, **kwargs):
    m = get_meta(sid) or {"id": sid, "title": "New chat", "created": _now()}
    m.update(kwargs)
    m["updated"] = _now()
    _meta_path(sid).write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    return m


def delete_session(sid):
    for p in (_session_path(sid), _meta_path(sid)):
        if p.exists():
            p.unlink()


def append_message(sid, msg):
    _ensure()
    msg = dict(msg)
    msg["ts"] = _now()
    with open(_session_path(sid), "a", encoding="utf-8") as f:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    update_meta(sid)


def load_messages(sid, tail=0):
    p = _session_path(sid)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if tail > 0:
        lines = lines[-tail:]
    out = []
    for line in lines:
        try:
            m = json.loads(line)
            m.pop("ts", None)
            out.append(m)
        except json.JSONDecodeError:
            continue
    return out


def clear_messages(sid):
    p = _session_path(sid)
    if p.exists():
        p.unlink()
    _session_path(sid).touch()
    update_meta(sid)


def get_current_session():
    if os.path.exists(CURRENT_SESSION_FILE):
        sid = Path(CURRENT_SESSION_FILE).read_text(encoding="utf-8").strip()
        if get_meta(sid):
            return sid
    s = list_sessions()
    return s[0]["id"] if s else create_session()


def set_current_session(sid):
    Path(CURRENT_SESSION_FILE).write_text(sid, encoding="utf-8")

def clear_history(sid=None):
    """Clear history for a session or all sessions."""
    if sid is None:
        # clear all
        for f in Path(SESSIONS_DIR).glob("*.jsonl"):
            f.unlink()
        return
    clear_messages(sid)

# ---------- branching / editing ----------

def truncate_after(sid, index):
    """Keep messages[0..index] inclusive, drop the rest."""
    msgs = load_messages(sid)
    if index < 0 or index >= len(msgs):
        return False
    keep = msgs[: index + 1]
    p = _session_path(sid)
    if p.exists():
        p.unlink()
    p.touch()
    for m in keep:
        append_message(sid, {k: v for k, v in m.items() if k != "ts"})
    return True


def replace_message(sid, index, new_content):
    """Replace message at index with new content, drop everything after it."""
    msgs = load_messages(sid)
    if index < 0 or index >= len(msgs):
        return False
    msgs[index] = {**msgs[index], "content": new_content}
    keep = msgs[: index + 1]
    p = _session_path(sid)
    if p.exists():
        p.unlink()
    p.touch()
    for m in keep:
        append_message(sid, {k: v for k, v in m.items() if k != "ts"})
    return True


def last_user_index(sid):
    """Return index of the last user message, or -1."""
    msgs = load_messages(sid)
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            return i
    return -1


def find_user_index_for_assistant(sid, assistant_index):
    """Find the user message index preceding an assistant message."""
    msgs = load_messages(sid)
    if assistant_index < 0 or assistant_index >= len(msgs):
        return -1
    for i in range(assistant_index - 1, -1, -1):
        if msgs[i].get("role") == "user":
            return i
    return -1