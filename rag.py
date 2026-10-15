# rag.py
"""Simple RAG for workspace files. Uses Ollama embeddings when available, otherwise keyword match."""

import json
import math
import os
import re
from pathlib import Path

INDEX_DIR = ".rag_index"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

TEXT_EXTS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
    ".js", ".ts", ".html", ".css", ".rs", ".go", ".java",
    ".c", ".cpp", ".h", ".sh", ".bash", ".sql",
}


def _embed(text, model="nomic-embed-text"):
    try:
        import requests
        r = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text[:6000]},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json().get("embedding")
    except Exception:
        pass
    return None


def _tokenize(text):
    return set(re.findall(r"\w+", text.lower()))


def _keyword_score(q, c):
    if not q or not c:
        return 0.0
    return len(q & c) / (len(q) + 1e-9)


def _cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def _chunk(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def index_workspace(workspace, force=False, use_embeddings=True):
    workspace = Path(workspace).resolve()
    index_path = workspace / INDEX_DIR
    index_path.mkdir(exist_ok=True)

    files_indexed = 0
    total_chunks = 0

    for f in workspace.rglob("*"):
        if not f.is_file():
            continue
        if INDEX_DIR in f.parts:
            continue
        if f.suffix.lower() not in TEXT_EXTS:
            continue
        try:
            if f.stat().st_size > 500_000:
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        rel = str(f.relative_to(workspace))
        meta_path = index_path / (rel.replace(os.sep, "__") + ".json")

        if meta_path.exists() and not force:
            try:
                if meta_path.stat().st_mtime >= f.stat().st_mtime:
                    continue
            except OSError:
                pass

        chunks = _chunk(text)
        entries = []
        for i, ch in enumerate(chunks):
            entry = {"file": rel, "chunk": i, "text": ch}
            if use_embeddings:
                emb = _embed(ch)
                if emb:
                    entry["embedding"] = emb
            entries.append(entry)

        meta_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        files_indexed += 1
        total_chunks += len(chunks)

    return {"files_indexed": files_indexed, "chunks": total_chunks}


def query(workspace, question, top_k=5, use_embeddings=True):
    workspace = Path(workspace).resolve()
    index_path = workspace / INDEX_DIR
    if not index_path.exists():
        return []

    q_tokens = _tokenize(question)
    q_emb = _embed(question) if use_embeddings else None

    results = []
    for f in index_path.glob("*.json"):
        try:
            entries = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for e in entries:
            if q_emb and e.get("embedding"):
                score = _cosine(q_emb, e["embedding"])
            else:
                score = _keyword_score(q_tokens, _tokenize(e["text"]))
            if score > 0.05:
                results.append((score, e))

    results.sort(key=lambda x: x[0], reverse=True)
    return [
        {"score": round(s, 3), "file": e["file"], "chunk": e["chunk"], "text": e["text"]}
        for s, e in results[:top_k]
    ]


def index_stats(workspace):
    index_path = Path(workspace).resolve() / INDEX_DIR
    if not index_path.exists():
        return {"files": 0, "chunks": 0}
    files = list(index_path.glob("*.json"))
    chunks = 0
    for f in files:
        try:
            chunks += len(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return {"files": len(files), "chunks": chunks}