# skills.py
from pathlib import Path

SKILLS_DIR = "skills"


def _ensure():
    Path(SKILLS_DIR).mkdir(exist_ok=True)


def list_skills():
    _ensure()
    out = []
    for f in sorted(Path(SKILLS_DIR).glob("*.md")):
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        title = f.stem
        desc = ""
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("# ") and title == f.stem:
                title = s[2:].strip()
            elif s and not desc and not s.startswith("#"):
                desc = s
                break
        out.append({"name": f.stem, "title": title, "description": desc})
    return out


def read_skill(name):
    _ensure()
    p = Path(SKILLS_DIR) / f"{name}.md"
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")