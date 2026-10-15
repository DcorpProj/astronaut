# planning.py
"""Plan-first execution for the agent loop.

Generates a numbered plan as JSON before the agent starts working, and
formats it into a system-prompt block that enforces ordered execution.
"""

import json
import re
from typing import Optional


_PLAN_SYSTEM = (
    "You are a planning assistant. Produce a step-by-step plan for the "
    "user's task.\n\n"
    "RULES:\n"
    "  1. Return ONLY a JSON object, no prose, no markdown fences.\n"
    "  2. Shape: {\"steps\": [{\"title\": \"short title\", "
    "\"detail\": \"what this step does\"}, ...]}\n"
    "  3. Between 2 and 8 steps. No more.\n"
    "  4. Each step is ONE concrete action (a tool call, a file write, a "
    "search, an analysis). No vague verbs like \"handle\" or \"process\".\n"
    "  5. Order matters. The first step must be what you'd actually do "
    "first. The last step must be the final user-visible deliverable.\n"
    "  6. If the task is trivial (single tool call, single sentence "
    "answer), return exactly: {\"steps\": []}\n"
    "\n"
    "The user's task follows. Reply with the JSON only."
)


def _strip_json(s: str) -> str:
    """Find the outer-most {...} object in the reply."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        return ""
    return s[start:end + 1]


def generate_plan(provider, model: str, user_text: str,
                  history: Optional[list] = None, tool_names=None, ) -> list:
    """Ask the model for a plan. Returns a list of dicts (possibly empty).

    Never raises — on any failure returns an empty list, which disables
    planning for this turn without affecting the agent loop.
    """
    if not user_text or not user_text.strip():
        return []

    msgs = [{"role": "system", "content": _PLAN_SYSTEM}]

    if history:
        ctx = [m for m in history if m.get("role") != "system"][-4:]
        for m in ctx:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content[:800]})

    msgs.append({"role": "user", "content": f"TASK:\n{user_text}"})

    tools_hint = ""
    if tool_names:
        names = ", ".join(sorted(tool_names))
        tools_hint = f"\n\nAVAILABLE TOOLS:\n{names}\n"

    msgs.append({"role": "user", "content": f"TASK:\n{user_text}{tools_hint}"})

    try:
        resp = provider.chat(msgs, tools=None, model=model, think=False)
        raw = (resp.content or "").strip()
    except Exception:
        return []

    obj = _strip_json(raw)
    if not obj:
        return []

    try:
        data = json.loads(obj)
    except Exception:
        m = re.search(r'\{\s*"steps"\s*:\s*\[.*?\]\s*\}', raw, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []

    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list):
        return []

    clean = []
    for i, s in enumerate(steps, 1):
        if not isinstance(s, dict):
            continue
        title = str(s.get("title") or "").strip()
        detail = str(s.get("detail") or "").strip()
        if not title:
            continue
        clean.append({"n": i, "title": title[:120], "detail": detail[:400]})
        if len(clean) >= 8:
            break

    if len(clean) < 2:
        return []
    return clean


def format_for_prompt(steps: list) -> str:
    """Render the plan as a system-prompt block that enforces ordering."""
    if not steps:
        return ""
    lines = [
        "[EXECUTION PLAN — FOLLOW IN ORDER]",
        "The user's task was decomposed into the following steps. Work them",
        "in order, top to bottom. After finishing a step, move to the next.",
        "Do NOT skip, merge, reorder, or add steps. If a step becomes",
        "impossible, say so and stop — do not silently proceed.",
        "",
    ]
    for s in steps:
        lines.append(f"  Step {s['n']}. {s['title']}")
        if s.get("detail"):
            lines.append(f"          {s['detail']}")
    lines.append("")
    lines.append("When you have completed ALL steps, give the final answer.")
    lines.append("[END EXECUTION PLAN]")
    return "\n".join(lines)


def match_step(steps: list, tool_name: str, tool_args: dict,
               current: int = 0) -> int:
    """Best-effort: which step does this tool call belong to?"""
    if not steps:
        return 0
    haystack_tool = (tool_name or "").lower()
    try:
        haystack_args = json.dumps(tool_args or {}).lower()
    except Exception:
        haystack_args = str(tool_args).lower()

    best_score = 0
    best_step = current or steps[0]["n"]
    for s in steps:
        text = (s["title"] + " " + s.get("detail", "")).lower()
        score = 0
        for tok in set(re.findall(r"[a-z_]{3,}", text)):
            if tok in haystack_tool:
                score += 2
            if tok in haystack_args:
                score += 1
        if s["n"] < (current or 0):
            score -= 1
        if score > best_score:
            best_score = score
            best_step = s["n"]
    return best_step