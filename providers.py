# providers.py
import json

class ChatResponse:
    def __init__(self, content="", tool_calls=None, thinking=""):
        self.content = content or ""
        self.tool_calls = tool_calls or []
        self.thinking = thinking or ""


class OllamaProvider:
    def __init__(self, keep_alive="30m", temperature=0.4):
        from ollama import chat as _chat, list as _list
        self._chat = _chat
        self._list = _list
        self.keep_alive = keep_alive
        self.temperature = temperature

    def chat(self, messages, tools=None, stream=False,
             model="qwen2.5:7b", think=None, **kw):
        p = {
            "model": model,
            "messages": messages,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
        }
        if tools:
            p["tools"] = tools
        if think is not None:
            p["think"] = think
        if stream:
            p["stream"] = True
            return self._stream(p)
        resp = self._chat(**p)
        return self._norm(resp.message)

    def _stream(self, p):
        for chunk in self._chat(**p):
            m = chunk.message
            if not m:
                continue
            thinking = getattr(m, "thinking", "") or ""
            tcs = []
            for tc in (getattr(m, "tool_calls", None) or []):
                tcs.append({"name": tc.function.name, "arguments": tc.function.arguments})
            yield ChatResponse(m.content or "", tcs, thinking=thinking)

    def _norm(self, m):
        tcs = []
        for tc in (getattr(m, "tool_calls", None) or []):
            tcs.append({"name": tc.function.name, "arguments": tc.function.arguments})
        return ChatResponse(
            getattr(m, "content", "") or "",
            tcs,
            thinking=getattr(m, "thinking", "") or "",
        )

    def list_models(self):
        try:
            return [m["name"] for m in self._list().get("models", [])]
        except Exception:
            return []


class OpenAIProvider:
    def __init__(self, api_key, base_url, temperature=0.4):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.temperature = temperature

    def chat(self, messages, tools=None, stream=False, model="gpt-4o-mini", **kw):
        p = {"model": model, "messages": messages, "temperature": self.temperature}
        if tools:
            p["tools"] = tools
            p["tool_choice"] = "auto"
        if stream:
            p["stream"] = True
            return self._stream(p)
        r = self.client.chat.completions.create(**p)
        return self._norm(r.choices[0].message)

    def _stream(self, p):
        for chunk in self.client.chat.completions.create(**p):
            d = chunk.choices[0].delta
            yield ChatResponse(d.content or "", None)

    def _norm(self, m):
        tcs = []
        for tc in (getattr(m, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            tcs.append({"name": tc.function.name, "arguments": args})
        return ChatResponse(m.content or "", tcs)

    def list_models(self):
        try:
            return [m.id for m in self.client.models.list().data]
        except Exception:
            return []


def get_provider(cfg):
    if cfg.get("provider") == "ollama":
        return OllamaProvider(cfg.get("keep_alive", "30m"), cfg.get("temperature", 0.4))
    if cfg.get("provider") == "openai":
        if not cfg.get("api_key"):
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAIProvider(cfg["api_key"], cfg.get("base_url", "https://api.openai.com/v1"),
                              cfg.get("temperature", 0.4))
    raise ValueError(f"Unknown provider: {cfg.get('provider')}")