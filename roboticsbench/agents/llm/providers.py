"""Model providers for the LLM agent.

* ``anthropic/<model>`` -- native Anthropic SDK (Messages API). Append-only transcript (keeps prompt caching and
  preserved thinking valid); old camera images are cleared server-side with context editing.
* ``openai/<model>``, ``openrouter/<vendor>/<model>``, ``xai/<model>``, ``gemini/<model>``, ``together/<model>``,
  ``fireworks/<model>``, ``deepseek/<model>``, ``vllm/<model>@http://host:port/v1`` -- OpenAI-compatible Chat
  Completions with function calling (images are delivered in a user message after the tool messages; old images
  are pruned client-side).
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
    parse_error: str | None = None


@dataclass
class ModelTurn:
    tool_calls: list[ToolCall]
    text: str = ""
    reasoning: str = ""
    stop_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0
    served_model: str = ""
    stop_details: dict[str, Any] | None = None  # e.g. refusal category / explanation


@dataclass
class ToolOutput:
    call_id: str
    text: str
    images: list[tuple[str, bytes]] = field(default_factory=list)  # (camera, JPEG bytes)
    is_error: bool = False


def _image_label(name: str) -> str:
    """Caption for an attached image: camera views are named after their camera, task documents after their file."""
    return f"document: {name[len('doc:'):]}" if name.startswith("doc:") else f"camera: {name}"


def encode_jpeg(img: np.ndarray, quality: int = 82) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


class Provider:
    vendor = "base"

    default_max_output_tokens = 16000

    def __init__(self, model: str, *, effort: str | None = "high", max_output_tokens: int | None = None,
                 temperature: float | None = None, show_thinking: bool = True, keep_images: int = 3, **kw: Any):
        self.model = model
        self.effort = effort
        self.max_output_tokens = int(max_output_tokens or self.default_max_output_tokens)
        self.temperature = temperature
        self.show_thinking = show_thinking
        self.keep_images = keep_images
        self.extra = kw

    # conversation API ---------------------------------------------------------------
    def start(self, system: str, tools: list[dict[str, Any]], text: str, images: list[tuple[str, bytes]]) -> None:
        raise NotImplementedError

    def generate(self) -> ModelTurn:
        raise NotImplementedError

    def add_results(self, outputs: list[ToolOutput]) -> None:
        raise NotImplementedError

    def add_user_text(self, text: str) -> None:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        return {"provider": self.vendor, "model_name": self.model, "reasoning_effort": self.effort,
                "max_output_tokens": self.max_output_tokens, "temperature": self.temperature}


# ============================================================================ Anthropic
# Models without adaptive thinking / effort: they get a fixed thinking budget instead (Haiku 4.5 and older).
_NO_ADAPTIVE = ("claude-haiku-4", "claude-3", "claude-sonnet-4-5", "claude-opus-4-5", "claude-opus-4-1",
                "claude-opus-4-0", "claude-sonnet-4-0", "claude-opus-4-2", "claude-sonnet-4-2")
_BUDGET_FOR_EFFORT = {"low": 2048, "medium": 4096, "high": 8192, "xhigh": 16000, "max": 24000}


class AnthropicProvider(Provider):
    """Messages API via the native SDK, streamed (long adaptive-thinking turns need a large ``max_tokens``).

    The transcript is append-only: assistant turns are replayed exactly as returned (thinking blocks included), so
    prompt caching and preserved thinking stay valid; old tool results (camera images) are cleared server-side with
    context editing, which does not count as a history edit.
    """

    vendor = "anthropic"
    default_max_output_tokens = 64000

    def __init__(self, model: str, **kw: Any):
        super().__init__(model, **kw)
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=6, timeout=900.0)
        self.messages: list[dict[str, Any]] = []
        self.system: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.adaptive = not self.model.startswith(_NO_ADAPTIVE)

    def start(self, system, tools, text, images):
        self.system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        self.tools = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                      for t in tools]
        self.messages = [{"role": "user", "content": self._content(text, images)}]

    @staticmethod
    def _content(text: str, images: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for cam, data in images:
            blocks.append({"type": "text", "text": _image_label(cam)})
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(data)}})
        return blocks

    def _request(self) -> dict[str, Any]:
        req: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_output_tokens,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
            cache_control={"type": "ephemeral"},
            betas=["context-management-2025-06-27"],
            context_management={"edits": [{
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": int(self.extra.get("clear_trigger_tokens", 60000))},
                "keep": {"type": "tool_uses", "value": int(self.extra.get("keep_tool_uses", 6))},
                "clear_at_least": {"type": "input_tokens", "value": 15000},
            }]},
        )
        if self.adaptive:
            req["thinking"] = {"type": "adaptive", "display": "summarized" if self.show_thinking else "omitted"}
            if self.effort:
                req["output_config"] = {"effort": self.effort}
        elif self.effort:  # Haiku 4.5 and older: no effort parameter, fixed thinking budget instead
            budget = _BUDGET_FOR_EFFORT.get(self.effort, 8192)
            req["thinking"] = {"type": "enabled", "budget_tokens": min(budget, self.max_output_tokens - 2048)}
        elif self.temperature is not None:  # sampling parameters only without thinking, on models that accept them
            req["temperature"] = self.temperature
        # Refusal fallbacks are deliberately OFF by default: a benchmark must score the requested model, not a
        # substitute. Opt in with the provider option fallbacks="default" (or a list of {"model": ...} targets).
        fb = self.extra.get("fallbacks")
        if fb:
            req["betas"] = req["betas"] + ["server-side-fallback-2026-07-01" if fb == "default"
                                           else "server-side-fallback-2026-06-01"]
            req["fallbacks"] = fb
        return req

    def generate(self) -> ModelTurn:
        t0 = time.time()
        with self.client.beta.messages.stream(**self._request()) as stream:
            resp = stream.get_final_message()
        dt = time.time() - t0
        self.messages.append({"role": "assistant", "content": resp.content})
        calls, text, reasoning = [], [], []
        for b in resp.content:
            if b.type == "tool_use":
                calls.append(ToolCall(id=b.id, name=b.name, args=dict(b.input or {})))
            elif b.type == "text":
                text.append(b.text)
            elif b.type == "thinking":
                reasoning.append(getattr(b, "thinking", "") or "")
        u = resp.usage
        usage = {
            "in_tokens": int(getattr(u, "input_tokens", 0) or 0),
            "out_tokens": int(getattr(u, "output_tokens", 0) or 0),
            "cache_read_tokens": int(getattr(u, "cache_read_input_tokens", 0) or 0),
            "cache_write_tokens": int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        }
        details = None
        sd = getattr(resp, "stop_details", None)
        if sd is not None:
            details = {"type": getattr(sd, "type", None), "category": getattr(sd, "category", None),
                       "explanation": getattr(sd, "explanation", None)}
        return ModelTurn(tool_calls=calls, text="\n".join(text), reasoning="\n".join(reasoning),
                         stop_reason=str(resp.stop_reason), usage=usage, duration_s=dt,
                         served_model=getattr(resp, "model", self.model), stop_details=details)

    def add_results(self, outputs):
        blocks = []
        for o in outputs:
            content = self._content(o.text, o.images)
            blocks.append({"type": "tool_result", "tool_use_id": o.call_id, "content": content,
                           **({"is_error": True} if o.is_error else {})})
        self.messages.append({"role": "user", "content": blocks})

    def add_user_text(self, text):
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})


# ================================================================== OpenAI-compatible
OPENAI_COMPAT = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "vllm": ("http://localhost:8000/v1", "VLLM_API_KEY"),
}


class OpenAICompatProvider(Provider):
    def __init__(self, vendor: str, model: str, *, base_url: str | None = None, **kw: Any):
        super().__init__(model, **kw)
        import openai

        self.vendor = vendor
        url, key_env = OPENAI_COMPAT[vendor]
        key = os.environ.get(key_env) or (os.environ.get("GOOGLE_API_KEY") if vendor == "gemini" else None)
        self.client = openai.OpenAI(base_url=base_url or url, api_key=key or "EMPTY", max_retries=6, timeout=900.0)
        self.messages: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []

    def start(self, system, tools, text, images):
        self.tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                         "parameters": t["parameters"]}} for t in tools]
        self.messages = [{"role": "system", "content": system},
                         {"role": "user", "content": self._content(text, images)}]

    @staticmethod
    def _content(text: str, images: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for cam, data in images:
            parts.append({"type": "text", "text": _image_label(cam)})
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(data)}"}})
        return parts

    def _prune_images(self) -> None:
        idx = [i for i, m in enumerate(self.messages) if m["role"] == "user" and isinstance(m["content"], list)
               and any(p.get("type") == "image_url" for p in m["content"])]
        for i in idx[:-self.keep_images] if self.keep_images > 0 else idx:
            self.messages[i]["content"] = [p if p.get("type") != "image_url" else
                                           {"type": "text", "text": "[older image removed]"}
                                           for p in self.messages[i]["content"]]

    def generate(self) -> ModelTurn:
        self._prune_images()
        req: dict[str, Any] = dict(model=self.model, messages=self.messages, tools=self.tools, tool_choice="auto")
        extra_body: dict[str, Any] = {}
        if self.vendor == "openai":
            req["max_completion_tokens"] = self.max_output_tokens
            if self.effort:
                req["reasoning_effort"] = self.effort
        else:
            req["max_tokens"] = self.max_output_tokens
            if self.effort and self.vendor == "openrouter":
                extra_body["reasoning"] = {"effort": self.effort}
            elif self.effort and self.vendor in ("gemini", "xai"):
                req["reasoning_effort"] = self.effort if self.effort in ("low", "medium", "high") else "high"
        if self.temperature is not None:
            req["temperature"] = self.temperature
        if extra_body:
            req["extra_body"] = extra_body
        t0 = time.time()
        resp = self.client.chat.completions.create(**req)
        dt = time.time() - t0
        choice = resp.choices[0]
        msg = choice.message
        calls = []
        tc_dicts = []
        for tc in msg.tool_calls or []:
            raw = tc.function.arguments or "{}"
            try:
                args = json.loads(raw) if raw.strip() else {}
                err = None if isinstance(args, dict) else "arguments must be a JSON object"
            except json.JSONDecodeError as e:
                args, err = {}, f"invalid JSON arguments: {e}"
            calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args if isinstance(args, dict) else {},
                                  parse_error=err))
            tc_dicts.append({"id": tc.id, "type": "function",
                             "function": {"name": tc.function.name, "arguments": raw}})
        amsg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if tc_dicts:
            amsg["tool_calls"] = tc_dicts
        self.messages.append(amsg)
        u = resp.usage
        usage = {}
        if u is not None:
            usage = {"in_tokens": int(u.prompt_tokens or 0), "out_tokens": int(u.completion_tokens or 0)}
            det = getattr(u, "completion_tokens_details", None)
            if det is not None and getattr(det, "reasoning_tokens", None):
                usage["reasoning_tokens"] = int(det.reasoning_tokens)
            pdet = getattr(u, "prompt_tokens_details", None)
            if pdet is not None and getattr(pdet, "cached_tokens", None):
                usage["cache_read_tokens"] = int(pdet.cached_tokens)
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
        return ModelTurn(tool_calls=calls, text=msg.content or "", reasoning=str(reasoning),
                         stop_reason=str(choice.finish_reason), usage=usage, duration_s=dt,
                         served_model=getattr(resp, "model", self.model) or self.model)

    def add_results(self, outputs):
        imgs: list[tuple[str, bytes]] = []
        for o in outputs:
            self.messages.append({"role": "tool", "tool_call_id": o.call_id, "content": o.text})
            imgs += o.images
        if imgs:
            self.messages.append({"role": "user", "content": self._content("Camera images after the last action:", imgs)})

    def add_user_text(self, text):
        self.messages.append({"role": "user", "content": text})


def parse_model_spec(spec: str) -> tuple[str, str, str | None]:
    """'anthropic/claude-opus-5' -> ('anthropic', 'claude-opus-5', None);
    'vllm/qwen@http://h:8000/v1' -> ('vllm', 'qwen', 'http://h:8000/v1')."""
    base_url = None
    if "@" in spec:
        spec, base_url = spec.split("@", 1)
    if "/" not in spec:
        vendor = "anthropic" if spec.startswith("claude") else "openai"
        return vendor, spec, base_url
    vendor, model = spec.split("/", 1)
    return vendor, model, base_url


def make_provider(spec: str, **kw: Any) -> Provider:
    vendor, model, base_url = parse_model_spec(spec)
    if vendor == "anthropic":
        return AnthropicProvider(model, **kw)
    if vendor in OPENAI_COMPAT:
        return OpenAICompatProvider(vendor, model, base_url=base_url, **kw)
    raise KeyError(f"unknown provider {vendor!r} (known: anthropic, {', '.join(OPENAI_COMPAT)})")
