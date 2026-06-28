#!/usr/bin/env python3
"""Unified VLM client for OpenAI / Anthropic / Gemini.

Each provider exposes .call(image_path, prompt, with_tools) returning a normalized
VlmCall with classification, confidence, parsed JSON, raw text, tool trace, usage.

Best-practice improvements (May 2026):
  - MAX_IMAGE_DIM = 512: images downscaled before API submission to reduce image
    token cost (~4x cheaper than 1024 on Anthropic Opus 4.7).
  - Anthropic prompt caching: reference card and prompt are marked with
    cache_control so they are cached across calls (90% token discount on the
    repeated prefix). Content order is [reference, prompt, image] so the two
    cacheable blocks form a stable prefix.
  - OpenAI: automatic prefix caching fires for inputs >1024 tokens (no action
    needed); store=False prevents response storage.
  - Retry: tenacity exponential backoff, 5 attempts, jitter 4–60 s.
  - JSON extraction: multi-strategy (fenced block → bare object → field scan).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from dotenv import load_dotenv
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)
import logging

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "openai":    os.environ.get("OPENAI_MODEL",    "gpt-5.4-mini"),
    "anthropic": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    "gemini":    os.environ.get("GEMINI_MODEL",    "gemini-2.5-pro"),
}

# Images downscaled to 512 px on the longest side before API submission.
# Stored at 1024 for generation quality; 512 is sufficient for classification
# and cuts image token cost ~4x.
MAX_IMAGE_DIM = 512
MAX_OUTPUT_TOKENS = 8192

# Anthropic prompt-caching beta — marks the cacheable prefix
_ANTHROPIC_CACHE_BETA = "prompt-caching-2024-07-31"
_ANTHROPIC_CODE_BETA  = "code-execution-2025-05-22"


@dataclass
class ToolEvent:
    kind: str       # "code" | "code_result" | "web_search" | "web_result"
    payload: Any


@dataclass
class VlmCall:
    provider: str
    model: str
    classification: str | None
    confidence: float | None
    parsed_json: dict | None
    raw_text: str
    tool_trace: list[ToolEvent] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "classification": self.classification,
            "confidence": self.confidence,
            "parsed_json": self.parsed_json,
            "raw_text": self.raw_text,
            "tool_trace": [{"kind": e.kind, "payload": _safe(e.payload)}
                           for e in self.tool_trace],
            "usage": self.usage,
            "latency_s": self.latency_s,
            "error": self.error,
        }


def _safe(v: Any) -> Any:
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


def load_image_bytes(path: Path) -> tuple[bytes, str]:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_final_json(text: str) -> dict | None:
    """Multi-strategy JSON extraction from model output.

    1. Last fenced ```json block (preferred — matches our prompt instruction).
    2. Last bare top-level JSON object in the text.
    3. Scan for required keys and try to parse the last object-like substring.
    """
    if not text:
        return None

    # Strategy 1: last fenced block
    matches = list(_JSON_FENCE.finditer(text))
    if matches:
        for m in reversed(matches):
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue

    # Strategy 2: last bare JSON object
    for start in range(len(text) - 1, -1, -1):
        if text[start] == "{":
            try:
                return json.loads(text[start:])
            except json.JSONDecodeError:
                continue

    return None


class VlmClient:
    provider = ""

    def call(self, image_path: Path, prompt: str, with_tools: bool,
             reference_text: str | None = None) -> VlmCall:
        raise NotImplementedError


# ──────────────────────────── OpenAI ────────────────────────────

class OpenAIClient(VlmClient):
    provider = "openai"

    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or DEFAULT_MODELS["openai"]

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(min=4, max=60),
           retry=retry_if_exception_type(Exception),
           before_sleep=before_sleep_log(logger, logging.WARNING))
    def call(self, image_path: Path, prompt: str, with_tools: bool,
             reference_text: str | None = None) -> VlmCall:
        img_bytes, mime = load_image_bytes(image_path)
        b64 = base64.b64encode(img_bytes).decode()
        data_url = f"data:{mime};base64,{b64}"

        tools = None
        if with_tools:
            tools = [
                {"type": "code_interpreter", "container": {"type": "auto"}},
                {"type": "web_search_preview"},
            ]

        # Order: reference (stable, cacheable prefix) → prompt → image
        # OpenAI auto-caches any prefix >1024 tokens; no explicit action needed.
        content: list[dict] = []
        if reference_text:
            content.append({"type": "input_text", "text": reference_text})
        content.append({"type": "input_text", "text": prompt})
        content.append({"type": "input_image", "image_url": data_url})

        t0 = time.time()
        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            tools=tools,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False,   # do not store responses on OpenAI servers
        )
        latency = time.time() - t0

        text = getattr(resp, "output_text", "") or ""
        trace: list[ToolEvent] = []
        for item in (getattr(resp, "output", None) or []):
            t = getattr(item, "type", "")
            if t == "code_interpreter_call":
                outs = []
                for o in (getattr(item, "outputs", None) or []):
                    dump = getattr(o, "model_dump", None)
                    outs.append(dump() if dump else str(o))
                trace.append(ToolEvent("code", {
                    "code": getattr(item, "code", None),
                    "outputs": outs,
                }))
            elif t == "web_search_call":
                trace.append(ToolEvent("web_search", {
                    "action": getattr(item, "action", None),
                    "status": getattr(item, "status", None),
                }))

        parsed = extract_final_json(text)
        cls  = (parsed or {}).get("classification")
        conf = (parsed or {}).get("confidence")
        u = getattr(resp, "usage", None)
        usage = {}
        if u:
            usage = {
                "input_tokens":  getattr(u, "input_tokens",  None),
                "output_tokens": getattr(u, "output_tokens", None),
                "total_tokens":  getattr(u, "total_tokens",  None),
            }
        return VlmCall(self.provider, self.model, cls, conf, parsed, text,
                       trace, usage, latency)


# ──────────────────────────── Anthropic ────────────────────────────

class AnthropicClient(VlmClient):
    provider = "anthropic"

    def __init__(self, model: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or DEFAULT_MODELS["anthropic"]

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(min=4, max=60),
           retry=retry_if_exception_type(Exception),
           before_sleep=before_sleep_log(logger, logging.WARNING))
    def call(self, image_path: Path, prompt: str, with_tools: bool,
             reference_text: str | None = None) -> VlmCall:
        img_bytes, mime = load_image_bytes(image_path)
        b64 = base64.b64encode(img_bytes).decode()

        # Prompt caching: place stable repeated content BEFORE the image so
        # they form a cacheable prefix. cache_control marks the end of each
        # cached block. The image is unique per call and must come after.
        #
        # Order: reference (cached) → prompt (cached) → image (not cached)
        # Cache discount: 90% on tokens in the cached prefix after the first
        # call within a 5-minute window.
        content: list[dict] = []
        if reference_text:
            content.append({
                "type": "text",
                "text": reference_text,
                "cache_control": {"type": "ephemeral"},
            })
        content.append({
            "type": "text",
            "text": prompt,
            "cache_control": {"type": "ephemeral"},
        })
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": b64},
        })

        beta_parts = [_ANTHROPIC_CACHE_BETA]
        tools = None
        if with_tools:
            beta_parts.append(_ANTHROPIC_CODE_BETA)
            tools = [
                {"type": "code_execution_20250522", "name": "code_execution"},
                {"type": "web_search_20250305",     "name": "web_search", "max_uses": 5},
            ]
        extra_headers = {"anthropic-beta": ",".join(beta_parts)}

        # Only pass tools when non-None — caching beta rejects tools=None
        create_kwargs: dict = dict(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[{"role": "user", "content": content}],
            extra_headers=extra_headers,
        )
        if tools:
            create_kwargs["tools"] = tools

        t0 = time.time()
        resp = self.client.messages.create(**create_kwargs)
        latency = time.time() - t0

        text_parts: list[str] = []
        trace: list[ToolEvent] = []
        for block in resp.content:
            bt = getattr(block, "type", "")
            if bt == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif bt == "server_tool_use":
                name = getattr(block, "name", "") or ""
                kind = "code" if "code" in name else "web_search"
                trace.append(ToolEvent(kind, {
                    "name": name,
                    "input": getattr(block, "input", None),
                }))
            elif bt in ("code_execution_tool_result", "web_search_tool_result", "tool_result"):
                kind = "code_result" if "code" in bt else "web_result"
                contents = []
                for c in (getattr(block, "content", None) or []):
                    dump = getattr(c, "model_dump", None)
                    contents.append(dump() if dump else str(c))
                trace.append(ToolEvent(kind, {"content": contents}))

        text = "\n".join(text_parts)
        parsed = extract_final_json(text)
        cls  = (parsed or {}).get("classification")
        conf = (parsed or {}).get("confidence")
        u = getattr(resp, "usage", None)
        usage = {}
        if u:
            usage = {
                "input_tokens":         getattr(u, "input_tokens",         None),
                "output_tokens":        getattr(u, "output_tokens",        None),
                "cache_creation_tokens": getattr(u, "cache_creation_input_tokens", None),
                "cache_read_tokens":    getattr(u, "cache_read_input_tokens",      None),
            }
        return VlmCall(self.provider, self.model, cls, conf, parsed, text,
                       trace, usage, latency)


# ──────────────────────────── Gemini ────────────────────────────

class GeminiClient(VlmClient):
    provider = "gemini"

    def __init__(self, model: str | None = None):
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = model or DEFAULT_MODELS["gemini"]

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(min=4, max=60),
           retry=retry_if_exception_type(Exception),
           before_sleep=before_sleep_log(logger, logging.WARNING))
    def call(self, image_path: Path, prompt: str, with_tools: bool,
             reference_text: str | None = None) -> VlmCall:
        from google.genai import types
        img_bytes, mime = load_image_bytes(image_path)

        tools = None
        if with_tools:
            tools = [
                types.Tool(code_execution=types.ToolCodeExecution()),
                types.Tool(google_search=types.GoogleSearch()),
            ]

        # Order: reference → prompt → image (consistent with other providers)
        contents: list = []
        if reference_text:
            contents.append(reference_text)
        contents.append(prompt)
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))

        config = types.GenerateContentConfig(
            tools=tools,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ) if tools else types.GenerateContentConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        t0 = time.time()
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        latency = time.time() - t0

        text_parts: list[str] = []
        trace: list[ToolEvent] = []
        for cand in (getattr(resp, "candidates", None) or []):
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in (getattr(content, "parts", None) or []):
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                ec = getattr(part, "executable_code", None)
                if ec:
                    trace.append(ToolEvent("code", {
                        "language": getattr(ec, "language", None),
                        "code":     getattr(ec, "code",     None),
                    }))
                cer = getattr(part, "code_execution_result", None)
                if cer:
                    trace.append(ToolEvent("code_result", {
                        "outcome": str(getattr(cer, "outcome", None)),
                        "output":  getattr(cer, "output",  None),
                    }))
            grounding = getattr(cand, "grounding_metadata", None)
            if grounding:
                for q in (getattr(grounding, "web_search_queries", None) or []):
                    trace.append(ToolEvent("web_search", {"query": q}))

        text = "\n".join(text_parts)
        parsed = extract_final_json(text)
        cls  = (parsed or {}).get("classification")
        conf = (parsed or {}).get("confidence")
        u = getattr(resp, "usage_metadata", None)
        usage = {}
        if u:
            usage = {
                "input_tokens":  getattr(u, "prompt_token_count",     None),
                "output_tokens": getattr(u, "candidates_token_count", None),
                "total_tokens":  getattr(u, "total_token_count",      None),
            }
        return VlmCall(self.provider, self.model, cls, conf, parsed, text,
                       trace, usage, latency)


# ──────────────────────────── Factory + CLI ────────────────────────────

def make_client(provider: str, model: str | None = None) -> VlmClient:
    p = provider.lower()
    if p == "openai":
        return OpenAIClient(model)
    if p == "anthropic":
        return AnthropicClient(model)
    if p == "gemini":
        return GeminiClient(model)
    raise ValueError(f"Unknown provider: {provider}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Single-image VLM call (smoke test).")
    ap.add_argument("--provider", required=True, choices=["openai", "anthropic", "gemini"])
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--prompt-file", required=True, type=Path)
    ap.add_argument("--with-tools", action="store_true",
                    help="Enable code execution + web search (condition B)")
    ap.add_argument("--reference-file", type=Path, default=None,
                    help="Optional Markdown reference passed as a second user text part")
    args = ap.parse_args()

    client = make_client(args.provider)
    prompt = args.prompt_file.read_text()
    reference = args.reference_file.read_text() if args.reference_file else None
    try:
        call = client.call(args.image, prompt, args.with_tools, reference_text=reference)
        print(json.dumps(call.to_dict(), indent=2, default=str))
    except Exception as e:
        print(json.dumps({"error": str(e), "provider": args.provider}, indent=2))
        raise


if __name__ == "__main__":
    main()
