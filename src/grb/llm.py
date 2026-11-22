"""LLM access with per-call usage and cost accounting. Every call returns `(output, Usage)`.

- `OllamaLLM` (default): free local open models served by Ollama (https://ollama.com).
- `ClaudeLLM` (optional, paid): the Anthropic API, if you ever want a frontier-model comparison.
- `FakeLLM`: deterministic heuristics so the pipeline and CI run with no model at all.

Model strings in the config pick the backend: "ollama:qwen2.5:7b-instruct", "claude-opus-5-5", "fake".
"""
from __future__ import annotations

import re
from typing import Protocol, TypeVar

from pydantic import BaseModel

from .types import Usage

T = TypeVar("T", bound=BaseModel)

# USD per million tokens (input, output). Anthropic first-party list prices as of Nov 2026.
# Check https://platform.claude.com/docs/en/about-claude/pricing before publishing numbers.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
}

# Models that accept the server-side refusal fallback. A decline is then retried on a fallback
# model inside the same call instead of silently returning nothing for that question.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


class LLM(Protocol):
    model: str

    def complete(self, system: str, user: str, *, max_tokens: int = 1024,
                 effort: str = "low") -> tuple[str, Usage]: ...

    def parse(self, system: str, user: str, schema: type[T], *, max_tokens: int = 4096,
              effort: str = "low") -> tuple[T | None, Usage]: ...


class OllamaLLM:
    """Local model via Ollama's /api/chat. Structured output uses Ollama's JSON-schema `format`."""

    def __init__(self, model: str, host: str | None = None, num_ctx: int = 8192, timeout: float = 600):
        import os

        import httpx

        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.num_ctx = num_ctx
        self.http = httpx.Client(timeout=timeout)

    def _chat(self, system: str, user: str, max_tokens: int, fmt=None) -> tuple[str, Usage]:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0, "seed": 0, "num_predict": max_tokens, "num_ctx": self.num_ctx},
        }
        if fmt is not None:
            body["format"] = fmt
        r = self.http.post(f"{self.host}/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        usage = Usage(input_tokens=data.get("prompt_eval_count", 0), output_tokens=data.get("eval_count", 0),
                      cost_usd=0.0, calls=1)
        return data["message"]["content"], usage

    def complete(self, system, user, *, max_tokens=1024, effort="low"):
        text, usage = self._chat(system, user, max_tokens)
        return text.strip().splitlines()[0].strip() if text.strip() else "", usage

    def parse(self, system, user, schema, *, max_tokens=4096, effort="low"):
        total = Usage()
        for _ in range(2):                      # small models occasionally emit invalid JSON: retry once
            text, usage = self._chat(system, user, max_tokens, fmt=schema.model_json_schema())
            total += usage
            try:
                return schema.model_validate_json(text), total
            except ValueError:
                continue
        return None, total


class ClaudeLLM:
    def __init__(self, model: str = "claude-opus-5-5", max_retries: int = 4):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(max_retries=max_retries)

    def _kwargs(self, effort: str) -> dict:
        kw: dict = {"model": self.model, "output_config": {"effort": effort}}
        if self.model in _FALLBACK_MODELS:
            kw["betas"] = [_FALLBACK_BETA]
            kw["fallbacks"] = "default"
        return kw

    def _usage(self, resp) -> Usage:
        served = getattr(resp, "model", None) or self.model
        u = resp.usage
        return Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cost_usd=cost_usd(served, u.input_tokens, u.output_tokens),
            calls=1,
        )

    def complete(self, system, user, *, max_tokens=1024, effort="low"):
        resp = self.client.beta.messages.create(
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **self._kwargs(effort),
        )
        if resp.stop_reason == "refusal":
            return "", self._usage(resp)
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text, self._usage(resp)

    def parse(self, system, user, schema, *, max_tokens=4096, effort="low"):
        resp = self.client.beta.messages.parse(
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            **self._kwargs(effort),
        )
        if resp.stop_reason in ("refusal", "max_tokens"):
            return None, self._usage(resp)
        return resp.parsed_output, self._usage(resp)


# --------------------------------------------------------------------------- offline stand-in

_CAP_SPAN = re.compile(r"\b(?:[A-Z][\w'\-]*(?:\s+(?:of|the|and|de|von)\s+|\s+)?)+[A-Z][\w'\-]*\b|\b[A-Z][\w'\-]{2,}\b")
_STOP_CAPS = {"The", "A", "An", "In", "It", "He", "She", "They", "His", "Her", "This", "What",
              "Which", "Who", "When", "Where", "How", "Is", "Was", "Are", "Were", "Did", "Does", "Its", "Their"}


def capitalized_spans(text: str) -> list[str]:
    spans = []
    for m in _CAP_SPAN.finditer(text):
        s = m.group(0).strip()
        words = s.split()
        while words and words[0] in _STOP_CAPS:
            words = words[1:]
        if words:
            spans.append(" ".join(words))
    return spans


def _toks(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class FakeLLM:
    """Deterministic heuristics with the same interface as ClaudeLLM. Reports zero cost."""

    model = "fake"

    def complete(self, system, user, *, max_tokens=1024, effort="low"):
        question = user.rsplit("Question:", 1)[-1].strip()
        context = user.split("Question:", 1)[0]
        qtok = _toks(question)
        best, best_score = "unknown", -1.0
        for sent in re.split(r"(?<=[.!?])\s+", context):
            overlap = len(qtok & _toks(sent))
            for pos, span in enumerate(capitalized_spans(sent)):
                if _toks(span) <= qtok:
                    continue
                score = overlap + 0.01 * pos          # later spans tend to be the object of the fact
                if score > best_score:
                    best, best_score = span, score
        return best, Usage(calls=1)

    def parse(self, system, user, schema, *, max_tokens=4096, effort="low"):
        name = schema.__name__
        if name == "ExtractionResult":
            return self._fake_extract(user, schema), Usage(calls=1)
        if name == "QuestionEntities":
            return schema(entities=capitalized_spans(user.rsplit("Question:", 1)[-1])), Usage(calls=1)
        if name == "JudgeVerdict":
            gold = re.search(r"Gold answer:\s*(.*)", user).group(1)
            pred = re.search(r"Predicted answer:\s*(.*)", user).group(1)
            ok = bool(_toks(gold)) and _toks(gold) <= _toks(pred)
            return schema(correct=ok, rationale="token containment (offline judge)"), Usage(calls=1)
        raise ValueError(f"FakeLLM cannot parse into {name}")

    @staticmethod
    def _fake_extract(user: str, schema):
        text = user.split("Paragraph:", 1)[-1]
        title = text.split(":", 1)[0].strip()
        entities, relations, seen = [], [], set()
        for sent in re.split(r"(?<=[.!?])\s+", text):
            ents = [e for e in capitalized_spans(sent)]
            for e in ents:
                if e not in seen:
                    seen.add(e)
                    entities.append({"name": e, "type": "Thing", "description": sent[:120]})
            for e in ents:
                if e != title:
                    relations.append({"source": title, "relation": "related_to", "target": e, "evidence": sent[:200]})
        return schema.model_validate({"entities": entities, "relations": relations})


def make_llm(model: str) -> LLM:
    if model == "fake":
        return FakeLLM()
    if model.startswith("ollama:"):
        return OllamaLLM(model.removeprefix("ollama:"))
    if model.startswith("claude-"):
        return ClaudeLLM(model)
    raise ValueError(f"unknown model {model!r}: use 'ollama:<name>', 'claude-<id>' or 'fake'")
