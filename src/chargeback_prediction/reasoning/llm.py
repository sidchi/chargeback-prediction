"""Provider-agnostic chat-completion interface, a deterministic mock, and an HTTP adapter.

The reasoning layer depends only on :class:`LLMClient`: one method that takes messages and
returns text. Anything that can do that — a hosted API, a local model server, a test double —
plugs in without touching the prompt, validation or fallback logic.

* :class:`MockLLM` writes a rationale from the structured context embedded in the prompt. It is
  the default for the CLI and tests, so the pipeline runs offline and deterministically, and it
  proves the prompt carries everything a model needs.
* :class:`ScriptedLLM` replays canned replies (including broken ones) to exercise guardrails.
* :class:`OpenAICompatibleClient` speaks the widely implemented ``/chat/completions`` wire
  format using only the standard library. The API key is read from the environment by the
  caller and is never logged or included in ``repr``.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]

CONTEXT_OPEN = "<context>"
CONTEXT_CLOSE = "</context>"


class LLMError(RuntimeError):
    """The provider could not produce a completion (network, HTTP status, malformed reply)."""


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    messages: tuple[Message, ...]
    temperature: float = 0.0
    """Zero by default: rationales should be reproducible for audit."""
    max_tokens: int = 600
    json_mode: bool = True
    """Ask the provider for a JSON object reply where it supports that."""

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("a completion request needs at least one message")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")


@runtime_checkable
class LLMClient(Protocol):
    @property
    def name(self) -> str:
        """Provider and model, for provenance on every rationale (e.g. ``mock``)."""
        ...

    def complete(self, request: CompletionRequest) -> str:
        """Return the assistant's reply text.

        Raises:
            LLMError: if no completion could be obtained.
        """
        ...


def extract_context(messages: Iterable[Message]) -> dict[str, Any]:
    """The JSON payload between ``<context>`` tags in the latest user message that has one."""
    for message in reversed(tuple(messages)):
        if message.role != "user":
            continue
        start = message.content.find(CONTEXT_OPEN)
        end = message.content.find(CONTEXT_CLOSE)
        if start != -1 and end > start:
            payload = json.loads(message.content[start + len(CONTEXT_OPEN) : end])
            if not isinstance(payload, dict):
                raise ValueError("context payload must be a JSON object")
            return payload
    raise ValueError("no <context> block in the request")


class MockLLM:
    """Deterministic stand-in for a model: reads the prompt's context, writes a valid reply."""

    @property
    def name(self) -> str:
        return "mock"

    def complete(self, request: CompletionRequest) -> str:
        try:
            ctx = extract_context(request.messages)
            playbook = ctx["playbook"]
            dispute = ctx["dispute"]
            model = ctx["model"]
            factors: list[dict[str, Any]] = ctx["top_factors"]
            signals: list[dict[str, Any]] = ctx["signals"]
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMError(f"mock could not read the prompt context: {exc}") from exc

        strategy = playbook["recommended_strategy"]
        opening = (
            f"Filed under {dispute['reason_code']} ({dispute['title']}). Before the dispute, "
            f"the model scored this charge at {model['dispute_probability']}, "
            f"{model['relative_risk']}× a typical transaction."
        )
        drivers = [f"{f['feature']} = {f['value']} ({f['odds_multiplier']} odds)" for f in factors]
        sentences = [opening]
        if drivers:
            sentences.append("The score was driven by " + ", ".join(drivers[:3]) + ".")
        if signals:
            sentences.append(
                "Playbook signals: " + "; ".join(s["detail"] for s in signals[:3]) + "."
            )
        sentences.append(f"Recommended action: {strategy}. {' '.join(playbook['reasons'])}")

        key = [f["feature"] for f in factors[:3]] + [s["feature"] for s in signals]
        confidence = "high" if strategy != "review" and len(signals) >= 2 else "medium"
        if strategy == "review":
            confidence = "low"
        return json.dumps(
            {
                "strategy": strategy,
                "confidence": confidence,
                "rationale": " ".join(sentences),
                "key_factors": list(dict.fromkeys(key)),
            }
        )


@dataclass
class ScriptedLLM:
    """Returns ``replies`` in order (an exception instance is raised instead) and records calls."""

    replies: Sequence[str | Exception]
    name: str = "scripted"
    requests: list[CompletionRequest] = field(default_factory=list)

    def complete(self, request: CompletionRequest) -> str:
        self.requests.append(request)
        if len(self.requests) > len(self.replies):
            raise LLMError("scripted replies exhausted")
        reply = self.replies[len(self.requests) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply


_SCHEME = re.compile(r"^https?://", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class OpenAICompatibleClient:
    """Minimal client for any server implementing ``POST {base_url}/chat/completions``."""

    base_url: str
    """e.g. ``http://localhost:11434/v1`` for a local model server."""
    model: str
    api_key: str | None = field(default=None, repr=False)
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not _SCHEME.match(self.base_url):
            raise ValueError("base_url must start with http:// or https://")
        if not self.model:
            raise ValueError("model must be set")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")

    @property
    def name(self) -> str:
        return f"openai-compatible:{self.model}"

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def complete(self, request: CompletionRequest) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(
            self.endpoint, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        # A fresh opener per call picks up proxy settings from the environment as they are now.
        opener = urllib.request.build_opener()
        try:
            with opener.open(http_request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise LLMError(f"HTTP {exc.code} from {self.endpoint}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"could not reach {self.endpoint}: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMError(f"non-JSON response from {self.endpoint}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("response has no choices[0].message.content") from exc
        if not isinstance(content, str):
            raise LLMError("choices[0].message.content is not a string")
        return content
