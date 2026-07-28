"""LLM provider interface.

One interface, two implementations: `none` (the default) and `anthropic`. The
whole application is specified to work at `LLM_PROVIDER=none`, so the null
provider is not a stub — it is a first-class path that every test exercises.

Nothing in this module decides *whether* to generate. That decision belongs to
the Phase 6 gate, which runs on backend-computed retrieval strength before a
provider is ever constructed. A weak-retrieval request must make zero provider
calls, and the cheapest way to guarantee that is for the pipeline never to reach
here at all.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from backend.app.core.config import Settings, get_settings

# Providers that speak the OpenAI chat-completions shape. All run permanent free
# tiers, which is what makes an LLM path viable on no budget.
OPENAI_COMPATIBLE: frozenset[str] = frozenset(
    {"openai_compatible", "groq", "gemini", "cerebras", "openrouter", "together", "mistral"}
)

# Sensible base URLs so a user only has to set provider + key. Overridable via
# LLM_BASE_URL; never treated as authoritative, since vendors move endpoints.
DEFAULT_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "cerebras": "https://api.cerebras.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "mistral": "https://api.mistral.ai/v1",
}


class LLMError(Exception):
    """Any provider failure. Callers fall back to deterministic mode."""


class LLMTimeout(LLMError):
    pass


class LLMRefusal(LLMError):
    """The provider's safety classifiers declined the request.

    A normal outcome, not a bug: support tickets legitimately discuss account
    takeover, fraud, and payment abuse, which sit near policy boundaries. The
    pipeline treats it exactly like any other generation failure and falls back
    to deterministic mode.
    """

    def __init__(self, category: str | None, explanation: str | None) -> None:
        super().__init__(f"provider refused the request (category={category})")
        self.category = category
        self.explanation = explanation


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str | None = None
    stop_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def usage(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "model": self.model,
            "stop_reason": self.stop_reason,
        }


class LLMProvider(Protocol):
    name: str
    enabled: bool

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], max_tokens: int
    ) -> LLMResponse:
        """Return raw model text expected to contain a single JSON object."""
        ...


# --- null provider ------------------------------------------------------------


class NullProvider:
    """`LLM_PROVIDER=none`. Refuses to pretend it produced anything."""

    name = "none"
    enabled = False

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], max_tokens: int
    ) -> LLMResponse:
        raise LLMError(
            "no LLM provider is configured (LLM_PROVIDER=none). The caller should "
            "never have reached a provider — deterministic mode handles this request."
        )


# --- anthropic provider -------------------------------------------------------


class AnthropicProvider:
    """Anthropic Messages API.

    Two model-specific details that are easy to get wrong and expensive to miss:

    * **No sampling parameters.** `temperature`, `top_p`, and `top_k` are removed
      on current Claude models and a request carrying one returns a 400. Output
      shape is controlled by the schema, not by turning temperature down.
    * **`max_tokens` bounds thinking and response text together**, and thinking is
      on by default. A budget sized for the JSON alone truncates it mid-object.
    """

    name = "anthropic"
    enabled = True

    def __init__(self, settings: Settings | None = None) -> None:
        import anthropic  # imported lazily so `none` mode needs no dependency

        self.settings = settings or get_settings()
        if not self.settings.llm_api_key:
            raise LLMError("LLM_PROVIDER=anthropic but no API key is configured")
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(
            api_key=self.settings.llm_api_key,
            timeout=float(self.settings.llm_timeout_seconds),
            max_retries=1,
        )
        self.model = self.settings.llm_model

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], max_tokens: int
    ) -> LLMResponse:
        t0 = time.perf_counter()
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
        except self._anthropic.APITimeoutError as exc:
            raise LLMTimeout(f"provider timed out after {self.settings.llm_timeout_seconds}s") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach the provider: {exc}") from exc
        except self._anthropic.RateLimitError as exc:
            raise LLMError(f"provider rate limited the request: {exc}") from exc
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"provider returned {exc.status_code}: {exc.message}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Check stop_reason BEFORE reading content: on a refusal the content list
        # is empty (or partial), and indexing it blindly raises IndexError far
        # from the actual cause.
        if getattr(message, "stop_reason", None) == "refusal":
            details = getattr(message, "stop_details", None)
            raise LLMRefusal(
                getattr(details, "category", None), getattr(details, "explanation", None)
            )

        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(message, "usage", None)
        return LLMResponse(
            text=text,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            latency_ms=latency_ms,
            model=getattr(message, "model", self.model),
            stop_reason=getattr(message, "stop_reason", None),
        )


class OpenAICompatibleProvider:
    """Any provider speaking the OpenAI chat-completions shape.

    One class covers Groq, Google Gemini (its OpenAI-compatible endpoint),
    Cerebras, OpenRouter, Together, and Mistral. All of them run permanent free
    tiers, which is what makes an LLM path realistic for a project with no
    budget. The differences that matter are a base URL and a model name, so both
    are configuration rather than code.

    Deliberately not doing: hard-coding any provider's free-tier rate limits as
    fact. Those change, and a number baked into source becomes a lie on a
    schedule. Failures surface as `LLMError` and the pipeline falls back.
    """

    name = "openai_compatible"
    enabled = True

    def __init__(self, settings: Settings | None = None) -> None:
        import httpx

        self.settings = settings or get_settings()
        if not self.settings.llm_api_key:
            raise LLMError("LLM_PROVIDER is set but no API key is configured")

        provider = (self.settings.llm_provider or "").strip().lower()
        base_url = self.settings.llm_base_url or DEFAULT_BASE_URLS.get(provider)
        if not base_url:
            raise LLMError(
                f"no base URL for provider {provider!r}. Set LLM_BASE_URL, or use one "
                f"of {sorted(DEFAULT_BASE_URLS)}."
            )

        self._httpx = httpx
        self.model = self.settings.llm_model
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=float(self.settings.llm_timeout_seconds),
            headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
        )

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], max_tokens: int
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Widely supported; providers that ignore it still tend to return an
            # object because the prompt demands one, and extract_json_object
            # tolerates a stray fence either way.
            "response_format": {"type": "json_object"},
        }

        t0 = time.perf_counter()
        try:
            res = self.client.post("/chat/completions", json=payload)
        except self._httpx.TimeoutException as exc:
            raise LLMTimeout(
                f"provider timed out after {self.settings.llm_timeout_seconds}s"
            ) from exc
        except self._httpx.HTTPError as exc:
            raise LLMError(f"could not reach the provider: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if res.status_code == 429:
            raise LLMError("provider rate limited the request (free tier quota)")
        if res.status_code >= 400:
            raise LLMError(f"provider returned {res.status_code}: {res.text[:200]}")

        try:
            body = res.json()
            choice = body["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"unexpected provider response shape: {exc}") from exc

        # A refusal here arrives as a finish_reason rather than a status code.
        if choice.get("finish_reason") == "content_filter":
            raise LLMRefusal("content_filter", "provider content filter declined the request")

        usage = body.get("usage") or {}
        return LLMResponse(
            text=text,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
            model=body.get("model", self.model),
            stop_reason=choice.get("finish_reason"),
        )


# --- factory ------------------------------------------------------------------


def build_provider(settings: Settings | None = None) -> LLMProvider:
    """Return the configured provider, or the null provider on any doubt.

    Never raises for configuration reasons. A misconfigured provider degrades to
    deterministic mode with a recorded reason; it does not take the app down.
    """
    s = settings or get_settings()
    provider = (s.llm_provider or "none").strip().lower()
    if provider in ("", "none"):
        return NullProvider()
    if provider == "anthropic":
        try:
            return AnthropicProvider(s)
        except (LLMError, ImportError):
            return NullProvider()
    if provider in OPENAI_COMPATIBLE:
        try:
            return OpenAICompatibleProvider(s)
        except (LLMError, ImportError):
            return NullProvider()
    return NullProvider()


# --- JSON extraction ----------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the single JSON object a generation prompt asked for.

    Structured output makes a bare object the normal case, but this stays
    tolerant of a markdown fence or surrounding prose: a recoverable formatting
    slip should not cost the request its generated content. Anything genuinely
    unparseable raises, and the caller falls back deterministically.
    """
    if not text or not text.strip():
        raise LLMError("provider returned empty text")

    candidates: list[str] = [text.strip()]

    fenced = _FENCE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise LLMError("provider output did not contain a JSON object")
