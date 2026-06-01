"""LLM protocol, message types, and built-in adapters.

The library does not hard-require any specific LLM provider. Any object that
satisfies the ``LLM`` Protocol can be used. LangChain ``BaseChatModel``
instances are auto-detected and wrapped transparently — existing code that
passes ``ChatOpenAI(...)`` continues to work without changes.

Built-in adapters
-----------------
``from_langchain(llm)``
    Explicitly wrap a LangChain BaseChatModel. Requires ``slotflow[langchain]``.
``from_openai(client, model)``
    Use ``openai.AsyncOpenAI`` directly without LangChain. Requires
    ``slotflow[openai]`` (openai >= 1.50 for ``beta.chat.completions.parse``).

Auto-detection
--------------
Public functions (``extract_slot``, ``next_message``, etc.) call
``_resolve_llm`` on the ``llm`` argument before doing anything. If the object
is a LangChain ``BaseChatModel``, it is wrapped automatically. Any other
object is passed through unchanged — it is assumed to already implement the
Protocol. This means passing ``ChatOpenAI(...)`` directly keeps working even
after the LangChain dependency was decoupled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class ParseError(Exception):
    """Raised when a structured LLM call cannot produce a parseable output.

    Adapters convert provider-specific parser exceptions (e.g. LangChain's
    ``OutputParserException``) into this so callers do not need to import
    any specific provider package to catch parse failures.
    """


@dataclass(frozen=True)
class ChatMessage:
    """An immutable chat message."""

    role: str  # "system" | "user" | "assistant"
    content: str


def system_msg(content: str) -> ChatMessage:
    return ChatMessage(role="system", content=content)


def user_msg(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


class StructuredOutput(Protocol):
    """An LLM bound to a specific Pydantic output model."""

    async def ainvoke(self, messages: list[ChatMessage]) -> Any:
        ...


@runtime_checkable
class LLM(Protocol):
    """Minimal interface for a language model with structured output support."""

    def with_structured_output(self, schema: type[BaseModel]) -> StructuredOutput:
        ...


# ── LangChain adapter ─────────────────────────────────────────────────────── #


class _LangChainStructuredOutput:
    __slots__ = ("_runnable",)

    def __init__(self, runnable: Any) -> None:
        self._runnable = runnable

    async def ainvoke(self, messages: list[ChatMessage]) -> Any:
        from langchain_core.messages import HumanMessage, SystemMessage

        lc: list[Any] = []
        for msg in messages:
            if msg.role == "system":
                lc.append(SystemMessage(content=msg.content))
            else:
                lc.append(HumanMessage(content=msg.content))
        try:
            return await self._runnable.ainvoke(lc)
        except Exception as exc:
            if type(exc).__name__ == "OutputParserException":
                raise ParseError(str(exc)) from exc
            raise


class _LangChainAdapter:
    __slots__ = ("_llm",)

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def with_structured_output(self, schema: type[BaseModel]) -> _LangChainStructuredOutput:
        return _LangChainStructuredOutput(self._llm.with_structured_output(schema))


def from_langchain(llm: Any) -> LLM:
    """Explicitly wrap a LangChain BaseChatModel for use with slotflow.

    Requires: ``pip install 'slotflow[langchain]'``
    """
    return _LangChainAdapter(llm)


# ── OpenAI adapter (no LangChain) ─────────────────────────────────────────── #


class _OpenAIStructuredOutput:
    __slots__ = ("_client", "_model", "_temperature", "_schema")

    def __init__(self, client: Any, model: str, temperature: float, schema: type[BaseModel]):
        self._client = client
        self._model = model
        self._temperature = temperature
        self._schema = schema

    async def ainvoke(self, messages: list[ChatMessage]) -> Any:
        try:
            response = await self._client.beta.chat.completions.parse(
                model=self._model,
                temperature=self._temperature,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                response_format=self._schema,
            )
            result = response.choices[0].message.parsed
            if result is None:
                raise ParseError("OpenAI returned null for structured output")
            return result
        except ParseError:
            raise
        except ValidationError as exc:
            raise ParseError(str(exc)) from exc
        except Exception as exc:
            name = type(exc).__name__
            # OpenAI SDK raises LengthFinishReasonError or similar on truncated output
            if "FinishReason" in name or "Parse" in name:
                raise ParseError(str(exc)) from exc
            raise


class _OpenAIAdapter:
    __slots__ = ("_client", "_model", "_temperature")

    def __init__(self, client: Any, model: str, temperature: float) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    def with_structured_output(self, schema: type[BaseModel]) -> _OpenAIStructuredOutput:
        return _OpenAIStructuredOutput(self._client, self._model, self._temperature, schema)


def from_openai(client: Any, model: str, *, temperature: float = 0.0) -> LLM:
    """Wrap an ``openai.AsyncOpenAI`` client for use with slotflow.

    Uses ``client.beta.chat.completions.parse`` for native structured outputs.

    Requires: ``pip install 'slotflow[openai]'``  (openai >= 1.50)

    Example::

        from openai import AsyncOpenAI
        from slotflow import from_openai

        llm = from_openai(AsyncOpenAI(), model="gpt-4o-mini")
    """
    return _OpenAIAdapter(client, model, temperature)


# ── Auto-detection ─────────────────────────────────────────────────────────── #


def _resolve_llm(llm: Any) -> Any:
    """Auto-wrap LangChain models; pass Protocol-conformant objects through.

    Called at the entry point of every public function that accepts an LLM.
    Safe to call multiple times on the same object — wrapping a
    ``_LangChainAdapter`` again returns it unchanged (it is not a
    ``BaseChatModel``).
    """
    try:
        from langchain_core.language_models import BaseChatModel

        if isinstance(llm, BaseChatModel):
            return _LangChainAdapter(llm)
    except ImportError:
        pass
    return llm
