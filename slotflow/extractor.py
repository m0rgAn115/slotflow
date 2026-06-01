"""LLM-driven extraction + validation for slots.

The module is deliberately ignorant of conversation history and onboarding
flow. It receives a plain string and produces validated values, retrying with
the validation error as feedback when the LLM gets it wrong.

``extract_slot`` handles a single slot and returns its raw value.
``extract_slots`` handles N slots in one LLM call and returns a dict.
Both share ``_build_wrapper_model`` and ``_run_extraction_loop``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from .llm import ChatMessage, ParseError, _resolve_llm, system_msg, user_msg
from .schema import OnboardingSchema

logger = logging.getLogger(__name__)

#: Default number of LLM attempts for an extraction call. Each retry feeds the
#: prior Pydantic ``ValidationError`` back to the model as feedback.
DEFAULT_MAX_ATTEMPTS = 3


class ExtractionError(Exception):
    """Non-recoverable error (e.g. slot name does not exist on the schema)."""


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of an extraction call.

    Attributes
    ----------
    success:
        ``True`` if the LLM produced a Pydantic-valid value within
        ``max_attempts``.
    value:
        The extracted value(s). For ``extract_slot`` this is the raw slot
        value (or ``None``). For ``extract_slots`` it is a ``dict[str, Any]``
        keyed by field name, or ``None`` on failure.
    attempts:
        How many LLM calls were made (always >= 1).
    error:
        The final ``ValidationError`` / ``ParseError`` string when
        ``success`` is ``False``; ``None`` on success.
    """

    success: bool
    value: Any
    attempts: int
    error: str | None


_SYSTEM_PROMPT = (
    "You extract structured fields from free-form user text. "
    "Return only values the user actually stated. "
    "If a value is not present and the field is optional, set it to null. "
    "Do not invent information."
)


def _build_wrapper_model(schema: type[OnboardingSchema], field_names: list[str]) -> type[BaseModel]:
    """Build a one-off Pydantic model whose fields mirror the given slots.

    Uses ``schema.model_fields[name]`` so the annotation is already resolved by
    Pydantic — works correctly under ``from __future__ import annotations``.
    The wrapper carries ``extra="forbid"`` to match ``OnboardingSchema``.
    """
    if not field_names:
        raise ExtractionError("field_names must not be empty")

    fields: dict[str, Any] = {}
    for name in field_names:
        if name not in schema.model_fields:
            raise ExtractionError(f"{name!r} is not a slot of {schema.__name__}")
        field_info = schema.model_fields[name]
        fields[name] = (field_info.annotation, field_info)

    return create_model(
        f"_Extract_{schema.__name__}_{'_'.join(field_names)}",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _build_messages(
    schema: type[OnboardingSchema],
    field_names: list[str],
    wrapper_schema_json: dict[str, Any],
    text: str,
    prior_error: str | None,
) -> list[ChatMessage]:
    field_lines = []
    for name in field_names:
        desc = schema.model_fields[name].description or ""
        field_lines.append(f"- {name}: {desc}".rstrip(": "))

    human_parts = [
        "Fields to extract:\n" + "\n".join(field_lines),
        f"JSON schema for the expected output:\n{wrapper_schema_json}",
        f"User text:\n{text}",
    ]
    if prior_error:
        human_parts.append(
            "Your previous attempt failed validation with this error:\n"
            f"{prior_error}\n"
            "Re-extract and fix the issue."
        )
    return [
        system_msg(_SYSTEM_PROMPT),
        user_msg("\n\n".join(human_parts)),
    ]


async def _run_extraction_loop(
    *,
    schema: type[OnboardingSchema],
    field_names: list[str],
    text: str,
    llm: Any,
    max_attempts: int,
) -> tuple[bool, BaseModel | None, int, str | None, type[BaseModel]]:
    Wrapper = _build_wrapper_model(schema, field_names)
    structured_llm = llm.with_structured_output(Wrapper)
    wrapper_schema_json = Wrapper.model_json_schema()

    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        messages = _build_messages(
            schema=schema,
            field_names=field_names,
            wrapper_schema_json=wrapper_schema_json,
            text=text,
            prior_error=last_error,
        )
        try:
            raw = await structured_llm.ainvoke(messages)
        except (ParseError, ValidationError) as exc:
            last_error = str(exc)
            logger.debug(
                "extraction attempt %d/%d failed (parse/validation): %s",
                attempt,
                max_attempts,
                last_error,
            )
            continue

        try:
            validated = raw if isinstance(raw, Wrapper) else Wrapper.model_validate(raw)
        except ValidationError as exc:
            last_error = str(exc)
            logger.debug(
                "extraction attempt %d/%d failed (post-validation): %s",
                attempt,
                max_attempts,
                last_error,
            )
            continue

        logger.debug(
            "extraction succeeded on attempt %d/%d for %s",
            attempt,
            max_attempts,
            field_names,
        )
        return True, validated, attempt, None, Wrapper

    logger.warning(
        "extraction exhausted %d attempts for %s on schema %s; last error: %s",
        max_attempts,
        field_names,
        schema.__name__,
        last_error,
    )
    return False, None, max_attempts, last_error, Wrapper


async def extract_slot(
    *,
    schema: type[OnboardingSchema],
    field_name: str,
    text: str,
    llm: Any,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> ExtractionResult:
    """Extract and validate one slot value from free-form text.

    Pydantic handles type/enum/date/Optional validation; on ``ValidationError``
    or ``ParseError`` the error text is fed back to the LLM verbatim for
    another attempt, up to ``max_attempts``.

    ``llm`` accepts any LangChain ``BaseChatModel`` (auto-wrapped) or any
    object implementing the ``LLM`` Protocol from ``slotflow.llm``.

    Parameters
    ----------
    schema:
        ``OnboardingSchema`` subclass that declares the target slot.
    field_name:
        Name of the slot to extract. Must exist on ``schema``.
    text:
        The user's raw text. The extractor is history-ignorant; pass a single
        string with any prior context inlined by the caller if needed.
    llm:
        A LangChain ``BaseChatModel``, an adapter from ``slotflow.llm``, or
        any object implementing the ``LLM`` Protocol.
    max_attempts:
        How many times to call the LLM if validation fails. Default ``3``.

    Returns
    -------
    ExtractionResult
        ``success`` flag plus the extracted value (or ``None`` on failure).

    Raises
    ------
    ExtractionError
        If ``field_name`` does not exist on ``schema``.
    """
    llm = _resolve_llm(llm)
    success, validated, attempts, error, _ = await _run_extraction_loop(
        schema=schema,
        field_names=[field_name],
        text=text,
        llm=llm,
        max_attempts=max_attempts,
    )
    value = getattr(validated, field_name) if success and validated else None
    return ExtractionResult(
        success=success,
        value=value,
        attempts=attempts,
        error=error,
    )


async def extract_slots(
    *,
    schema: type[OnboardingSchema],
    field_names: list[str],
    text: str,
    llm: Any,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> ExtractionResult:
    """Extract and validate N slots from one piece of text in a single LLM call.

    On success, ``result.value`` is a ``dict[str, Any]`` keyed by the field
    names. If ANY field fails validation the whole batch is retried — the LLM
    receives the combined Pydantic error and re-emits the entire object.

    Parameters
    ----------
    schema:
        ``OnboardingSchema`` subclass that declares the target slots.
    field_names:
        Names of the slots to extract. All must exist on ``schema``.
    text:
        The user's raw text.
    llm:
        A LangChain ``BaseChatModel``, an adapter from ``slotflow.llm``, or
        any object implementing the ``LLM`` Protocol.
    max_attempts:
        How many times to call the LLM if validation fails. Default ``3``.

    Returns
    -------
    ExtractionResult
        ``success`` flag plus a ``dict[str, Any]`` of slot values (or ``None``
        on failure).

    Raises
    ------
    ExtractionError
        If ``field_names`` is empty or contains a name not on ``schema``.
    """
    llm = _resolve_llm(llm)
    success, validated, attempts, error, _ = await _run_extraction_loop(
        schema=schema,
        field_names=field_names,
        text=text,
        llm=llm,
        max_attempts=max_attempts,
    )
    value: dict[str, Any] | None = None
    if success and validated:
        value = {name: getattr(validated, name) for name in field_names}
    return ExtractionResult(
        success=success,
        value=value,
        attempts=attempts,
        error=error,
    )
