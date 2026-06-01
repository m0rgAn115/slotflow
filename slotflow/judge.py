"""LLM-based classification of a user's response to an onboarding question.

The judge is the second LLM call in a turn (always after extraction). Its job
is to decide whether to advance, follow up, or treat the slot as
skipped/refused.

It never raises to the caller. If the LLM fails after retries, it returns a
default ``INSUFFICIENT`` verdict so the conversation keeps moving — mirroring
the contract of ``extract_slot`` at ``extractor.py``.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from .llm import ParseError, _resolve_llm, system_msg, user_msg
from .schema import OnboardingSchema

logger = logging.getLogger(__name__)

#: Default attempts for a judge call. Smaller than the extractor's default
#: because the judge falls back to a safe ``INSUFFICIENT`` verdict on
#: exhaustion — retrying many times only adds latency.
DEFAULT_MAX_ATTEMPTS = 2


class ResponseJudgement(str, Enum):
    """How the user's response should be acted upon.

    ``PARTIAL`` is only meaningful in multi-slot turns (FREEFORM / UNORDERED).
    In single-slot turns the judge must pick ``COMPLETE`` or ``INSUFFICIENT``.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    SKIP_INTENT = "skip_intent"
    REFUSED = "refused"


class JudgementResult(BaseModel):
    """Structured output of the judge LLM call."""

    model_config = ConfigDict(extra="forbid")

    verdict: ResponseJudgement
    reasoning: str
    refers_to_slots: tuple[str, ...] = ()


_SYSTEM_PROMPT = (
    "You classify how a user responded to a question in an onboarding "
    "conversation. Output exactly one verdict:\n"
    "- complete: the user provided a usable answer for all target slot(s).\n"
    "- partial: there are 2+ target slots and the user answered some but not all.\n"
    "- insufficient: the user attempted to answer but the value is ambiguous, "
    "incomplete, or off-topic — a clarifying follow-up for the SAME slot is needed.\n"
    "- skip_intent: the user signaled they want to skip the slot "
    "(e.g. 'I'd rather not say', 'later', 'skip', 'next', 'pass', "
    "'I don't want to answer'). Use this even if the slot is required — the runner "
    "decides whether to honor the skip based on whether the slot is optional.\n"
    "- refused: the user explicitly refuses to engage with the question or "
    "the onboarding (stronger than skip_intent).\n"
    "\n"
    "Rules:\n"
    "- 'partial' is ONLY valid when there are 2 or more target slots. With one "
    "target use 'complete' or 'insufficient'.\n"
    "- If extraction already succeeded for all target slots, default to "
    "'complete' unless the user explicitly skipped/refused.\n"
    "- 'refers_to_slots' must be a subset of the target slot list. When the "
    "user skipped/refused a specific slot, list it there; for 'partial', list "
    "the slots still missing.\n"
    "- Keep 'reasoning' to one short sentence; do not invent details."
)


def _build_messages(
    *,
    schema: type[OnboardingSchema],
    target_slots: tuple[str, ...],
    user_text: str,
    extracted: dict[str, Any] | None,
    extraction_error: str | None,
) -> list[Any]:
    field_lines = []
    for s in target_slots:
        info = schema.model_fields[s]
        req = "required" if info.is_required() else "optional"
        desc = info.description or ""
        field_lines.append(f"- {s} ({req}): {desc}")

    if extracted is not None:
        extracted_block = (
            "\n".join(f"- {s}: {extracted.get(s, '<not extracted>')!r}" for s in target_slots)
            if target_slots
            else "<no targets>"
        )
    else:
        extracted_block = "<extraction not run or failed>"

    human = (
        "Target slot(s):\n"
        + "\n".join(field_lines)
        + "\n\nUser said:\n"
        + user_text
        + "\n\nExtraction result:\n"
        + extracted_block
    )
    if extraction_error:
        human += f"\n\nExtraction error from previous step:\n{extraction_error}"

    return [
        system_msg(_SYSTEM_PROMPT),
        user_msg(human),
    ]


async def judge_response(
    *,
    schema: type[OnboardingSchema],
    target_slots: tuple[str, ...],
    user_text: str,
    extracted: dict[str, Any] | None,
    extraction_error: str | None,
    llm: Any,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> JudgementResult:
    """Classify how the user responded to the target slot question.

    Returns a ``JudgementResult``. Never raises: on repeated LLM failure,
    returns an ``INSUFFICIENT`` verdict so the runner can ask a clarifying
    follow-up.

    ``llm`` accepts any LangChain ``BaseChatModel`` (auto-wrapped) or any
    object implementing the ``LLM`` Protocol from ``slotflow.llm``.
    """
    llm = _resolve_llm(llm)
    structured = llm.with_structured_output(JudgementResult)
    messages = _build_messages(
        schema=schema,
        target_slots=target_slots,
        user_text=user_text,
        extracted=extracted,
        extraction_error=extraction_error,
    )

    for attempt in range(1, max_attempts + 1):
        try:
            raw = await structured.ainvoke(messages)
        except (ParseError, ValidationError) as exc:
            logger.debug(
                "judge attempt %d/%d failed (parse): %s",
                attempt,
                max_attempts,
                exc,
            )
            continue
        try:
            return raw if isinstance(raw, JudgementResult) else JudgementResult.model_validate(raw)
        except ValidationError as exc:
            logger.debug(
                "judge attempt %d/%d failed (post-validation): %s",
                attempt,
                max_attempts,
                exc,
            )
            continue

    logger.warning(
        "judge exhausted %d attempts for %s; defaulting to INSUFFICIENT",
        max_attempts,
        target_slots,
    )
    return JudgementResult(
        verdict=ResponseJudgement.INSUFFICIENT,
        reasoning="Judge LLM failed; defaulting to INSUFFICIENT to keep the conversation moving.",
        refers_to_slots=target_slots,
    )
