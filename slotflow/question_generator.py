"""LLM helpers to render slot descriptions as natural conversational questions.

Two entry points:

- ``generate_question`` produces the base question for a slot. The runner
  consults the ``SlotPrompt`` override first; only when there is no override
  does it call this. Results are memoized on the ``OnboardingFlow``'s question
  cache, so a given slot's base question is generated at most once per process.
- ``generate_followup`` produces a clarifying question after a response is
  judged insufficient or partial. It references what the user said and the
  optional ``follow_up_hint`` from the ``SlotPrompt``. Not cached — the wording
  depends on the user's text.

Both functions return a plain string and fall back to the slot's description
if the LLM call fails. They never raise to the caller.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from .llm import ParseError, _resolve_llm, system_msg, user_msg
from .schema import OnboardingSchema

logger = logging.getLogger(__name__)


class GeneratedQuestion(BaseModel):
    """Structured output wrapper for question/follow-up generation."""

    model_config = ConfigDict(extra="forbid")

    question: str


_BASE_SYSTEM_PROMPT = (
    "You write a single short conversational question to gather one specific "
    "piece of information during an onboarding chat. Requirements:\n"
    "- One sentence, friendly, natural.\n"
    "- Use the same language as the slot description.\n"
    "- Do not enumerate examples in the question.\n"
    "- Do not include the slot's internal name or quotes around it."
)


_FOLLOWUP_SYSTEM_PROMPT = (
    "You write a single short follow-up question to help the user complete one "
    "specific piece of information they did not provide clearly. Requirements:\n"
    "- One sentence, friendly.\n"
    "- Briefly reference what the user just said when it helps clarify what is missing.\n"
    "- Use the same language as the user's message and the slot description.\n"
    "- Do not include the slot's internal name."
)


async def generate_question(
    *,
    schema: type[OnboardingSchema],
    slot_name: str,
    llm: Any,
) -> str:
    """Render a slot description as a natural-language question.

    Falls back to the slot's description verbatim if the LLM call fails.

    ``llm`` accepts any LangChain ``BaseChatModel`` (auto-wrapped) or any
    object implementing the ``LLM`` Protocol from ``slotflow.llm``.
    """
    llm = _resolve_llm(llm)
    info = schema.model_fields[slot_name]
    desc = info.description or slot_name
    examples = info.examples or []

    human = f"Slot description:\n{desc}"
    if examples:
        human += (
            f"\n\nExample values (for context only; do not enumerate in the question):\n{examples}"
        )

    structured = llm.with_structured_output(GeneratedQuestion)
    try:
        raw = await structured.ainvoke(
            [
                system_msg(_BASE_SYSTEM_PROMPT),
                user_msg(human),
            ]
        )
        result = (
            raw if isinstance(raw, GeneratedQuestion) else GeneratedQuestion.model_validate(raw)
        )
        return result.question
    except (ParseError, ValidationError) as exc:
        logger.warning(
            "generate_question failed for slot %r on %s; falling back to description. Error: %s",
            slot_name,
            schema.__name__,
            exc,
        )
        return desc


async def generate_followup(
    *,
    schema: type[OnboardingSchema],
    slot_name: str,
    user_text: str,
    follow_up_hint: str | None,
    llm: Any,
) -> str:
    """Render a clarifying follow-up question for an insufficient response.

    ``llm`` accepts any LangChain ``BaseChatModel`` (auto-wrapped) or any
    object implementing the ``LLM`` Protocol from ``slotflow.llm``.
    """
    llm = _resolve_llm(llm)
    info = schema.model_fields[slot_name]
    desc = info.description or slot_name

    human = f"Information we still need:\n{desc}\n\nWhat the user just said:\n{user_text}"
    if follow_up_hint:
        human += f"\n\nExtra guidance for the follow-up:\n{follow_up_hint}"

    structured = llm.with_structured_output(GeneratedQuestion)
    try:
        raw = await structured.ainvoke(
            [
                system_msg(_FOLLOWUP_SYSTEM_PROMPT),
                user_msg(human),
            ]
        )
        result = (
            raw if isinstance(raw, GeneratedQuestion) else GeneratedQuestion.model_validate(raw)
        )
        return result.question
    except (ParseError, ValidationError) as exc:
        logger.warning(
            "generate_followup failed for slot %r on %s; falling back to description. Error: %s",
            slot_name,
            schema.__name__,
            exc,
        )
        return desc
