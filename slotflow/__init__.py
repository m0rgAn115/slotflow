from typing import Any

from slotflow.flow import (
    FlowMode,
    OnboardingFlow,
    SlotPrompt,
    Step,
)
from slotflow.llm import LLM, ChatMessage, ParseError, from_langchain, from_openai
from slotflow.schema import OnboardingSchema
from slotflow.slot import Slot

__all__ = [
    # Schema layer
    "OnboardingSchema",
    "Slot",
    # Flow layer
    "OnboardingFlow",
    "FlowMode",
    "SlotPrompt",
    "Step",
    # LLM abstraction
    "LLM",
    "ChatMessage",
    "ParseError",
    "from_langchain",
    "from_openai",
    # Runner + extractor (lazy-loaded)
    "extract_slot",
    "extract_slots",
    "ExtractionResult",
    "ExtractionError",
    "FlowState",
    "FlowTurn",
    "Turn",
    "ResponseJudgement",
    "initial_state",
    "next_message",
    "process_response",
]

_EXTRACTOR_EXPORTS = {
    "extract_slot",
    "extract_slots",
    "ExtractionResult",
    "ExtractionError",
}

_RUNNER_EXPORTS = {
    "FlowState",
    "FlowTurn",
    "Turn",
    "initial_state",
    "next_message",
    "process_response",
}

_JUDGE_EXPORTS = {
    "ResponseJudgement",
}


def __getattr__(name: str) -> Any:
    if name in _EXTRACTOR_EXPORTS:
        from slotflow import extractor

        return getattr(extractor, name)
    if name in _RUNNER_EXPORTS:
        from slotflow import runner

        return getattr(runner, name)
    if name in _JUDGE_EXPORTS:
        from slotflow import judge

        return getattr(judge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
