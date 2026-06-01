"""Declarative composition of slots into a conversational onboarding flow.

This module is intentionally LLM-free: it imports neither langchain nor the
extractor. The flow is a configuration object that describes *how* to ask, and
its validation runs at construction time so any reference to slots / steps is
guaranteed to be consistent before any conversation begins.

Runtime orchestration lives in ``runner.py``, question wording in
``question_generator.py``, and response classification in ``judge.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from .schema import OnboardingSchema

if TYPE_CHECKING:
    from .runner import FlowState


DEFAULT_FREEFORM_INTRO = "Tell me: {items}."


class FlowMode(str, Enum):
    """Conversational shape the flow takes.

    SEQUENTIAL: ask slots one at a time in schema declaration order.
    UNORDERED: ask one at a time but accept answers for any pending slot
        in the same turn (extraction is multi-slot under the hood).
    FREEFORM: open-ended turns; extract many slots from one response.
    STEPS: slots grouped into ordered Step()s; finish a step before moving on.
    """

    SEQUENTIAL = "sequential"
    UNORDERED = "unordered"
    FREEFORM = "freeform"
    STEPS = "steps"


@dataclass(frozen=True)
class SlotPrompt:
    """Per-slot prompt overrides used by OnboardingFlow.

    ``question`` skips LLM question generation entirely. ``follow_up_hint`` is
    fed to the follow-up generator when a response is judged insufficient.
    """

    question: str | None = None
    follow_up_hint: str | None = None


@dataclass(frozen=True)
class Step:
    """A named group of slots used in FlowMode.STEPS.

    Slot order within ``slots`` is the order they will be asked. ``name`` and
    ``intro`` are surfaced via FlowTurn metadata so callers can render section
    headers or transitional messages.
    """

    slots: tuple[str, ...]
    name: str | None = None
    intro: str | None = None


@dataclass
class OnboardingFlow:
    """Composes an ``OnboardingSchema`` with conversational metadata.

    Validation in ``__post_init__`` rejects unknown slot references in
    ``prompts`` and, for ``FlowMode.STEPS``, enforces that the steps partition
    the schema's slots (no duplicates, no missing).

    Parameters
    ----------
    schema:
        The ``OnboardingSchema`` subclass describing the slots to capture.
    mode:
        How the conversation is shaped. Defaults to ``FlowMode.SEQUENTIAL``.
    prompts:
        Optional per-slot overrides. Keys must be slot names on ``schema``.
    steps:
        Required when ``mode is FlowMode.STEPS``; the steps must collectively
        cover every slot in the schema exactly once.
    freeform_intro:
        Template for the FREEFORM opening message. ``{items}`` is replaced with
        the joined slot descriptions / overrides. Defaults to ``"Tell me: {items}."``.

    Notes
    -----
    ``_question_cache`` is private state that memoizes LLM-generated base
    questions. It is intentionally excluded from the public/equality surface.
    """

    schema: type[OnboardingSchema]
    mode: FlowMode = FlowMode.SEQUENTIAL
    prompts: Mapping[str, SlotPrompt] = field(default_factory=dict)
    steps: tuple[Step, ...] = ()
    freeform_intro: str = DEFAULT_FREEFORM_INTRO
    _question_cache: dict[str, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        slot_names = set(self.schema.model_fields)

        unknown_prompts = set(self.prompts) - slot_names
        if unknown_prompts:
            raise ValueError(f"prompts references unknown slot(s): {sorted(unknown_prompts)}")

        if "{items}" not in self.freeform_intro:
            raise ValueError(
                "freeform_intro must contain the '{items}' placeholder "
                "(it is replaced with the joined slot descriptions)."
            )

        if self.mode is FlowMode.STEPS:
            if not self.steps:
                raise ValueError("FlowMode.STEPS requires non-empty steps")
            seen: set[str] = set()
            for step in self.steps:
                if not step.slots:
                    raise ValueError("Step.slots must be non-empty")
                for s in step.slots:
                    if s not in slot_names:
                        raise ValueError(f"Step references unknown slot: {s!r}")
                    if s in seen:
                        raise ValueError(f"Slot {s!r} appears in multiple steps")
                    seen.add(s)
            missing = slot_names - seen
            if missing:
                raise ValueError(f"Steps don't cover all schema slots; missing: {sorted(missing)}")
        elif self.steps:
            raise ValueError(f"steps is only valid with FlowMode.STEPS, not {self.mode!r}")

    def is_required(self, slot_name: str) -> bool:
        return self.schema.model_fields[slot_name].is_required()

    def step_for(self, slot_name: str) -> Step | None:
        if self.mode is not FlowMode.STEPS:
            return None
        for step in self.steps:
            if slot_name in step.slots:
                return step
        return None

    def is_done(self, state: FlowState) -> bool:
        """True when every required slot is filled and every optional is filled or skipped."""
        for name, field_info in self.schema.model_fields.items():
            if name in state.filled:
                continue
            if field_info.is_required():
                return False
            if name not in state.skipped:
                return False
        return True

    def targets_for(self, state: FlowState) -> tuple[str, ...]:
        """Slot(s) to ask next given the mode and what's already filled/skipped.

        Returns an empty tuple if the onboarding is complete. ``pending_targets``
        on the state always takes precedence (so a clarifying follow-up keeps
        the same target regardless of mode).
        """
        if self.is_done(state):
            return ()

        if state.pending_targets:
            return state.pending_targets

        if self.mode is FlowMode.FREEFORM:
            return tuple(
                s
                for s in self.schema.model_fields
                if s not in state.filled and s not in state.skipped
            )

        if self.mode is FlowMode.STEPS:
            for step in self.steps:
                for s in step.slots:
                    if s not in state.filled and s not in state.skipped:
                        return (s,)
            return ()

        for s in self.schema.model_fields:
            if s not in state.filled and s not in state.skipped:
                return (s,)
        return ()
