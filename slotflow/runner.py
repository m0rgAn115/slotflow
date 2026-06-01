"""Async orchestration of an ``OnboardingFlow``.

This module wires together extraction (``extract_slot`` / ``extract_slots``),
classification (``judge_response``) and natural-language generation
(``generate_question`` / ``generate_followup``) without holding any prompts of
its own. State transitions are pure: every call returns a new ``FlowState``
and the input state is never mutated.

Per-turn LLM call order
-----------------------

Every turn runs extraction first, then the judge with the extraction result.
This is more robust than judging in advance — the judge can confirm that a
Pydantic-valid value is also semantically usable, or override extraction when
the user's intent was actually to skip (e.g. a refusal phrase that extraction
took literally). It also gives the judge more signal: it sees both the raw
text and what extraction produced.

UNORDERED note: at the extraction layer, UNORDERED uses ``extract_slots`` over
all pending slots — the same call FREEFORM makes. The only difference is the
question phrasing (one-at-a-time vs open-ended). This mirrors how a real
onboarding feels: a single visible prompt, but the user can drop in extra info
and the system will pick it up.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .extractor import extract_slot, extract_slots
from .flow import FlowMode, OnboardingFlow
from .judge import ResponseJudgement, judge_response
from .llm import _resolve_llm
from .question_generator import generate_followup, generate_question


@dataclass(frozen=True)
class Turn:
    """One entry in the conversation history.

    Attributes
    ----------
    role:
        ``"assistant"`` for messages the library produced, ``"user"`` for
        responses fed via ``process_response``.
    content:
        The literal text of the message.
    slot:
        For assistant turns, the slot the question is about (``None`` for
        FREEFORM intros). For user turns, the target slot the response was
        attributed to (``None`` in multi-target turns).
    judgement:
        Set only on user turns; the verdict the judge returned.
    """

    role: Literal["assistant", "user"]
    content: str
    slot: str | None = None
    judgement: ResponseJudgement | None = None


@dataclass(frozen=True)
class FlowState:
    """Immutable snapshot of an onboarding conversation.

    Every call to ``next_message`` / ``process_response`` returns a new
    ``FlowState``; the input is never mutated. ``FlowState`` is trivially
    picklable so it can be persisted between turns (Redis, DB, queue).

    Attributes
    ----------
    filled:
        Validated slot values, keyed by slot name.
    skipped:
        Tuple of optional slots the user explicitly declined. A tuple (not a
        set) so it JSON-serializes without a custom encoder.
    history:
        Full conversation transcript as a tuple of ``Turn``\\ s.
    pending_targets:
        Slot(s) the last assistant message asked about. When non-empty it
        overrides mode-based target selection so a follow-up sticks to the
        same slot regardless of mode.
    last_judgement:
        Verdict from the most recent user turn, or ``None`` before the first
        response.
    """

    filled: Mapping[str, Any] = field(default_factory=dict)
    skipped: tuple[str, ...] = ()
    history: tuple[Turn, ...] = ()
    pending_targets: tuple[str, ...] = ()
    last_judgement: ResponseJudgement | None = None


@dataclass(frozen=True)
class FlowTurn:
    """The assistant's side of a single conversation turn.

    Attributes
    ----------
    message:
        The text to show the user, or ``None`` when the onboarding is
        complete (``done=True``).
    target_slots:
        Slot(s) the message is asking about. Empty tuple when ``done`` is
        ``True``.
    done:
        ``True`` when every required slot is filled and every optional is
        either filled or skipped.
    filled_this_turn:
        Slot values captured in *this* turn (a subset of ``state.filled``).
    skipped_this_turn:
        Slot names skipped in *this* turn (a subset of ``state.skipped``).
    judgement:
        Verdict the judge returned for the user's input this turn (``None`` on
        the very first ``next_message`` call before any user input exists).
    """

    message: str | None
    target_slots: tuple[str, ...]
    done: bool
    filled_this_turn: Mapping[str, Any] = field(default_factory=dict)
    skipped_this_turn: tuple[str, ...] = ()
    judgement: ResponseJudgement | None = None


def initial_state(flow: OnboardingFlow) -> FlowState:
    """Return an empty ``FlowState`` for the given flow.

    Today this just returns ``FlowState()`` — the ``flow`` argument is
    accepted for API symmetry and so that future flow-specific initialization
    (e.g. seeding ``skipped`` from a profile) can land without a breaking
    change.
    """
    del flow  # accepted for API symmetry; no flow-specific initialization today
    return FlowState()


def _get_followup_hint(flow: OnboardingFlow, slot_name: str) -> str | None:
    prompt = flow.prompts.get(slot_name)
    return prompt.follow_up_hint if prompt else None


async def _render_question(*, flow: OnboardingFlow, slot_name: str, llm: Any) -> str:
    """Use the SlotPrompt override if present, else LLM-generate and cache."""
    prompt = flow.prompts.get(slot_name)
    if prompt and prompt.question:
        return prompt.question
    cached = flow._question_cache.get(slot_name)
    if cached is not None:
        return cached
    q = await generate_question(schema=flow.schema, slot_name=slot_name, llm=llm)
    flow._question_cache[slot_name] = q
    return q


def _render_freeform_intro(*, flow: OnboardingFlow, targets: tuple[str, ...]) -> str:
    """Build an open-ended message listing everything still missing.

    For each target we prefer the ``SlotPrompt.question`` override, falling
    back to the slot description. No LLM call is made — the FREEFORM intro is
    purely a composition of slot-level texts. (LLM cost is paid per slot at
    extraction time, not in question generation here.)

    The template comes from ``flow.freeform_intro``; ``{items}`` is replaced
    with the joined descriptions.
    """
    descs: list[str] = []
    for s in targets:
        info = flow.schema.model_fields[s]
        prompt = flow.prompts.get(s)
        if prompt and prompt.question:
            descs.append(prompt.question)
        else:
            descs.append(info.description or s)
    return flow.freeform_intro.format(items="; ".join(descs))


async def next_message(
    *, flow: OnboardingFlow, state: FlowState, llm: Any
) -> tuple[FlowState, FlowTurn]:
    """Return ``(new_state, next_turn)`` for the first / next assistant message.

    Appends the assistant message to ``state.history`` so the transcript stays
    consistent with what ``process_response`` records. The input state is never
    mutated — a new state is returned.

    Call once after ``initial_state`` to get the opening question, then call
    ``process_response`` for each user reply.

    ``llm`` accepts any LangChain ``BaseChatModel`` (auto-wrapped) or any
    object implementing the ``LLM`` Protocol from ``slotflow.llm``.

    Returns
    -------
    tuple[FlowState, FlowTurn]
        Updated state and the turn to show. When ``turn.done`` is ``True``,
        ``turn.message`` is ``None`` and the onboarding is complete.
    """
    llm = _resolve_llm(llm)
    targets = flow.targets_for(state)
    if not targets:
        return state, FlowTurn(message=None, target_slots=(), done=True)

    if flow.mode is FlowMode.FREEFORM and not state.pending_targets:
        message = _render_freeform_intro(flow=flow, targets=targets)
        history_slot: str | None = None
    else:
        message = await _render_question(flow=flow, slot_name=targets[0], llm=llm)
        history_slot = targets[0]

    new_state = replace(
        state,
        history=state.history + (Turn(role="assistant", content=message, slot=history_slot),),
    )
    return new_state, FlowTurn(message=message, target_slots=targets, done=False)


async def process_response(
    *,
    flow: OnboardingFlow,
    state: FlowState,
    user_text: str,
    llm: Any,
) -> tuple[FlowState, FlowTurn]:
    """Process the user's response. Returns ``(new_state, next_turn)``.

    The returned ``next_turn.message`` is the question (or follow-up) to show
    next, or ``None`` only when the onboarding is fully complete.

    When a required slot is ``REFUSED``, the library generates a polite nudge
    asking once more. The caller can detect this via ``turn.judgement is
    ResponseJudgement.REFUSED`` and choose to override the message or escalate.

    ``llm`` accepts any LangChain ``BaseChatModel`` (auto-wrapped) or any
    object implementing the ``LLM`` Protocol from ``slotflow.llm``.

    Parameters
    ----------
    flow:
        The same ``OnboardingFlow`` used for the prior ``next_message`` call.
    state:
        The current ``FlowState`` from the previous turn.
    user_text:
        The user's raw text.
    llm:
        A LangChain ``BaseChatModel``, an adapter from ``slotflow.llm``, or
        any object implementing the ``LLM`` Protocol.

    Returns
    -------
    tuple[FlowState, FlowTurn]
        Updated state and the next turn to render.
    """
    llm = _resolve_llm(llm)
    targets = flow.targets_for(state)
    if not targets:
        new_state = replace(
            state,
            history=state.history + (Turn(role="user", content=user_text),),
        )
        return new_state, FlowTurn(message=None, target_slots=(), done=True)

    if len(targets) == 1:
        return await _process_single_target(
            flow=flow,
            state=state,
            user_text=user_text,
            target=targets[0],
            llm=llm,
        )
    return await _process_multi_target(
        flow=flow, state=state, user_text=user_text, targets=targets, llm=llm
    )


async def _process_single_target(
    *,
    flow: OnboardingFlow,
    state: FlowState,
    user_text: str,
    target: str,
    llm: Any,
) -> tuple[FlowState, FlowTurn]:
    """Run one turn against a single target slot.

    Order: extract first, then judge with the extraction result. The verdict
    decides the dispatch; this function stays a thin coordinator and delegates
    each branch to a focused helper below.
    """
    extraction = await extract_slot(
        schema=flow.schema,
        field_name=target,
        text=user_text,
        llm=llm,
    )
    extracted_dict: dict[str, Any] = {}
    if extraction.success and extraction.value is not None:
        extracted_dict[target] = extraction.value

    judgement = await judge_response(
        schema=flow.schema,
        target_slots=(target,),
        user_text=user_text,
        extracted=extracted_dict or None,
        extraction_error=extraction.error,
        llm=llm,
    )
    history_with_user = state.history + (
        Turn(
            role="user",
            content=user_text,
            slot=target,
            judgement=judgement.verdict,
        ),
    )
    is_required = flow.is_required(target)

    if judgement.verdict is ResponseJudgement.SKIP_INTENT:
        return await _handle_skip_intent_single(
            flow=flow,
            state=state,
            history_with_user=history_with_user,
            target=target,
            user_text=user_text,
            is_required=is_required,
            llm=llm,
        )

    if judgement.verdict is ResponseJudgement.REFUSED:
        return await _handle_refused_single(
            flow=flow,
            state=state,
            history_with_user=history_with_user,
            target=target,
            user_text=user_text,
            is_required=is_required,
            llm=llm,
        )

    extraction_unusable = not extraction.success or extraction.value is None
    if judgement.verdict is ResponseJudgement.INSUFFICIENT or (extraction_unusable and is_required):
        return await _emit_followup(
            flow=flow,
            state=state,
            history_with_user=history_with_user,
            target=target,
            user_text=user_text,
            verdict=ResponseJudgement.INSUFFICIENT,
            llm=llm,
        )

    if extraction_unusable:
        # Optional slot, extraction returned None and judge said COMPLETE.
        # Treat as skipped so we don't loop on the same slot forever.
        return await _advance_after_skip(
            flow=flow,
            state=state,
            history_with_user=history_with_user,
            target=target,
            verdict=ResponseJudgement.SKIP_INTENT,
            llm=llm,
        )

    return await _finalize_complete_single(
        flow=flow,
        state=state,
        history_with_user=history_with_user,
        target=target,
        value=extraction.value,
        llm=llm,
    )


async def _handle_skip_intent_single(
    *,
    flow: OnboardingFlow,
    state: FlowState,
    history_with_user: tuple[Turn, ...],
    target: str,
    user_text: str,
    is_required: bool,
    llm: Any,
) -> tuple[FlowState, FlowTurn]:
    """A skip on a required slot becomes a follow-up; on an optional slot it advances."""
    if is_required:
        return await _emit_followup(
            flow=flow,
            state=state,
            history_with_user=history_with_user,
            target=target,
            user_text=user_text,
            verdict=ResponseJudgement.SKIP_INTENT,
            llm=llm,
        )
    return await _advance_after_skip(
        flow=flow,
        state=state,
        history_with_user=history_with_user,
        target=target,
        verdict=ResponseJudgement.SKIP_INTENT,
        llm=llm,
    )


async def _handle_refused_single(
    *,
    flow: OnboardingFlow,
    state: FlowState,
    history_with_user: tuple[Turn, ...],
    target: str,
    user_text: str,
    is_required: bool,
    llm: Any,
) -> tuple[FlowState, FlowTurn]:
    """Handle a REFUSED verdict for a single target slot.

    For an optional slot, treat it like a skip and advance.
    For a required slot, generate a polite nudge and stay on the slot.
    The caller can detect this via ``turn.judgement is ResponseJudgement.REFUSED``
    and choose to replace or override the message.
    """
    if is_required:
        custom_hint = _get_followup_hint(flow, target)
        refuse_hint = (
            f"{custom_hint} " if custom_hint else ""
        ) + "The user declined to provide this required field. Politely explain it is needed and ask once more."
        nudge = await generate_followup(
            schema=flow.schema,
            slot_name=target,
            user_text=user_text,
            follow_up_hint=refuse_hint,
            llm=llm,
        )
        new_state = replace(
            state,
            history=history_with_user + (Turn(role="assistant", content=nudge, slot=target),),
            pending_targets=(target,),
            last_judgement=ResponseJudgement.REFUSED,
        )
        return new_state, FlowTurn(
            message=nudge,
            target_slots=(target,),
            done=False,
            judgement=ResponseJudgement.REFUSED,
        )
    return await _advance_after_skip(
        flow=flow,
        state=state,
        history_with_user=history_with_user,
        target=target,
        verdict=ResponseJudgement.REFUSED,
        llm=llm,
    )


async def _finalize_complete_single(
    *,
    flow: OnboardingFlow,
    state: FlowState,
    history_with_user: tuple[Turn, ...],
    target: str,
    value: Any,
    llm: Any,
) -> tuple[FlowState, FlowTurn]:
    """Fill the slot and advance to the next target (or finish)."""
    new_filled = dict(state.filled)
    new_filled[target] = value
    next_state_base = replace(
        state,
        filled=new_filled,
        history=history_with_user,
        pending_targets=(),
        last_judgement=ResponseJudgement.COMPLETE,
    )
    return await _advance_or_finish(
        flow=flow,
        state=next_state_base,
        skipped_this_turn=(),
        filled_this_turn={target: value},
        verdict=ResponseJudgement.COMPLETE,
        llm=llm,
    )


async def _process_multi_target(
    *,
    flow: OnboardingFlow,
    state: FlowState,
    user_text: str,
    targets: tuple[str, ...],
    llm: Any,
) -> tuple[FlowState, FlowTurn]:
    """Run one turn against multiple target slots (FREEFORM / UNORDERED)."""
    extraction = await extract_slots(
        schema=flow.schema,
        field_names=list(targets),
        text=user_text,
        llm=llm,
    )
    extracted_dict: dict[str, Any] = {}
    if extraction.success and extraction.value:
        for s in targets:
            v = extraction.value.get(s)
            if v is None:
                continue
            extracted_dict[s] = v

    judgement = await judge_response(
        schema=flow.schema,
        target_slots=targets,
        user_text=user_text,
        extracted=extracted_dict,
        extraction_error=extraction.error,
        llm=llm,
    )

    history_with_user = state.history + (
        Turn(role="user", content=user_text, judgement=judgement.verdict),
    )

    new_filled = dict(state.filled)
    new_filled.update(extracted_dict)
    filled_this_turn = dict(extracted_dict)

    skipped_this_turn = _skipped_slots_for_multi(
        flow=flow,
        targets=targets,
        new_filled=new_filled,
        verdict=judgement.verdict,
        refers_to_slots=judgement.refers_to_slots,
    )
    new_skipped = state.skipped + skipped_this_turn

    remaining = tuple(s for s in targets if s not in new_filled and s not in new_skipped)

    next_state_base = replace(
        state,
        filled=new_filled,
        skipped=new_skipped,
        history=history_with_user,
        pending_targets=(),
        last_judgement=judgement.verdict,
    )

    if remaining and judgement.verdict in {
        ResponseJudgement.PARTIAL,
        ResponseJudgement.INSUFFICIENT,
    }:
        return await _emit_followup_multi(
            flow=flow,
            state=next_state_base,
            history_with_user=history_with_user,
            target=remaining[0],
            user_text=user_text,
            filled_this_turn=filled_this_turn,
            skipped_this_turn=skipped_this_turn,
            verdict=judgement.verdict,
            llm=llm,
        )

    return await _advance_or_finish(
        flow=flow,
        state=next_state_base,
        skipped_this_turn=skipped_this_turn,
        filled_this_turn=filled_this_turn,
        verdict=judgement.verdict,
        llm=llm,
    )


def _skipped_slots_for_multi(
    *,
    flow: OnboardingFlow,
    targets: tuple[str, ...],
    new_filled: Mapping[str, Any],
    verdict: ResponseJudgement,
    refers_to_slots: tuple[str, ...],
) -> tuple[str, ...]:
    """Compute which optional slots to mark as skipped on a multi-target turn."""
    if verdict not in {ResponseJudgement.SKIP_INTENT, ResponseJudgement.REFUSED}:
        return ()
    return tuple(
        s
        for s in refers_to_slots
        if s in targets and s not in new_filled and not flow.is_required(s)
    )


async def _emit_followup_multi(
    *,
    flow: OnboardingFlow,
    state: FlowState,
    history_with_user: tuple[Turn, ...],
    target: str,
    user_text: str,
    filled_this_turn: Mapping[str, Any],
    skipped_this_turn: tuple[str, ...],
    verdict: ResponseJudgement,
    llm: Any,
) -> tuple[FlowState, FlowTurn]:
    """Like ``_emit_followup`` but preserves the multi-turn capture metadata."""
    followup = await generate_followup(
        schema=flow.schema,
        slot_name=target,
        user_text=user_text,
        follow_up_hint=_get_followup_hint(flow, target),
        llm=llm,
    )
    new_state = replace(
        state,
        history=history_with_user + (Turn(role="assistant", content=followup, slot=target),),
        pending_targets=(target,),
        last_judgement=ResponseJudgement.INSUFFICIENT,
    )
    return new_state, FlowTurn(
        message=followup,
        target_slots=(target,),
        done=False,
        filled_this_turn=filled_this_turn,
        skipped_this_turn=skipped_this_turn,
        judgement=verdict,
    )


async def _emit_followup(
    *,
    flow: OnboardingFlow,
    state: FlowState,
    history_with_user: tuple[Turn, ...],
    target: str,
    user_text: str,
    verdict: ResponseJudgement,
    llm: Any,
) -> tuple[FlowState, FlowTurn]:
    followup = await generate_followup(
        schema=flow.schema,
        slot_name=target,
        user_text=user_text,
        follow_up_hint=_get_followup_hint(flow, target),
        llm=llm,
    )
    new_state = replace(
        state,
        history=history_with_user + (Turn(role="assistant", content=followup, slot=target),),
        pending_targets=(target,),
        last_judgement=verdict,
    )
    return new_state, FlowTurn(
        message=followup,
        target_slots=(target,),
        done=False,
        judgement=verdict,
    )


async def _advance_after_skip(
    *,
    flow: OnboardingFlow,
    state: FlowState,
    history_with_user: tuple[Turn, ...],
    target: str,
    verdict: ResponseJudgement,
    llm: Any,
) -> tuple[FlowState, FlowTurn]:
    new_skipped = state.skipped + (target,)
    next_state_base = replace(
        state,
        skipped=new_skipped,
        history=history_with_user,
        pending_targets=(),
        last_judgement=verdict,
    )
    return await _advance_or_finish(
        flow=flow,
        state=next_state_base,
        skipped_this_turn=(target,),
        filled_this_turn={},
        verdict=verdict,
        llm=llm,
    )


async def _advance_or_finish(
    *,
    flow: OnboardingFlow,
    state: FlowState,
    skipped_this_turn: tuple[str, ...],
    filled_this_turn: Mapping[str, Any],
    verdict: ResponseJudgement | None,
    llm: Any,
) -> tuple[FlowState, FlowTurn]:
    next_targets = flow.targets_for(state)
    if not next_targets:
        return state, FlowTurn(
            message=None,
            target_slots=(),
            done=True,
            filled_this_turn=filled_this_turn,
            skipped_this_turn=skipped_this_turn,
            judgement=verdict,
        )

    if flow.mode is FlowMode.FREEFORM:
        message = _render_freeform_intro(flow=flow, targets=next_targets)
        history_slot: str | None = None
    else:
        message = await _render_question(flow=flow, slot_name=next_targets[0], llm=llm)
        history_slot = next_targets[0]

    new_state = replace(
        state,
        history=state.history + (Turn(role="assistant", content=message, slot=history_slot),),
    )
    return new_state, FlowTurn(
        message=message,
        target_slots=next_targets,
        done=False,
        filled_this_turn=filled_this_turn,
        skipped_this_turn=skipped_this_turn,
        judgement=verdict,
    )
