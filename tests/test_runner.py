"""Orchestration tests for the runner. Uses scripted FakeStructuredLLM."""

from datetime import date
from enum import Enum

from fakes import FakeStructuredLLM

from slotflow import (
    FlowMode,
    OnboardingFlow,
    OnboardingSchema,
    ResponseJudgement,
    Slot,
    SlotPrompt,
    Step,
    initial_state,
    next_message,
    process_response,
)
from slotflow.judge import JudgementResult
from slotflow.question_generator import GeneratedQuestion


class DocumentType(str, Enum):
    DNI = "DNI"
    CE = "CE"
    PASSPORT = "Passport"


class UserOnboarding(OnboardingSchema):
    full_name: str = Slot(description="Nombre completo")
    document_type: DocumentType = Slot(description="Tipo de documento")
    birth_date: date = Slot(description="Fecha de nacimiento")
    phone: str | None = Slot(default=None, description="Teléfono")


def _verdict(v: ResponseJudgement, refers=()):
    return JudgementResult(verdict=v, reasoning="", refers_to_slots=refers)


def _q(text: str) -> GeneratedQuestion:
    return GeneratedQuestion(question=text)


# Overrides skip the LLM question-generation path, letting tests focus on
# orchestration behavior rather than asserting on LLM-generated text.
_QUESTION_OVERRIDES = {
    "full_name": SlotPrompt(question="¿Cuál es tu nombre completo?"),
    "document_type": SlotPrompt(question="¿Cuál es tu tipo de documento?"),
    "birth_date": SlotPrompt(question="¿Cuál es tu fecha de nacimiento?"),
    "phone": SlotPrompt(question="¿Cuál es tu teléfono?"),
}


async def test_sequential_golden_path_completes_onboarding():
    flow = OnboardingFlow(
        schema=UserOnboarding,
        mode=FlowMode.SEQUENTIAL,
        prompts=_QUESTION_OVERRIDES,
    )

    llm = FakeStructuredLLM(
        responses={
            JudgementResult: [_verdict(ResponseJudgement.COMPLETE)] * 4,
        },
        default=[
            {"full_name": "Juan Pérez"},
            {"document_type": "DNI"},
            {"birth_date": "1990-05-15"},
            {"phone": "+51 999"},
        ],
    )

    state, turn = await next_message(flow=flow, state=initial_state(flow), llm=llm)
    assert turn.message == "¿Cuál es tu nombre completo?"
    assert turn.target_slots == ("full_name",)

    state, turn = await process_response(
        flow=flow, state=state, user_text="Soy Juan Pérez", llm=llm
    )
    assert state.filled["full_name"] == "Juan Pérez"
    assert turn.message == "¿Cuál es tu tipo de documento?"

    state, turn = await process_response(flow=flow, state=state, user_text="DNI", llm=llm)
    assert state.filled["document_type"] == DocumentType.DNI

    state, turn = await process_response(flow=flow, state=state, user_text="15/05/1990", llm=llm)
    assert state.filled["birth_date"] == date(1990, 5, 15)

    state, turn = await process_response(flow=flow, state=state, user_text="+51 999", llm=llm)
    assert state.filled["phone"] == "+51 999"
    assert turn.done is True
    assert turn.message is None


async def test_freeform_extracts_multiple_slots_from_one_response():
    flow = OnboardingFlow(
        schema=UserOnboarding,
        mode=FlowMode.FREEFORM,
        prompts=_QUESTION_OVERRIDES,
    )

    llm = FakeStructuredLLM(
        responses={
            JudgementResult: [
                _verdict(ResponseJudgement.COMPLETE),
            ],
        },
        default=[
            {
                "full_name": "Juan Pérez",
                "document_type": "DNI",
                "birth_date": "1990-05-15",
                "phone": "+51 999",
            }
        ],
    )

    state, turn = await next_message(flow=flow, state=initial_state(flow), llm=llm)
    assert turn.target_slots == ("full_name", "document_type", "birth_date", "phone")

    state, turn = await process_response(
        flow=flow,
        state=state,
        user_text="Soy Juan Pérez, DNI, 15/05/1990, +51 999",
        llm=llm,
    )
    assert state.filled["full_name"] == "Juan Pérez"
    assert state.filled["document_type"] == DocumentType.DNI
    assert state.filled["birth_date"] == date(1990, 5, 15)
    assert state.filled["phone"] == "+51 999"
    assert turn.done is True


async def test_freeform_partial_response_generates_followup_for_missing_slot():
    flow = OnboardingFlow(
        schema=UserOnboarding,
        mode=FlowMode.FREEFORM,
        prompts=_QUESTION_OVERRIDES,
    )

    llm = FakeStructuredLLM(
        responses={
            JudgementResult: [
                _verdict(ResponseJudgement.PARTIAL, refers=("phone",)),
                _verdict(ResponseJudgement.COMPLETE),
            ],
            GeneratedQuestion: [_q("¿Y tu teléfono?")],
        },
        default=[
            # First turn: user gave name, document, birth_date but not phone.
            {
                "full_name": "Juan",
                "document_type": "DNI",
                "birth_date": "1990-05-15",
                "phone": None,
            },
            # Single-slot follow-up extraction for phone.
            {"phone": "+51 999"},
        ],
    )

    state, _ = await next_message(flow=flow, state=initial_state(flow), llm=llm)

    state, turn = await process_response(
        flow=flow,
        state=state,
        user_text="Soy Juan con DNI, nací el 15/05/1990",
        llm=llm,
    )
    assert state.filled["full_name"] == "Juan"
    assert state.filled["document_type"] == DocumentType.DNI
    assert state.filled["birth_date"] == date(1990, 5, 15)
    assert "phone" not in state.filled
    assert turn.target_slots == ("phone",)
    assert turn.judgement is ResponseJudgement.PARTIAL
    assert state.pending_targets == ("phone",)
    assert turn.message == "¿Y tu teléfono?"

    state, turn = await process_response(flow=flow, state=state, user_text="+51 999", llm=llm)
    assert state.filled["phone"] == "+51 999"
    assert turn.done is True


async def test_required_slot_with_insufficient_response_triggers_followup():
    flow = OnboardingFlow(
        schema=UserOnboarding,
        mode=FlowMode.SEQUENTIAL,
        prompts={
            "full_name": SlotPrompt(question="¿Cuál es tu nombre?"),
            "document_type": SlotPrompt(question="¿Documento?"),
            "birth_date": SlotPrompt(
                question="¿Tu fecha de nacimiento?",
                follow_up_hint="Pide una fecha concreta (día/mes/año).",
            ),
            "phone": SlotPrompt(question="¿Teléfono?"),
        },
    )

    llm = FakeStructuredLLM(
        responses={
            JudgementResult: [
                _verdict(ResponseJudgement.COMPLETE),  # full_name
                _verdict(ResponseJudgement.COMPLETE),  # document_type
                _verdict(ResponseJudgement.INSUFFICIENT),  # birth_date 1st attempt
                _verdict(ResponseJudgement.COMPLETE),  # birth_date 2nd attempt
                _verdict(ResponseJudgement.COMPLETE),  # phone
            ],
            GeneratedQuestion: [
                _q("¿Podrías darme una fecha concreta?"),
            ],
        },
        default=[
            {"full_name": "Juan"},
            {"document_type": "DNI"},
            # INSUFFICIENT turn: extraction runs but the value will be overridden
            # by the judge's verdict. Provide any Pydantic-valid date.
            {"birth_date": "1900-01-01"},
            {"birth_date": "1990-05-15"},
            {"phone": "+51 999"},
        ],
    )

    state = initial_state(flow)
    state, _ = await next_message(flow=flow, state=state, llm=llm)

    state, _ = await process_response(flow=flow, state=state, user_text="Juan", llm=llm)
    state, _ = await process_response(flow=flow, state=state, user_text="DNI", llm=llm)

    state, turn = await process_response(flow=flow, state=state, user_text="hace mucho", llm=llm)
    assert "birth_date" not in state.filled
    assert state.pending_targets == ("birth_date",)
    assert turn.judgement is ResponseJudgement.INSUFFICIENT
    assert turn.message == "¿Podrías darme una fecha concreta?"

    state, turn = await process_response(
        flow=flow, state=state, user_text="el 15 de mayo de 1990", llm=llm
    )
    assert state.filled["birth_date"] == date(1990, 5, 15)


async def test_optional_slot_with_skip_intent_is_marked_skipped_and_advances():
    flow = OnboardingFlow(
        schema=UserOnboarding,
        mode=FlowMode.SEQUENTIAL,
        prompts=_QUESTION_OVERRIDES,
    )

    llm = FakeStructuredLLM(
        responses={
            JudgementResult: [
                _verdict(ResponseJudgement.COMPLETE),  # full_name
                _verdict(ResponseJudgement.COMPLETE),  # document_type
                _verdict(ResponseJudgement.COMPLETE),  # birth_date
                _verdict(ResponseJudgement.SKIP_INTENT, refers=("phone",)),  # phone skip
            ],
        },
        default=[
            {"full_name": "Juan"},
            {"document_type": "DNI"},
            {"birth_date": "1990-05-15"},
            # Phone turn: extraction runs but the judge will override with SKIP_INTENT.
            {"phone": None},
        ],
    )

    state = initial_state(flow)
    state, _ = await next_message(flow=flow, state=state, llm=llm)
    state, _ = await process_response(flow=flow, state=state, user_text="Juan", llm=llm)
    state, _ = await process_response(flow=flow, state=state, user_text="DNI", llm=llm)
    state, _ = await process_response(flow=flow, state=state, user_text="15/05/1990", llm=llm)

    state, turn = await process_response(
        flow=flow, state=state, user_text="prefiero no decir", llm=llm
    )
    assert "phone" in state.skipped
    assert "phone" not in state.filled
    assert turn.done is True
    assert turn.judgement is ResponseJudgement.SKIP_INTENT


async def test_required_slot_with_skip_intent_triggers_followup_not_skipped():
    flow = OnboardingFlow(
        schema=UserOnboarding,
        mode=FlowMode.SEQUENTIAL,
        prompts=_QUESTION_OVERRIDES,
    )

    llm = FakeStructuredLLM(
        responses={
            JudgementResult: [
                _verdict(ResponseJudgement.SKIP_INTENT, refers=("full_name",)),
            ],
            GeneratedQuestion: [_q("Necesito tu nombre, ¿podrías compartirlo?")],
        },
        # Extraction runs first; the judge will override with SKIP_INTENT and
        # the extracted value will be ignored.
        default=[{"full_name": "prefiero no decir"}],
    )

    state = initial_state(flow)
    state, _ = await next_message(flow=flow, state=state, llm=llm)

    state, turn = await process_response(
        flow=flow, state=state, user_text="prefiero no decir", llm=llm
    )
    assert "full_name" not in state.filled
    assert "full_name" not in state.skipped
    assert state.pending_targets == ("full_name",)
    assert turn.judgement is ResponseJudgement.SKIP_INTENT
    assert turn.message == "Necesito tu nombre, ¿podrías compartirlo?"


async def test_required_refused_generates_nudge_and_stays_on_slot():
    flow = OnboardingFlow(
        schema=UserOnboarding,
        mode=FlowMode.SEQUENTIAL,
        prompts=_QUESTION_OVERRIDES,
    )

    llm = FakeStructuredLLM(
        responses={
            JudgementResult: [
                _verdict(ResponseJudgement.REFUSED, refers=("full_name",)),
            ],
            # Runner calls generate_followup to produce the nudge message.
            GeneratedQuestion: [_q("Necesito tu nombre para continuar, ¿podrías compartirlo?")],
        },
        # Extraction runs first; the judge will override with REFUSED.
        default=[{"full_name": "no quiero"}],
    )

    state = initial_state(flow)
    state, _ = await next_message(flow=flow, state=state, llm=llm)

    state, turn = await process_response(
        flow=flow, state=state, user_text="No quiero hacer este onboarding", llm=llm
    )
    assert turn.done is False
    assert turn.message == "Necesito tu nombre para continuar, ¿podrías compartirlo?"
    assert turn.judgement is ResponseJudgement.REFUSED
    assert state.pending_targets == ("full_name",)


async def test_judge_llm_failure_defaults_to_insufficient():
    flow = OnboardingFlow(
        schema=UserOnboarding,
        mode=FlowMode.SEQUENTIAL,
        prompts=_QUESTION_OVERRIDES,
    )

    # Judge call returns a malformed dict twice (max_attempts=2 in judge.py),
    # so judge falls back to INSUFFICIENT. Then a follow-up question is generated.
    llm = FakeStructuredLLM(
        responses={
            JudgementResult: [
                {"oops": "bad"},
                {"oops": "still bad"},
            ],
            GeneratedQuestion: [_q("¿Podrías repetirme tu nombre?")],
        },
        # Extraction runs first; its value is irrelevant when judge defaults
        # to INSUFFICIENT.
        default=[{"full_name": "hola"}],
    )

    state = initial_state(flow)
    state, _ = await next_message(flow=flow, state=state, llm=llm)

    state, turn = await process_response(flow=flow, state=state, user_text="hola", llm=llm)
    assert turn.judgement is ResponseJudgement.INSUFFICIENT
    assert turn.message == "¿Podrías repetirme tu nombre?"
    assert state.pending_targets == ("full_name",)


async def test_input_state_is_not_mutated_between_calls():
    flow = OnboardingFlow(
        schema=UserOnboarding,
        mode=FlowMode.SEQUENTIAL,
        prompts=_QUESTION_OVERRIDES,
    )

    llm = FakeStructuredLLM(
        responses={
            JudgementResult: [_verdict(ResponseJudgement.COMPLETE)],
        },
        default=[{"full_name": "Juan"}],
    )

    state = initial_state(flow)
    state, _ = await next_message(flow=flow, state=state, llm=llm)
    history_len_before = len(state.history)
    filled_before = dict(state.filled)

    new_state, _ = await process_response(flow=flow, state=state, user_text="Juan", llm=llm)

    # Original state is unchanged
    assert len(state.history) == history_len_before
    assert dict(state.filled) == filled_before
    # New state moved forward
    assert "full_name" in new_state.filled
    assert len(new_state.history) > history_len_before


async def test_steps_mode_walks_each_step_in_order():
    flow = OnboardingFlow(
        schema=UserOnboarding,
        mode=FlowMode.STEPS,
        prompts=_QUESTION_OVERRIDES,
        steps=(
            Step(slots=("full_name", "document_type"), name="identidad"),
            Step(slots=("birth_date", "phone"), name="adicional"),
        ),
    )

    llm = FakeStructuredLLM(
        responses={
            JudgementResult: [_verdict(ResponseJudgement.COMPLETE)] * 4,
        },
        default=[
            {"full_name": "Juan"},
            {"document_type": "DNI"},
            {"birth_date": "1990-05-15"},
            {"phone": "+51 999"},
        ],
    )

    state = initial_state(flow)
    state, turn = await next_message(flow=flow, state=state, llm=llm)
    assert turn.target_slots == ("full_name",)

    state, turn = await process_response(flow=flow, state=state, user_text="Juan", llm=llm)
    assert turn.target_slots == ("document_type",)
    state, turn = await process_response(flow=flow, state=state, user_text="DNI", llm=llm)
    assert turn.target_slots == ("birth_date",)
    state, turn = await process_response(flow=flow, state=state, user_text="15/05/1990", llm=llm)
    assert turn.target_slots == ("phone",)
    state, turn = await process_response(flow=flow, state=state, user_text="+51 999", llm=llm)
    assert turn.done is True


async def test_question_override_skips_llm_generation():
    flow = OnboardingFlow(
        schema=UserOnboarding,
        mode=FlowMode.SEQUENTIAL,
        prompts={"full_name": SlotPrompt(question="Override: dime tu nombre")},
    )

    # GeneratedQuestion is intentionally missing from the script — if the runner
    # tried to call the LLM for the overridden slot, the fake would raise.
    llm = FakeStructuredLLM(
        responses={
            GeneratedQuestion: [],
            JudgementResult: [],
        },
    )

    state = initial_state(flow)
    state, turn = await next_message(flow=flow, state=state, llm=llm)
    assert turn.message == "Override: dime tu nombre"
    assert turn.target_slots == ("full_name",)
