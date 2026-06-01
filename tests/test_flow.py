"""Declarative-layer tests for OnboardingFlow. No LLM involved."""

from datetime import date
from enum import Enum

import pytest

from slotflow import (
    FlowMode,
    OnboardingFlow,
    OnboardingSchema,
    Slot,
    SlotPrompt,
    Step,
)
from slotflow.runner import FlowState


class DocumentType(str, Enum):
    DNI = "DNI"
    CE = "CE"
    PASSPORT = "Passport"


class UserOnboarding(OnboardingSchema):
    full_name: str = Slot(description="Nombre completo")
    document_type: DocumentType = Slot(description="Tipo de documento")
    birth_date: date = Slot(description="Fecha de nacimiento")
    phone: str | None = Slot(default=None, description="Teléfono (opcional)")


def test_sequential_targets_first_unfilled_in_declaration_order():
    flow = OnboardingFlow(schema=UserOnboarding, mode=FlowMode.SEQUENTIAL)
    state = FlowState()

    assert flow.targets_for(state) == ("full_name",)


def test_sequential_skips_filled_slots():
    flow = OnboardingFlow(schema=UserOnboarding, mode=FlowMode.SEQUENTIAL)
    state = FlowState(filled={"full_name": "Juan"})

    assert flow.targets_for(state) == ("document_type",)


def test_freeform_targets_all_unfilled_at_once():
    flow = OnboardingFlow(schema=UserOnboarding, mode=FlowMode.FREEFORM)
    state = FlowState(filled={"full_name": "Juan"})

    assert flow.targets_for(state) == ("document_type", "birth_date", "phone")


def test_is_done_requires_required_filled_and_optional_filled_or_skipped():
    flow = OnboardingFlow(schema=UserOnboarding, mode=FlowMode.SEQUENTIAL)

    state = FlowState(
        filled={
            "full_name": "Juan",
            "document_type": DocumentType.DNI,
            "birth_date": date(1990, 1, 1),
        }
    )
    assert flow.is_done(state) is False  # phone still pending

    state_skipped = FlowState(
        filled=state.filled,
        skipped=("phone",),
    )
    assert flow.is_done(state_skipped) is True

    state_filled = FlowState(
        filled={**state.filled, "phone": "+51999"},
    )
    assert flow.is_done(state_filled) is True


def test_pending_targets_override_mode_dispatch():
    flow = OnboardingFlow(schema=UserOnboarding, mode=FlowMode.FREEFORM)
    state = FlowState(pending_targets=("phone",))

    assert flow.targets_for(state) == ("phone",)


def test_prompts_with_unknown_slot_name_rejected():
    with pytest.raises(ValueError, match="unknown slot"):
        OnboardingFlow(
            schema=UserOnboarding,
            mode=FlowMode.SEQUENTIAL,
            prompts={"nonexistent": SlotPrompt(question="?")},
        )


def test_steps_mode_requires_steps():
    with pytest.raises(ValueError, match="STEPS requires non-empty steps"):
        OnboardingFlow(schema=UserOnboarding, mode=FlowMode.STEPS)


def test_steps_mode_rejects_missing_slots():
    with pytest.raises(ValueError, match="missing"):
        OnboardingFlow(
            schema=UserOnboarding,
            mode=FlowMode.STEPS,
            steps=(
                Step(slots=("full_name", "document_type"), name="ids"),
                # birth_date and phone missing
            ),
        )


def test_steps_mode_rejects_duplicates():
    with pytest.raises(ValueError, match="multiple steps"):
        OnboardingFlow(
            schema=UserOnboarding,
            mode=FlowMode.STEPS,
            steps=(
                Step(slots=("full_name", "document_type")),
                Step(slots=("full_name", "birth_date", "phone")),
            ),
        )


def test_steps_mode_rejects_unknown_slot():
    with pytest.raises(ValueError, match="unknown slot"):
        OnboardingFlow(
            schema=UserOnboarding,
            mode=FlowMode.STEPS,
            steps=(Step(slots=("full_name", "document_type", "birth_date", "phone", "ghost")),),
        )


def test_steps_mode_walks_step_by_step():
    flow = OnboardingFlow(
        schema=UserOnboarding,
        mode=FlowMode.STEPS,
        steps=(
            Step(slots=("full_name", "document_type"), name="identidad"),
            Step(slots=("birth_date", "phone"), name="adicional"),
        ),
    )

    assert flow.targets_for(FlowState()) == ("full_name",)

    s1 = FlowState(filled={"full_name": "Juan"})
    assert flow.targets_for(s1) == ("document_type",)

    s2 = FlowState(filled={"full_name": "Juan", "document_type": DocumentType.DNI})
    assert flow.targets_for(s2) == ("birth_date",)


def test_steps_only_allowed_with_steps_mode():
    with pytest.raises(ValueError, match="only valid with FlowMode.STEPS"):
        OnboardingFlow(
            schema=UserOnboarding,
            mode=FlowMode.SEQUENTIAL,
            steps=(Step(slots=("full_name",)),),
        )


def test_is_required_matches_schema_definition():
    flow = OnboardingFlow(schema=UserOnboarding)
    assert flow.is_required("full_name") is True
    assert flow.is_required("phone") is False


def test_targets_empty_when_done():
    flow = OnboardingFlow(schema=UserOnboarding, mode=FlowMode.SEQUENTIAL)
    state = FlowState(
        filled={
            "full_name": "Juan",
            "document_type": DocumentType.DNI,
            "birth_date": date(1990, 1, 1),
            "phone": "+51999",
        },
    )
    assert flow.targets_for(state) == ()
    assert flow.is_done(state) is True
