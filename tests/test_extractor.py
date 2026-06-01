from datetime import date
from enum import Enum

import pytest
from fakes import FakeStructuredLLM

from slotflow import (
    ExtractionError,
    OnboardingSchema,
    Slot,
    extract_slot,
    extract_slots,
)


class DocumentType(str, Enum):
    DNI = "DNI"
    CE = "CE"
    PASSPORT = "Passport"


class UserOnboarding(OnboardingSchema):
    full_name: str = Slot(
        description="Nombre completo del usuario",
        examples=["Juan Pérez", "María García"],
    )
    document_type: DocumentType = Slot(description="Tipo de documento")
    birth_date: date = Slot(description="Fecha de nacimiento")
    phone: str | None = Slot(default=None, description="Teléfono de contacto")


async def test_success_on_first_attempt_with_dict_response():
    llm = FakeStructuredLLM(responses=[{"full_name": "Juan Pérez"}])

    result = await extract_slot(
        schema=UserOnboarding,
        field_name="full_name",
        text="me llamo Juan Pérez",
        llm=llm,
    )

    assert result.success is True
    assert result.value == "Juan Pérez"
    assert result.attempts == 1
    assert result.error is None


async def test_success_on_first_attempt_with_instance_response():
    class CapturingLLM(FakeStructuredLLM):
        def with_structured_output(self, model_cls):
            instance = model_cls(full_name="María García")
            self._responses = [instance]
            return super().with_structured_output(model_cls)

    llm = CapturingLLM(responses=[])

    result = await extract_slot(
        schema=UserOnboarding,
        field_name="full_name",
        text="soy María García",
        llm=llm,
    )

    assert result.success is True
    assert result.value == "María García"
    assert result.attempts == 1


async def test_retry_on_validation_error_then_succeeds():
    llm = FakeStructuredLLM(
        responses=[
            {"document_type": "INVALID"},  # not in enum → ValidationError
            {"document_type": "DNI"},
        ]
    )

    result = await extract_slot(
        schema=UserOnboarding,
        field_name="document_type",
        text="tengo DNI",
        llm=llm,
    )

    assert result.success is True
    assert result.value == DocumentType.DNI
    assert result.attempts == 2
    assert result.error is None
    assert len(llm.invocations) == 2
    retry_text = llm.invocations[1][1].content
    assert "Previous attempt failed" in retry_text or "previous attempt" in retry_text.lower()


async def test_failure_after_exhausting_attempts():
    llm = FakeStructuredLLM(
        responses=[
            {"document_type": "INVALID"},
            {"document_type": "STILL_BAD"},
            {"document_type": "NOPE"},
        ]
    )

    result = await extract_slot(
        schema=UserOnboarding,
        field_name="document_type",
        text="qué documento tengo",
        llm=llm,
        max_attempts=3,
    )

    assert result.success is False
    assert result.value is None
    assert result.attempts == 3
    assert result.error is not None


async def test_unknown_field_name_raises_extraction_error():
    llm = FakeStructuredLLM(responses=[])

    with pytest.raises(ExtractionError):
        await extract_slot(
            schema=UserOnboarding,
            field_name="nonexistent_slot",
            text="anything",
            llm=llm,
        )


async def test_date_slot_is_validated_by_pydantic():
    llm = FakeStructuredLLM(responses=[{"birth_date": "1990-05-15"}])

    result = await extract_slot(
        schema=UserOnboarding,
        field_name="birth_date",
        text="nací el 15 de mayo de 1990",
        llm=llm,
    )

    assert result.success is True
    assert result.value == date(1990, 5, 15)


async def test_optional_slot_accepts_none():
    llm = FakeStructuredLLM(responses=[{"phone": None}])

    result = await extract_slot(
        schema=UserOnboarding,
        field_name="phone",
        text="no doy mi teléfono",
        llm=llm,
    )

    assert result.success is True
    assert result.value is None


async def test_enum_slot_returns_enum_member():
    llm = FakeStructuredLLM(responses=[{"document_type": "Passport"}])

    result = await extract_slot(
        schema=UserOnboarding,
        field_name="document_type",
        text="tengo pasaporte",
        llm=llm,
    )

    assert result.success is True
    assert result.value == DocumentType.PASSPORT


async def test_first_invocation_has_no_retry_feedback():
    llm = FakeStructuredLLM(responses=[{"full_name": "Ana"}])

    await extract_slot(
        schema=UserOnboarding,
        field_name="full_name",
        text="soy Ana",
        llm=llm,
    )

    human_content = llm.invocations[0][1].content
    assert "Previous attempt failed" not in human_content


async def test_wrapper_model_is_passed_to_structured_output():
    llm = FakeStructuredLLM(responses=[{"full_name": "Pedro"}])

    await extract_slot(
        schema=UserOnboarding,
        field_name="full_name",
        text="soy Pedro",
        llm=llm,
    )

    assert llm.last_model_cls is not None
    assert "full_name" in llm.last_model_cls.model_fields
    assert llm.last_model_cls.model_config.get("extra") == "forbid"


async def test_llm_transport_exception_triggers_retry():
    from slotflow.llm import ParseError

    llm = FakeStructuredLLM(
        responses=[
            ParseError("garbled JSON from provider"),
            {"full_name": "Lucía"},
        ]
    )

    result = await extract_slot(
        schema=UserOnboarding,
        field_name="full_name",
        text="soy Lucía",
        llm=llm,
    )

    assert result.success is True
    assert result.value == "Lucía"
    assert result.attempts == 2


async def test_extract_slots_returns_dict_keyed_by_field_names():
    llm = FakeStructuredLLM(
        responses=[
            {
                "full_name": "Juan Pérez",
                "document_type": "DNI",
                "birth_date": "1990-05-15",
            }
        ]
    )

    result = await extract_slots(
        schema=UserOnboarding,
        field_names=["full_name", "document_type", "birth_date"],
        text="me llamo Juan Pérez, DNI, nací el 15 de mayo de 1990",
        llm=llm,
    )

    assert result.success is True
    assert result.value == {
        "full_name": "Juan Pérez",
        "document_type": DocumentType.DNI,
        "birth_date": date(1990, 5, 15),
    }
    assert result.attempts == 1


async def test_extract_slots_retries_whole_batch_on_any_field_invalid():
    llm = FakeStructuredLLM(
        responses=[
            {"full_name": "Ana", "document_type": "INVALID"},
            {"full_name": "Ana", "document_type": "CE"},
        ]
    )

    result = await extract_slots(
        schema=UserOnboarding,
        field_names=["full_name", "document_type"],
        text="soy Ana con carnet de extranjería",
        llm=llm,
    )

    assert result.success is True
    assert result.value == {"full_name": "Ana", "document_type": DocumentType.CE}
    assert result.attempts == 2


async def test_extract_slots_empty_list_raises():
    llm = FakeStructuredLLM(responses=[])

    with pytest.raises(ExtractionError):
        await extract_slots(
            schema=UserOnboarding,
            field_names=[],
            text="anything",
            llm=llm,
        )


async def test_extract_slots_unknown_field_raises():
    llm = FakeStructuredLLM(responses=[])

    with pytest.raises(ExtractionError):
        await extract_slots(
            schema=UserOnboarding,
            field_names=["full_name", "nope"],
            text="anything",
            llm=llm,
        )


async def test_extract_slots_wrapper_contains_all_requested_fields():
    llm = FakeStructuredLLM(responses=[{"full_name": "Z", "phone": None}])

    await extract_slots(
        schema=UserOnboarding,
        field_names=["full_name", "phone"],
        text="Z, sin teléfono",
        llm=llm,
    )

    assert llm.last_model_cls is not None
    assert set(llm.last_model_cls.model_fields.keys()) == {"full_name", "phone"}
