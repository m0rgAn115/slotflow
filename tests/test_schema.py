from datetime import date
from enum import Enum

import pytest
from pydantic import ValidationError

from slotflow import OnboardingSchema, Slot


class DocumentType(str, Enum):
    DNI = "DNI"
    CE = "CE"
    PASSPORT = "Passport"


class Address(OnboardingSchema):
    street: str = Slot(description="Calle y número")
    city: str = Slot(description="Ciudad")


class UserOnboarding(OnboardingSchema):
    full_name: str = Slot(
        description="Nombre completo del usuario",
        examples=["Juan Pérez", "María García"],
    )
    document_type: DocumentType = Slot(description="Tipo de documento")
    birth_date: date = Slot(description="Fecha de nacimiento")
    phone: str | None = Slot(default=None, description="Teléfono de contacto")
    address: Address | None = Slot(default=None, description="Dirección")


def test_required_and_optional_slots_appear_correctly_in_schema():
    schema = UserOnboarding.model_json_schema()

    assert set(schema["required"]) == {"full_name", "document_type", "birth_date"}
    assert "phone" not in schema["required"]
    assert "address" not in schema["required"]


def test_description_is_exposed_in_schema():
    schema = UserOnboarding.model_json_schema()
    assert schema["properties"]["full_name"]["description"] == "Nombre completo del usuario"


def test_examples_are_exposed_when_provided():
    schema = UserOnboarding.model_json_schema()
    assert schema["properties"]["full_name"]["examples"] == ["Juan Pérez", "María García"]


def test_examples_are_omitted_when_not_provided():
    schema = UserOnboarding.model_json_schema()
    assert "examples" not in schema["properties"]["document_type"]


def test_enum_slot_is_exposed_as_enum_in_schema():
    schema = UserOnboarding.model_json_schema()
    doc_ref = schema["properties"]["document_type"]["$ref"]
    enum_name = doc_ref.rsplit("/", 1)[-1]
    assert schema["$defs"][enum_name]["enum"] == ["DNI", "CE", "Passport"]


def test_nested_schema_is_supported():
    instance = UserOnboarding(
        full_name="Juan Pérez",
        document_type=DocumentType.DNI,
        birth_date=date(1990, 1, 1),
        address={"street": "Av. Siempre Viva 123", "city": "Lima"},
    )
    assert isinstance(instance.address, Address)
    assert instance.address.city == "Lima"


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        UserOnboarding(
            full_name="Juan Pérez",
            document_type=DocumentType.DNI,
            birth_date=date(1990, 1, 1),
            hallucinated_field="oops",
        )


def test_slot_metadata_is_preserved_for_future_use():
    schema = UserOnboarding.model_json_schema()
    full_name_props = schema["properties"]["full_name"]
    assert full_name_props.get("_slot_meta") == {"examples": ["Juan Pérez", "María García"]}
    assert schema["properties"]["document_type"].get("_slot_meta") == {}
