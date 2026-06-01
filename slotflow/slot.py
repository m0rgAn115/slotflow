"""Slot declaration helper. Wraps ``pydantic.Field`` with a small metadata namespace.

A slot is a single piece of information the LLM should extract from the user's
text. Declare slots as class attributes on an ``OnboardingSchema`` subclass.
"""

from typing import Any

from pydantic import Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined


def Slot(
    *,
    description: str,
    examples: list[Any] | None = None,
    default: Any = PydanticUndefined,
    **kwargs: Any,
) -> FieldInfo:
    """Declare a slot to be filled during a conversational onboarding.

    ``description`` is required — it is the primary signal the LLM uses to
    understand what to extract. ``examples`` is optional; when provided it is
    surfaced in the generated JSON Schema so the LLM can mimic the format.

    Slots without a ``default`` are required; slots with one (typically
    ``default=None``) are optional. Whether the onboarding is "complete" is
    decided by the caller, not here.

    Extra keyword arguments are forwarded to ``pydantic.Field`` unchanged
    (e.g. ``ge``, ``le``, ``pattern``, ``min_length``).

    Returns
    -------
    pydantic.fields.FieldInfo
        Use as a class-attribute default on an ``OnboardingSchema``:

        >>> class Profile(OnboardingSchema):
        ...     name: str = Slot(description="Full name")
        ...     age: int | None = Slot(default=None, description="Age in years")
    """
    slot_meta: dict[str, Any] = {}
    if examples is not None:
        slot_meta["examples"] = list(examples)

    json_schema_extra = kwargs.pop("json_schema_extra", None) or {}
    json_schema_extra["_slot_meta"] = slot_meta

    return Field(
        default=default,
        description=description,
        examples=examples,
        json_schema_extra=json_schema_extra,
        **kwargs,
    )
