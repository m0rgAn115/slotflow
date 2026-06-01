"""Base class for onboarding schemas.

Subclass ``OnboardingSchema`` and declare fields with ``Slot(...)`` to describe
what an LLM should extract from a conversation.
"""

from pydantic import BaseModel, ConfigDict


class OnboardingSchema(BaseModel):
    """Base class for declaring a conversational onboarding schema.

    Subclass and declare fields with ``Slot(...)`` to describe what an LLM
    should extract from the conversation.

    Configuration
    -------------
    ``extra="forbid"``
        Rejects unknown fields so hallucinated slots fail validation loudly
        rather than silently going through. Do not weaken this without an
        explicit reason — it is the library's main hallucination guard.

    ``validate_assignment=True``
        Re-validates on attribute assignment so post-construction mutation
        cannot bypass the type system.

    Example
    -------
    >>> from datetime import date
    >>> from slotflow import OnboardingSchema, Slot
    >>> class Profile(OnboardingSchema):
    ...     name: str = Slot(description="Full name")
    ...     birth_date: date = Slot(description="Date of birth")
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
