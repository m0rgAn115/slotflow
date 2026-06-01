"""Interactive FREEFORM onboarding.

One open-ended prompt lists everything missing; the user can answer any
subset, and the bot follows up on what's still missing.

Run with: ``.venv/bin/python examples/03_freeform_flow.py``
"""

from __future__ import annotations

import asyncio
import os
from datetime import date
from enum import Enum

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI

from slotflow import (
    FlowMode,
    OnboardingFlow,
    OnboardingSchema,
    Slot,
    initial_state,
    next_message,
    process_response,
)


class DocumentType(str, Enum):
    DNI = "DNI"
    CE = "CE"
    PASSPORT = "Passport"


class UserOnboarding(OnboardingSchema):
    full_name: str = Slot(description="Full name")
    document_type: DocumentType = Slot(description="Type of identity document")
    birth_date: date = Slot(description="Date of birth")
    phone: str | None = Slot(default=None, description="Phone number")


async def main() -> None:
    load_dotenv(find_dotenv(usecwd=True))
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing — copy .env.example to .env and fill it in.")

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    flow = OnboardingFlow(
        schema=UserOnboarding,
        mode=FlowMode.FREEFORM,
        # Default intro is "Tell me: {items}." — override per locale if needed:
        # freeform_intro="Erzählen Sie mir: {items}.",
    )

    state = initial_state(flow)
    state, turn = await next_message(flow=flow, state=state, llm=llm)

    while not turn.done:
        if turn.message is None:
            print(f"\nAssistant > (no message, verdict: {turn.judgement})")
            break
        print(f"\nAssistant > {turn.message}")

        user_text = input("You      > ").strip()
        if not user_text:
            print("(empty response; exiting demo)")
            return

        state, turn = await process_response(flow=flow, state=state, user_text=user_text, llm=llm)
        if turn.filled_this_turn:
            print(f"  · captured: {dict(turn.filled_this_turn)}")

    print("\nFinal:")
    print(f"  filled  = {dict(state.filled)}")
    print(f"  skipped = {list(state.skipped)}")


if __name__ == "__main__":
    asyncio.run(main())
