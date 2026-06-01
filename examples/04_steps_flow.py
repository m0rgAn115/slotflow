"""Interactive STEPS onboarding.

Slots are grouped into named steps walked in order. Each step is internally
sequential; the next step starts only after the current one finishes.

Run with: ``.venv/bin/python examples/04_steps_flow.py``
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
    SlotPrompt,
    Step,
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
        mode=FlowMode.STEPS,
        prompts={
            "phone": SlotPrompt(question="What's your contact phone number? (optional)"),
        },
        steps=(
            Step(slots=("full_name", "document_type"), name="identity"),
            Step(slots=("birth_date", "phone"), name="additional info"),
        ),
    )

    state = initial_state(flow)
    state, turn = await next_message(flow=flow, state=state, llm=llm)
    current_step: str | None = None

    while not turn.done:
        if turn.message is None:
            print(f"\nAssistant > (no message, verdict: {turn.judgement})")
            break

        if turn.target_slots:
            step = flow.step_for(turn.target_slots[0])
            if step and step.name != current_step:
                current_step = step.name
                print(f"\n--- Step: {current_step} ---")

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
