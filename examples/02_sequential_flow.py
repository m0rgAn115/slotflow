"""Interactive SEQUENTIAL onboarding.

Asks one slot at a time in declaration order. Try answering ambiguously to
trigger a follow-up, or saying "I'd rather not say" on the optional phone slot.

Run with: ``.venv/bin/python examples/02_sequential_flow.py``
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
    ResponseJudgement,
    Slot,
    SlotPrompt,
    initial_state,
    next_message,
    process_response,
)


class DocumentType(str, Enum):
    DNI = "DNI"
    CE = "CE"
    PASSPORT = "Passport"


class UserOnboarding(OnboardingSchema):
    full_name: str = Slot(description="User's full name")
    document_type: DocumentType = Slot(description="Type of identity document")
    birth_date: date = Slot(description="Date of birth")
    phone: str | None = Slot(default=None, description="Contact phone number (optional)")


async def main() -> None:
    load_dotenv(find_dotenv(usecwd=True))
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing — copy .env.example to .env and fill it in.")

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    flow = OnboardingFlow(
        schema=UserOnboarding,
        mode=FlowMode.SEQUENTIAL,
        prompts={
            "phone": SlotPrompt(follow_up_hint="If the user hesitates, remind them it's optional."),
        },
    )

    state = initial_state(flow)
    state, turn = await next_message(flow=flow, state=state, llm=llm)

    while not turn.done:
        if turn.message is None:
            print(
                f"\nAssistant > (no message, verdict: {turn.judgement}). "
                "The user refused a required slot."
            )
            break
        print(f"\nAssistant > {turn.message}")

        user_text = input("You      > ").strip()
        if not user_text:
            print("(empty response; exiting demo)")
            return

        state, turn = await process_response(flow=flow, state=state, user_text=user_text, llm=llm)
        if turn.filled_this_turn:
            print(f"  · captured: {dict(turn.filled_this_turn)}")
        if turn.skipped_this_turn:
            print(f"  · skipped : {list(turn.skipped_this_turn)}")
        if turn.judgement and turn.judgement is not ResponseJudgement.COMPLETE:
            print(f"  · verdict : {turn.judgement.value}")

    print("\nFinal:")
    print(f"  filled  = {dict(state.filled)}")
    print(f"  skipped = {list(state.skipped)}")


if __name__ == "__main__":
    asyncio.run(main())
