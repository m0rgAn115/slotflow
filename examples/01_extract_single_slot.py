"""Minimal extraction example.

Declares a schema with one slot and asks the LLM to pull it out of a sentence.
Run with: ``.venv/bin/python examples/01_extract_single_slot.py``
"""

from __future__ import annotations

import asyncio
import os
from datetime import date

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI

from slotflow import OnboardingSchema, Slot, extract_slot


class Profile(OnboardingSchema):
    birth_date: date = Slot(description="User's date of birth")


async def main() -> None:
    load_dotenv(find_dotenv(usecwd=True))
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing — copy .env.example to .env and fill it in.")

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    result = await extract_slot(
        schema=Profile,
        field_name="birth_date",
        text="I was born on May 15, 1990 in New York.",
        llm=llm,
    )
    print(f"success={result.success}  attempts={result.attempts}")
    print(f"value={result.value!r}")
    if result.error:
        print(f"error={result.error}")


if __name__ == "__main__":
    asyncio.run(main())
