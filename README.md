# slotflow

Drive multi-turn LLM conversations from a Pydantic schema. Declare what to collect with `Slot()`, choose a flow mode (sequential, freeform, steps), and let slotflow handle question generation, extraction, response judging, and follow-ups — with immutable state that serializes to Redis out of the box.

- Composition over modification — slots do not know about flows
- Pydantic does all validation, including dynamic per-call wrapper models
- Immutable state — every turn returns a new `FlowState`
- LLM is injected, never constructed internally (provider-agnostic)
- Core install is LLM-free; LangChain and OpenAI extras are optional

---

## Installation

```bash
# Schema layer only (no LLM dependency)
pip install llm-slotflow

# With LangChain backend
pip install "slotflow[langchain]"

# With OpenAI backend (no LangChain)
pip install "slotflow[openai]"
```

Requires Python 3.10+.

---

## Quickstart

### 1. Declare a schema

```python
from datetime import date
from enum import Enum
from typing import Optional

from slotflow import OnboardingSchema, Slot


class DocumentType(str, Enum):
    DNI = "DNI"
    CE = "CE"
    PASSPORT = "Passport"


class UserOnboarding(OnboardingSchema):
    full_name: str = Slot(description="User's full name")
    document_type: DocumentType = Slot(description="Type of identity document")
    birth_date: date = Slot(description="Date of birth")
    phone: Optional[str] = Slot(default=None, description="Phone number (optional)")
```

Slots without a `default` are required; slots with one are optional.

### 2. Extract slots from free-form text

```python
import asyncio
from langchain_openai import ChatOpenAI
from slotflow import extract_slots

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

result = asyncio.run(extract_slots(
    schema=UserOnboarding,
    field_names=["full_name", "document_type", "birth_date"],
    text="I'm John Smith, passport, born on 15/05/1990.",
    llm=llm,
))
print(result.value)
# {'full_name': 'John Smith', 'document_type': <DocumentType.PASSPORT>, 'birth_date': date(1990, 5, 15)}
```

`extract_slot` / `extract_slots` retry up to `max_attempts` (default 3),
feeding the Pydantic `ValidationError` back to the LLM as feedback.

### 3. Drive a full conversation

```python
import asyncio
from langchain_openai import ChatOpenAI
from slotflow import (
    FlowMode, OnboardingFlow, SlotPrompt,
    initial_state, next_message, process_response,
)

flow = OnboardingFlow(
    schema=UserOnboarding,
    mode=FlowMode.SEQUENTIAL,
    prompts={
        "phone": SlotPrompt(
            follow_up_hint="If the user hesitates, remind them it's optional."
        ),
    },
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


async def main() -> None:
    state = initial_state(flow)
    state, turn = await next_message(flow=flow, state=state, llm=llm)

    while not turn.done:
        print("Bot >", turn.message)
        user_text = input("You > ")
        state, turn = await process_response(
            flow=flow, state=state, user_text=user_text, llm=llm
        )

    print("Captured:", dict(state.filled))
    print("Skipped:", list(state.skipped))


asyncio.run(main())
```

---

## Flow modes

| Mode | What it does |
|---|---|
| `SEQUENTIAL` | Asks one slot at a time in schema declaration order. |
| `UNORDERED` | Asks one slot at a time but accepts answers for *any* pending slot in the same turn. |
| `FREEFORM` | Opens with everything that is missing; extracts many slots from one response. |
| `STEPS` | Groups slots into ordered `Step(...)`s; finishes a step before moving on. |

---

## Per-slot prompt overrides

Override question wording per slot, or hint the follow-up generator:

```python
OnboardingFlow(
    schema=UserOnboarding,
    prompts={
        "full_name": SlotPrompt(question="What's your full name?"),
        "birth_date": SlotPrompt(
            follow_up_hint="Ask for an explicit day/month/year.",
        ),
    },
)
```

When `SlotPrompt.question` is set, the LLM question-generation call is skipped
entirely — useful for tightly controlled wording or to save tokens.

---

## How a turn works

1. **Extract** — `extract_slot` / `extract_slots` is always called first.
2. **Judge** — `judge_response` classifies the response as one of
   `COMPLETE / PARTIAL / INSUFFICIENT / SKIP_INTENT / REFUSED`, given the
   user's raw text *and* the extracted value.
3. **Decide** — the runner fills the slot, marks an optional slot as skipped,
   asks a follow-up, or generates a nudge for a refused required slot.

---

## Stateless runner

`FlowState`, `Turn`, and `FlowTurn` are all frozen dataclasses. Every call to
`next_message` / `process_response` returns a new `FlowState`; the input is
never mutated. This makes it trivial to:

- run many conversations concurrently
- pickle the state between turns (Redis, DB, queue)
- snapshot/restore for testing or replay

```python
import pickle
serialized = pickle.dumps(state)  # works out of the box
```

---

## Examples

Runnable scripts live in [`examples/`](examples/):

- `01_extract_single_slot.py` — minimal extraction example
- `02_sequential_flow.py` — `SEQUENTIAL` mode end to end
- `03_freeform_flow.py` — `FREEFORM` mode end to end
- `04_steps_flow.py` — `STEPS` mode with grouped slots

All read `OPENAI_API_KEY` from `.env` (see `.env.example`).

---

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Run tests
.venv/bin/pytest tests/

# Lint + format
.venv/bin/ruff check .
.venv/bin/ruff format .

# Type check
.venv/bin/mypy

# Tests with coverage
.venv/bin/pytest --cov
```

---

## License

MIT — see [LICENSE](LICENSE).
