# Examples

Runnable demos for the library. Each script is self-contained.

## Setup

```bash
# From the repo root
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

## Run

```bash
.venv/bin/python examples/01_extract_single_slot.py
.venv/bin/python examples/02_sequential_flow.py
.venv/bin/python examples/03_freeform_flow.py
.venv/bin/python examples/04_steps_flow.py
```

The flow examples are interactive — they read from stdin. Press Enter on an
empty line to abort.

| Script | What it shows |
|---|---|
| `01_extract_single_slot.py` | Minimal: one schema + one call to `extract_slot`. |
| `02_sequential_flow.py` | `FlowMode.SEQUENTIAL` end to end with `next_message` / `process_response`. |
| `03_freeform_flow.py` | `FlowMode.FREEFORM`: capture many slots from one response, follow up on what's missing. |
| `04_steps_flow.py` | `FlowMode.STEPS`: slots grouped into named steps walked in order. |
