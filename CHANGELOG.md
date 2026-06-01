# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `py.typed` marker so consumers receive the library's type hints (PEP 561).
- `README.md` with installation, quickstart, flow-mode overview, and an
  examples pointer.
- `examples/` directory with runnable end-to-end scripts for each flow mode.
- `OnboardingFlow.freeform_intro` template (default `"Tell me: {items}."`)
  so the FREEFORM intro is configurable per locale.
- Module-level loggers (`extractor`, `judge`, `question_generator`) — debug
  logs per attempt, warnings when fallbacks kick in.
- Module-level constants for default retry counts:
  `DEFAULT_EXTRACTION_MAX_ATTEMPTS = 3`, `DEFAULT_JUDGE_MAX_ATTEMPTS = 2`.
- Tooling: `ruff` (lint + format), `mypy`, `pytest-cov` configured in
  `pyproject.toml`.

### Changed
- `OnboardingFlow.is_done` and `OnboardingFlow.targets_for` now have a real
  `FlowState` annotation via `TYPE_CHECKING`, replacing the placeholder
  `state: Any`.
- `runner._process_single_target` split into per-verdict helpers
  (`_handle_skip_intent`, `_handle_refused`, `_handle_insufficient`,
  `_advance_complete`) for readability.
- Docstrings added/expanded on the public API (`initial_state`,
  `next_message`, `process_response`, `OnboardingFlow`, `OnboardingSchema`,
  `Slot`).

## [0.1.0]

Initial release.

- `OnboardingSchema` + `Slot()` declarative layer.
- `extract_slot` / `extract_slots` LLM extraction with retry-on-validation.
- `OnboardingFlow` composition layer with `SEQUENTIAL`, `UNORDERED`,
  `FREEFORM`, `STEPS` modes.
- `judge_response` for response classification.
- `next_message` / `process_response` runner with immutable `FlowState`.
- Optional `[llm]` extra for LangChain dependencies; core install stays
  Pydantic-only and lazily loads the LLM modules.
