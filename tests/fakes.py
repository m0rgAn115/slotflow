"""Shared test doubles.

``FakeStructuredLLM`` mirrors the ``LLM`` Protocol from ``slotflow.llm``:
``with_structured_output(cls)`` and ``ainvoke(messages)``.

Three response-script forms are supported:

- ``list``: a single ordered queue, used by tests that only exercise one
  structured-output model per call (e.g. extractor-only tests).
- ``dict[type[BaseModel], list]`` + optional ``default=...``: a queue per
  output model, plus an optional fallback queue for any model not in the dict.
  The fallback is how we script responses for the extractor's dynamic wrapper
  model — its class name is generated at runtime so we can't key it directly.

Each scripted response is one of:

- a ``dict`` (mimics providers that return raw JSON; goes through
  ``model_validate``)
- a Pydantic model instance (mimics providers that return the parsed model)
- an ``Exception`` instance (raised on ``ainvoke``; mimics ``ParseError``)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class FakeStructuredLLM:
    def __init__(
        self,
        responses: list[Any] | dict[type[BaseModel], list[Any]],
        default: list[Any] | None = None,
    ):
        if isinstance(responses, dict):
            self._responses_by_model: dict[type[BaseModel], list[Any]] = {
                k: list(v) for k, v in responses.items()
            }
            self._default_queue: list[Any] | None = list(default) if default is not None else None
            self._responses: list[Any] | None = None
        else:
            if default is not None:
                raise ValueError("default= can only be used when responses is a dict")
            self._responses_by_model = {}
            self._default_queue = None
            self._responses = list(responses)
        self._index = 0
        self.last_model_cls: type[BaseModel] | None = None
        self.invocations: list[list[Any]] = []
        self.invocations_by_model: dict[type[BaseModel], list[list[Any]]] = {}

    def with_structured_output(self, model_cls: type[BaseModel]) -> FakeStructuredLLM:
        self.last_model_cls = model_cls
        return self

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.invocations.append(messages)
        model_cls = self.last_model_cls
        if model_cls is not None:
            self.invocations_by_model.setdefault(model_cls, []).append(messages)

        if self._responses is not None:
            if self._index >= len(self._responses):
                raise RuntimeError("FakeStructuredLLM ran out of scripted responses")
            resp = self._responses[self._index]
            self._index += 1
        else:
            queue = self._responses_by_model.get(model_cls)
            if not queue:
                if self._default_queue:
                    queue = self._default_queue
                else:
                    raise RuntimeError(
                        f"FakeStructuredLLM has no scripted response for model "
                        f"{getattr(model_cls, '__name__', model_cls)!r}"
                    )
            resp = queue.pop(0)

        if isinstance(resp, Exception):
            raise resp
        return resp
