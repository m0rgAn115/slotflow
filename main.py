"""Demos for the library.

Three demos are included:

1. ``demo_single_slot`` — declare a schema and extract one slot from text.
2. ``demo_multi_slot`` — extract several slots in a single LLM call.
3. ``demo_interactive_flow`` — run a full conversational onboarding in the
   terminal using the new flow layer.

Setup:
  1. cp .env.example .env
  2. Put your OPENAI_API_KEY in .env
  3. .venv/bin/python main.py
"""

import asyncio
import json
import os
import sys
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
    extract_slot,
    extract_slots,
    initial_state,
    next_message,
    process_response,
)


class DocumentType(str, Enum):
    DNI = "DNI"
    CE = "CE"
    PASSPORT = "Passport"


class Address(OnboardingSchema):
    street: str = Slot(description="Calle y número de la dirección")
    city: str = Slot(description="Ciudad de residencia")


class UserOnboarding(OnboardingSchema):
    full_name: str = Slot(
        description="Nombre completo del usuario",
        examples=["Juan Pérez", "María García"],
    )
    document_type: DocumentType = Slot(description="Tipo de documento de identidad")
    birth_date: date = Slot(description="Fecha de nacimiento del usuario")
    phone: str | None = Slot(default=None, description="Teléfono de contacto (opcional)")
    address: Address | None = Slot(default=None, description="Dirección de residencia")


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


async def demo_single_slot(llm: ChatOpenAI) -> None:
    _print_section("Extracción de un slot (birth_date)")
    text = "Hola, nací el 15 de mayo de 1990 en Lima."
    print(f"Texto del usuario: {text!r}")

    result = await extract_slot(
        schema=UserOnboarding,
        field_name="birth_date",
        text=text,
        llm=llm,
    )
    print(f"success={result.success}  attempts={result.attempts}")
    print(f"value={result.value!r}")
    if result.error:
        print(f"error={result.error}")


async def demo_multi_slot(llm: ChatOpenAI) -> None:
    _print_section("Extracción de varios slots en una sola llamada")
    text = (
        "Soy Juan Pérez, mi DNI es el documento que uso, "
        "nací el 15/05/1990 y mi teléfono es +51 987654321."
    )
    print(f"Texto del usuario: {text!r}")

    result = await extract_slots(
        schema=UserOnboarding,
        field_names=["full_name", "document_type", "birth_date", "phone"],
        text=text,
        llm=llm,
    )
    print(f"success={result.success}  attempts={result.attempts}")
    print(f"value={result.value!r}")
    if result.error:
        print(f"error={result.error}")


async def demo_interactive_flow(llm: ChatOpenAI) -> None:
    _print_section("Onboarding conversacional interactivo")
    print(
        "Modo SEQUENTIAL: te pregunto un slot a la vez en el orden del schema.\n"
        "Prueba a saltar el teléfono diciendo 'prefiero no decir', dar respuestas "
        "ambiguas para forzar follow-ups, o cambia el modo de FlowMode abajo "
        "para probar FREEFORM, UNORDERED o STEPS."
    )

    flow = OnboardingFlow(
        schema=UserOnboarding,
        mode=FlowMode.SEQUENTIAL,
        prompts={
            "phone": SlotPrompt(follow_up_hint="Si el usuario duda, recuérdale que es opcional."),
        },
    )

    state = initial_state(flow)
    state, turn = await next_message(flow=flow, state=state, llm=llm)

    while not turn.done:
        if turn.message is not None:
            print(f"\nAssistant > {turn.message}")
        else:
            # REFUSED on a required slot — surface the situation and stop.
            print(
                f"\nAssistant > (Sin mensaje. Juicio: {turn.judgement}. "
                "El usuario rechazó un slot requerido; la app debería decidir "
                "cómo escalar.)"
            )
            break

        user_text = input("You      > ").strip()
        if not sys.stdin.isatty():
            print(user_text)
        if not user_text:
            print("(Respuesta vacía; saliendo del demo.)")
            return

        state, turn = await process_response(flow=flow, state=state, user_text=user_text, llm=llm)
        if turn.filled_this_turn:
            print(f"  · Capturado: {dict(turn.filled_this_turn)}")
        if turn.skipped_this_turn:
            print(f"  · Saltado: {list(turn.skipped_this_turn)}")
        if turn.judgement and turn.judgement is not ResponseJudgement.COMPLETE:
            print(f"  · Juicio: {turn.judgement.value}")

    print("\nResultado final:")
    print(f"  filled  = {dict(state.filled)}")
    print(f"  skipped = {list(state.skipped)}")


async def amain() -> None:
    load_dotenv(find_dotenv(usecwd=True))
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY no está definida. Copia .env.example a .env y pon tu clave."
        )

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)

    _print_section("JSON Schema generado para el LLM")
    print(json.dumps(UserOnboarding.model_json_schema(), indent=2, ensure_ascii=False))

    await demo_single_slot(llm)
    await demo_multi_slot(llm)
    await demo_interactive_flow(llm)


if __name__ == "__main__":
    asyncio.run(amain())
