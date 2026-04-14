"""
Deployed test script — runs against a live LangSmith Deployment.
=================================================================
This shows how to interact with the agent via the LangGraph SDK once it's
deployed to LangSmith Cloud (or any self-hosted deployment).

The graph accepts **empty** initial input: the first interrupt collects
destination and dates. If GeoIP fails, the next asks for departure location;
then preferences are collected after weather.

Environment variables needed:
    DEPLOYMENT_URL      — e.g. https://travel-planner-abc123.langsmith.com
    LANGSMITH_API_KEY   — your LangSmith API key
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from langgraph_sdk import get_client

load_dotenv()


def _is_origin_followup_prompt(payload) -> bool:
    t = str(payload or "").lower()
    return "couldn't automatically" in t or "departing from" in t


DEPLOYMENT_URL = os.environ.get("DEPLOYMENT_URL")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")

if not DEPLOYMENT_URL:
    sys.exit(
        "Error: DEPLOYMENT_URL is not set.\n"
        "Add it to your .env file: DEPLOYMENT_URL=https://your-deployment-url"
    )


async def run():
    client = get_client(url=DEPLOYMENT_URL, api_key=LANGSMITH_API_KEY)

    print("=" * 55)
    print("  Travel Planner Agent — Deployed Test")
    print("=" * 55)
    print()

    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    print(f"  Thread created: {thread_id}\n")

    async def stream_until_interrupt(thread_id_: str, inp: dict | None, command: dict | None):
        interrupted = False
        payload = None
        kwargs: dict = {
            "thread_id": thread_id_,
            "assistant_id": "travel_planner",
            "stream_mode": "updates",
        }
        if command is not None:
            kwargs["command"] = command
        else:
            kwargs["input"] = inp if inp is not None else {}

        async for chunk in client.runs.stream(**kwargs):
            if chunk.event != "updates":
                continue
            if "__interrupt__" in chunk.data:
                intr = chunk.data["__interrupt__"][0]
                payload = intr["value"] if isinstance(intr, dict) else intr.value
                interrupted = True
                break
            node_name = list(chunk.data.keys())[0]
            print(f"  + {node_name}")
        return payload, interrupted

    # ── Phase 1: empty input — first interrupt (trip details) ───────────────────

    print("Starting agent (first HITL = trip details)...\n")

    payload, interrupted = await stream_until_interrupt(thread_id, {}, None)
    if not interrupted:
        await _print_final_agenda(client, thread_id)
        return

    print()
    print("─" * 55)
    print("  Trip details (first interrupt)")
    print("─" * 55)
    print()
    print(f"  {payload}")
    print()

    trip_reply = input("  Your answer: ").strip()
    if not trip_reply:
        trip_reply = "Tokyo, Japan — June 10 through June 17, 2025"

    # ── Phase 2: resume → weather → second interrupt ────────────────────────────

    print()
    print("Resuming...\n")

    payload, interrupted = await stream_until_interrupt(
        thread_id, None, {"resume": trip_reply}
    )
    if not interrupted:
        await _print_final_agenda(client, thread_id)
        return

    if _is_origin_followup_prompt(payload):
        print()
        print("─" * 55)
        print("  Starting location (GeoIP could not infer)")
        print("─" * 55)
        print()
        print(f"  {payload}")
        print()

        origin_reply = input("  City/region you're departing from: ").strip()
        if not origin_reply:
            origin_reply = "San Francisco, CA"

        print()
        print("Resuming...\n")

        payload, interrupted = await stream_until_interrupt(
            thread_id, None, {"resume": origin_reply}
        )
        if not interrupted:
            await _print_final_agenda(client, thread_id)
            return

    print()
    print("─" * 55)
    print("  Preferences (interrupt)")
    print("─" * 55)
    print()
    print(f"  {payload}")
    print()

    pref_reply = input("  Your preferences: ").strip()
    if not pref_reply:
        pref_reply = "Mix of culture, local food, and some outdoor walks"

    # ── Phase 3: resume to completion ───────────────────────────────────────────

    print()
    print("Resuming...\n")

    _, interrupted = await stream_until_interrupt(
        thread_id, None, {"resume": pref_reply}
    )
    if interrupted:
        print("  (Unexpected extra interrupt.)")

    await _print_final_agenda(client, thread_id)


async def _print_final_agenda(client, thread_id: str):
    state = await client.threads.get_state(thread_id)
    values = state["values"]

    print()
    print("=" * 55)
    print("  Your Personalized Travel Agenda")
    print("=" * 55)
    print()
    print(values.get("agenda", "(no agenda generated)"))
    print()
    print("─" * 55)
    print(f"  Preferences captured: {values.get('activity_preferences', 'N/A')}")
    print(f"  Thread ID (for resuming later): {thread_id}")
    print("─" * 55)


if __name__ == "__main__":
    asyncio.run(run())
