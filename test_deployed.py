"""
Deployed test script — runs against a live LangSmith Deployment.
=================================================================
This shows how to interact with the agent via the LangGraph SDK once it's
deployed to LangSmith Cloud (or any self-hosted deployment).

The flow is the same as test_local.py, but state lives on the server and
we communicate via HTTP using the SDK instead of calling the graph directly.

Setup:
    1. Deploy the agent: push to GitHub and connect in LangSmith UI
       OR run: langgraph deploy (requires Docker)
    2. Copy your deployment URL from the LangSmith Deployments view
    3. Set environment variables (see below) and run:
       python test_deployed.py

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

    # ── Create a thread ────────────────────────────────────────────────────────
    # A thread is the server-side equivalent of a thread_id + checkpointer.
    # All runs on the same thread share state — this is what makes HITL possible.
    # The thread persists on the server, so you can resume hours later.
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    print(f"  Thread created: {thread_id}\n")

    # ── Phase 1: Initial run ───────────────────────────────────────────────────

    print("Starting agent...\n")

    interrupted = False
    interrupt_payload = None

    async for chunk in client.runs.stream(
        thread_id,
        "travel_planner",       # Must match the key in langgraph.json
        input={
            "location": "Tokyo, Japan",
            "start_date": "2025-06-10",
            "end_date": "2025-06-17",
        },
        stream_mode="updates",
    ):
        if chunk.event != "updates":
            continue

        if "__interrupt__" in chunk.data:
            interrupted = True
            interrupt_payload = chunk.data["__interrupt__"][0]["value"]
            break

        node_name = list(chunk.data.keys())[0]
        print(f"  ✓ {node_name}")

    if not interrupted:
        await _print_final_agenda(client, thread_id)
        return

    # ── Phase 2: Human-in-the-loop ─────────────────────────────────────────────

    print()
    print("─" * 55)
    print("  Agent paused — waiting for your input")
    print("─" * 55)
    print()
    print(f"  {interrupt_payload}")
    print()

    user_preferences = input("  Your preferences: ").strip()
    if not user_preferences:
        user_preferences = "Mix of culture, local food, and some outdoor walks"

    # ── Phase 3: Resume from server-side checkpoint ────────────────────────────

    print()
    print("Resuming...\n")

    # Passing command={"resume": ...} instead of input tells the server to
    # resume from the checkpoint where the interrupt fired. The user_preferences
    # string becomes the return value of interrupt() in ask_preferences().
    async for chunk in client.runs.stream(
        thread_id,
        "travel_planner",
        command={"resume": user_preferences},
        stream_mode="updates",
    ):
        if chunk.event == "updates" and "__interrupt__" not in chunk.data:
            node_name = list(chunk.data.keys())[0]
            print(f"  ✓ {node_name}")

    # ── Final output ───────────────────────────────────────────────────────────
    await _print_final_agenda(client, thread_id)


async def _print_final_agenda(client, thread_id: str):
    """Fetch the completed thread state and print the agenda."""
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
