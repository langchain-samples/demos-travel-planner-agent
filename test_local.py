"""
Local test script — no deployment required.
============================================
Runs the travel planner agent using an in-memory checkpointer so you can
test the full interrupt → resume flow on your machine before deploying.

Usage:
    pip install -r requirements.txt
    cp .env.example .env   # fill in your keys
    python test_local.py

What this demonstrates:
  - Compiling the graph with MemorySaver for local testing
  - Streaming updates from each node as they complete
  - Detecting the HITL interrupt mid-stream
  - Resuming execution with Command(resume=...) after user input
  - Reading the final state once the graph completes
"""

import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

# Import the builder (not the pre-compiled graph) so we can attach MemorySaver
from agent.graph import builder


def run():
    # ── Setup ──────────────────────────────────────────────────────────────────

    # MemorySaver keeps state in memory for the lifetime of this process.
    # In production (LangSmith Deployment), a PostgreSQL-backed checkpointer
    # is injected automatically — you don't configure this yourself.
    graph = builder.compile(checkpointer=MemorySaver())

    # thread_id is the persistent cursor that links the initial run to the
    # resumed run. Reusing the same ID = same checkpoint. New ID = fresh start.
    config = {"configurable": {"thread_id": "tokyo-trip-test-1"}}

    print("=" * 55)
    print("  Travel Planner Agent — Local Test")
    print("=" * 55)
    print()

    # ── Phase 1: Initial run (will pause at the HITL interrupt) ────────────────

    print("Starting agent...\n")

    interrupted = False
    interrupt_payload = None

    for chunk in graph.stream(
        {
            "location": "Tokyo, Japan",
            "start_date": "2025-06-10",
            "end_date": "2025-06-17",
        },
        config=config,
        stream_mode="updates",
    ):
        # Each chunk is a dict like {"node_name": {...state updates...}}
        # When an interrupt fires, the key is "__interrupt__"
        if "__interrupt__" in chunk:
            interrupted = True
            # LangGraph streams Interrupt objects (use .value), not dicts with ["value"].
            interrupt_payload = chunk["__interrupt__"][0].value
            break

        node_name = list(chunk.keys())[0]
        print(f"  ✓ {node_name}")

    if not interrupted:
        # Graph ran all the way through without interrupting (shouldn't happen
        # in this agent, but handle it cleanly anyway)
        _print_agenda(graph.get_state(config).values)
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

    # ── Phase 3: Resume from checkpoint ───────────────────────────────────────

    print()
    print("Resuming...\n")

    # Command(resume=...) is passed instead of new input.
    # LangGraph loads the saved checkpoint, injects the resume value as the
    # return of interrupt(), and continues execution from that exact point.
    for chunk in graph.stream(
        Command(resume=user_preferences),
        config=config,
        stream_mode="updates",
    ):
        if "__interrupt__" not in chunk:
            node_name = list(chunk.keys())[0]
            print(f"  ✓ {node_name}")

    # ── Final output ───────────────────────────────────────────────────────────

    final_state = graph.get_state(config).values
    _print_agenda(final_state)


def _print_agenda(state: dict):
    print()
    print("=" * 55)
    print("  Your Personalized Travel Agenda")
    print("=" * 55)
    print()
    print(state.get("agenda", "(no agenda generated)"))
    print()
    print("─" * 55)
    print(f"  Weather summary: {state.get('weather_summary', '')[:120]}...")
    print(f"  Preferences captured: {state.get('activity_preferences', 'N/A')}")
    print("─" * 55)


if __name__ == "__main__":
    run()
