"""
Local test script — no deployment required.
============================================
Runs the travel planner agent using an in-memory checkpointer so you can
test the full interrupt → resume flow on your machine before deploying.

Usage:
    pip install -r requirements.txt
    cp .env.example .env   # fill in your keys
    python test_local.py

The graph starts with **empty** input: the first interrupt asks for destination
and dates. If GeoIP cannot infer a starting point, a follow-up asks where you are
departing from. Then (after weather) preferences are collected.
"""

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

from agent.graph import builder


def _is_origin_followup_prompt(payload) -> bool:
    t = str(payload or "").lower()
    return "couldn't automatically" in t or "departing from" in t


def run():
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "tokyo-trip-test-1"}}

    print("=" * 55)
    print("  Travel Planner Agent — Local Test")
    print("=" * 55)
    print()

    def stream_until_interrupt(inp):
        """Stream until the next interrupt or graph completion."""
        for chunk in graph.stream(inp, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                return chunk["__interrupt__"][0].value, True
            node_name = list(chunk.keys())[0]
            print(f"  + {node_name}")
        return None, False

    # ── Phase 1: start with no trip fields — first interrupt is trip details ────

    print("Starting agent (trip details from first HITL prompt)...\n")

    payload, interrupted = stream_until_interrupt({})
    if not interrupted:
        _print_agenda(graph.get_state(config).values)
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

    # ── Phase 2: resume → weather → second interrupt (preferences) ─────────────

    print()
    print("Resuming...\n")

    payload, interrupted = stream_until_interrupt(Command(resume=trip_reply))
    if not interrupted:
        _print_agenda(graph.get_state(config).values)
        return

    if _is_origin_followup_prompt(payload):
        print()
        print("─" * 55)
        print("  Starting location (GeoIP could not infer — second interrupt)")
        print("─" * 55)
        print()
        print(f"  {payload}")
        print()

        origin_reply = input("  City/region you're departing from: ").strip()
        if not origin_reply:
            origin_reply = "San Francisco, CA"

        print()
        print("Resuming...\n")

        payload, interrupted = stream_until_interrupt(Command(resume=origin_reply))
        if not interrupted:
            _print_agenda(graph.get_state(config).values)
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

    _, interrupted = stream_until_interrupt(Command(resume=pref_reply))
    if interrupted:
        print("  (Unexpected extra interrupt — check graph.)")

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
