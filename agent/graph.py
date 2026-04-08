"""
Travel Planner Agent
====================
A LangGraph agent that plans a personalized travel itinerary.

Flow:
  1. research_weather  — searches for expected weather during the trip dates
  2. ask_preferences   — HITL interrupt: pauses and asks the user what they enjoy
  3. research_attractions — searches for activities matching location + preferences
  4. assemble_agenda   — builds a day-by-day itinerary from everything gathered

The interrupt() call in step 2 is what makes this a human-in-the-loop agent.
Execution saves its state to a checkpoint and waits — indefinitely — until
resumed with Command(resume=<user_input>).

Deployment note:
  - When running via `langgraph dev` or LangSmith Deployment, the platform
    provides a PostgreSQL-backed checkpointer automatically. We compile the
    graph WITHOUT a checkpointer here so the platform can inject its own.
  - For standalone Python testing (test_local.py), compile with MemorySaver.
"""

import os
from typing import NotRequired, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


# ── State ──────────────────────────────────────────────────────────────────────
#
# LangSmith Studio uses `input_schema` for new-run input. Agent-populated keys
# use NotRequired so they are not required in forms. Do not set per-node
# `input_schema`: on Continue/retry, the API can invoke a node with a partial
# channel view, which caused KeyError despite the full checkpoint showing values.


class TravelInputState(TypedDict):
    """User-provided fields only (thread / run input in Studio)."""

    location: str
    start_date: str
    end_date: str


class TravelState(TypedDict):
    """Full checkpoint state."""

    location: str
    start_date: str
    end_date: str
    weather_summary: NotRequired[str]
    activity_preferences: NotRequired[str]
    attractions: NotRequired[list]
    agenda: NotRequired[str]


class TravelOutputState(TypedDict):
    """Primary deliverable shown as graph output in Studio."""

    agenda: str


# ── Tools + model ──────────────────────────────────────────────────────────────

search = TavilySearchResults(max_results=4)
# claude-3-5-sonnet-20241022 is retired; override with ANTHROPIC_MODEL if needed.
llm = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"))


# ── Nodes ──────────────────────────────────────────────────────────────────────

def research_weather(state: TravelState) -> dict:
    """
    Search for expected weather during the trip and produce a plain-language summary.
    Uses Tavily for up-to-date forecast + historical average data.
    """
    query = (
        f"weather forecast and typical climate {state['location']} "
        f"{state['start_date']} through {state['end_date']}"
    )
    results = search.invoke(query)

    summary = llm.invoke(
        f"Summarize the expected weather in {state['location']} between "
        f"{state['start_date']} and {state['end_date']} based on these results. "
        f"Be specific: temperature range, rain chance, and what to pack.\n\n"
        f"Search results:\n{results}"
    )
    return {"weather_summary": summary.content}


def ask_preferences(state: TravelState) -> dict:
    """
    Human-in-the-loop step.

    interrupt() saves the current graph state to a checkpoint and surfaces
    the payload to the caller. Execution is suspended here until the caller
    resumes with Command(resume=<user_response>), at which point `user_response`
    becomes the return value of interrupt() and the node completes normally.

    The payload is a plain string so Studio and other UIs show a readable
    prompt instead of raw JSON.
    """
    user_response = interrupt(
        f"I've pulled the weather for {state['location']} "
        f"({state['start_date']} → {state['end_date']}). "
        f"Before I build your itinerary, tell me what you enjoy. "
        f"For example: outdoor adventures, museums, local food markets, "
        f"nightlife, family-friendly activities, off-the-beaten-path spots — "
        f"the more detail, the better!"
    )

    return {"activity_preferences": user_response}


def research_attractions(state: TravelState) -> dict:
    """
    Search for activities and attractions tailored to the user's stated preferences.
    Runs after the HITL step so results are personalized, not generic.
    """
    query = (
        f"best things to do {state['location']} "
        f"{state['activity_preferences']} "
        f"tourist activities attractions local experiences"
    )
    results = search.invoke(query)
    return {"attractions": results}


def assemble_agenda(state: TravelState) -> dict:
    """
    Build a day-by-day itinerary from the weather summary, user preferences,
    and researched attractions. This is the final output of the agent.
    """
    num_days = _count_days(state["start_date"], state["end_date"])

    prompt = f"""You are an expert travel planner. Build a detailed, day-by-day itinerary.

DESTINATION: {state['location']}
DATES: {state['start_date']} to {state['end_date']} ({num_days} days)
WEATHER: {state['weather_summary']}
TRAVELER PREFERENCES: {state['activity_preferences']}
RESEARCHED ATTRACTIONS:
{state['attractions']}

Write a warm, practical itinerary. For each day include:
- Morning, afternoon, and evening suggestions
- At least one recommendation tied directly to their stated preferences
- One practical tip (book ahead, best time to visit, nearest transit, etc.)

Don't list generic tourist highlights — lean into their preferences.
Keep the tone helpful and personal, not brochure-like."""

    result = llm.invoke(prompt)
    return {"agenda": result.content}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _count_days(start: str, end: str) -> int:
    """Return number of days between two ISO date strings."""
    from datetime import date
    try:
        s = date.fromisoformat(start)
        e = date.fromisoformat(end)
        return max((e - s).days, 1)
    except ValueError:
        return 1


# ── Graph definition ───────────────────────────────────────────────────────────

builder = StateGraph(
    TravelState,
    input_schema=TravelInputState,
    output_schema=TravelOutputState,
)

builder.add_node("research_weather", research_weather)
builder.add_node("ask_preferences", ask_preferences)
builder.add_node("research_attractions", research_attractions)
builder.add_node("assemble_agenda", assemble_agenda)

builder.add_edge(START, "research_weather")
builder.add_edge("research_weather", "ask_preferences")
builder.add_edge("ask_preferences", "research_attractions")
builder.add_edge("research_attractions", "assemble_agenda")
builder.add_edge("assemble_agenda", END)

# Compile WITHOUT a checkpointer.
# LangSmith Deployment and `langgraph dev` both inject a durable checkpointer.
# For standalone Python testing, see test_local.py which adds MemorySaver.
graph = builder.compile()
