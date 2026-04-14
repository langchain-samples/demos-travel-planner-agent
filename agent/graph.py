"""
Travel Planner Agent
====================
A LangGraph agent that plans a personalized travel itinerary.

Flow:
  1. user_location_subagent — nested graph: GeoIP (or optional origin_city),
     geocode destination, great-circle distance, LLM summary of plane / train / car
  2. research_weather — searches for expected weather during the trip dates
  3. ask_preferences — HITL interrupt: pauses and asks the user what they enjoy
  4. research_attractions — searches for activities matching location + preferences
  5. assemble_agenda   — builds a day-by-day itinerary from everything gathered

The interrupt() in step 3 is what makes this a human-in-the-loop agent.

Deployment note:
  - When running via `langgraph dev` or LangSmith Deployment, the platform
    provides a PostgreSQL-backed checkpointer automatically. We compile the
    graph WITHOUT a checkpointer here so the platform can inject its own.
  - For standalone Python testing (test_local.py), compile with MemorySaver.

GeoIP note:
  - Default GeoIP reflects the **server egress IP** (deployment or your laptop),
    not the browser. For realistic demos, pass optional `origin_city` in graph input.
"""

from __future__ import annotations

import math
import os
from typing import Any, NotRequired, TypedDict

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

# ── State ──────────────────────────────────────────────────────────────────────
#
# LangSmith Studio uses `input_schema` for new-run input. Agent-populated keys
# use NotRequired so they are not shown as required fields. Do not set per-node
# `input_schema`: on Continue/retry, the API can invoke a node with a partial
# channel view, which caused KeyError despite the full checkpoint showing values.


class TravelInputState(TypedDict):
    """User-provided fields only (thread / run input in Studio)."""

    location: str
    start_date: str
    end_date: str
    origin_city: NotRequired[str]


class TravelState(TypedDict):
    """Full checkpoint state."""

    location: str
    start_date: str
    end_date: str
    origin_city: NotRequired[str]
    user_geo_summary: NotRequired[str]
    user_lat: NotRequired[float]
    user_lon: NotRequired[float]
    destination_lat: NotRequired[float]
    destination_lon: NotRequired[float]
    distance_km: NotRequired[float]
    travel_leg_summary: NotRequired[str]
    weather_summary: NotRequired[str]
    activity_preferences: NotRequired[str]
    attractions: NotRequired[list]
    agenda: NotRequired[str]


class TravelOutputState(TypedDict):
    """Primary deliverable shown as graph output in Studio."""

    agenda: str


# ── Tools + model ──────────────────────────────────────────────────────────────

search = TavilySearchResults(max_results=4)
llm = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"))

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT = "demos-travel-planner-agent/1.0 (LangChain travel planner demo)"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points in kilometers."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def _nominatim_geocode(query: str) -> tuple[float, float] | None:
    """Resolve a place name to (lat, lon) via Nominatim (be polite: one call per step)."""
    q = (query or "").strip()
    if not q:
        return None
    try:
        with httpx.Client(timeout=20.0) as client:
            r = client.get(
                _NOMINATIM_URL,
                params={"q": q, "format": "json", "limit": 1},
                headers={"User-Agent": _USER_AGENT},
            )
            r.raise_for_status()
            rows = r.json()
            if not rows:
                return None
            return float(rows[0]["lat"]), float(rows[0]["lon"])
    except (httpx.HTTPError, KeyError, ValueError, TypeError):
        return None


# ── Subagent: user location + travel leg ───────────────────────────────────────


def subagent_resolve_user_origin(state: TravelState) -> dict[str, Any]:
    """Infer where the user is starting from: optional `origin_city` or GeoIP."""
    hint = (state.get("origin_city") or "").strip()
    if hint:
        coords = _nominatim_geocode(hint)
        if coords:
            lat, lon = coords
            return {
                "user_lat": lat,
                "user_lon": lon,
                "user_geo_summary": f"Starting from {hint} (geocoded).",
            }
        return {
            "user_geo_summary": (
                f"Starting from {hint!r} (geocoding failed; distance may be unknown)."
            ),
        }

    try:
        with httpx.Client(timeout=20.0) as client:
            r = client.get("https://ipapi.co/json/")
            r.raise_for_status()
            data = r.json()
        if data.get("error"):
            raise ValueError(data.get("reason", "ipapi error"))
        lat, lon = data.get("latitude"), data.get("longitude")
        city = data.get("city") or ""
        region = data.get("region") or ""
        country = data.get("country_name") or ""
        parts = [p for p in (city, region, country) if p]
        summary = ", ".join(parts) if parts else "an unknown area (GeoIP)"
        out: dict[str, Any] = {
            "user_geo_summary": f"GeoIP suggests you are near {summary}.",
        }
        if lat is not None and lon is not None:
            out["user_lat"] = float(lat)
            out["user_lon"] = float(lon)
        return out
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        return {
            "user_geo_summary": (
                "Could not determine your location (GeoIP failed). "
                "Re-run with `origin_city` in the graph input for a precise trip leg."
            ),
        }


def subagent_geocode_destination(state: TravelState) -> dict[str, Any]:
    coords = _nominatim_geocode(state["location"])
    if not coords:
        return {}
    lat, lon = coords
    return {"destination_lat": lat, "destination_lon": lon}


def subagent_summarize_travel_options(state: TravelState) -> dict[str, Any]:
    """Use distance to suggest plane / train / driving at a high level (no turn-by-turn)."""
    ulat = state.get("user_lat")
    ulon = state.get("user_lon")
    dlat = state.get("destination_lat")
    dlon = state.get("destination_lon")
    origin_blurb = state.get("user_geo_summary") or "Origin unknown."
    dest = state["location"]

    if ulat is None or ulon is None or dlat is None or dlon is None:
        prompt = f"""You are a travel logistics assistant. We could not compute exact distance.

ORIGIN CONTEXT: {origin_blurb}
DESTINATION: {dest}

Give 4–6 short sentences: suggest how someone might typically reach {dest} (flights, rail if relevant, driving) without claiming a specific distance. No turn-by-turn directions. Warm, practical tone."""

        msg = llm.invoke(prompt)
        return {"travel_leg_summary": msg.content}

    km = _haversine_km(ulat, ulon, dlat, dlon)
    mi = km * 0.621371

    prompt = f"""You are a travel logistics assistant. The traveler is planning a trip to {dest}.

ORIGIN: {origin_blurb}
STRAIGHT-LINE DISTANCE (great circle, approximate): {km:.0f} km ({mi:.0f} mi).

Based on this distance, explain sensible options:
- **Air**: when flying is worth it vs overkill
- **Train / rail**: when it is realistic (mention only if plausibly relevant to this region)
- **Car**: driving time **very rough** only (e.g. "often around X–Y hours on the road" — NOT turn-by-turn)

Rules: no step-by-step driving directions; no invented exact flight prices or schedules. If the trip is very short (e.g. under ~150 km), lean into driving or regional rail. If very long, emphasize flying. Keep it to one tight paragraph plus up to 3 bullet points."""

    msg = llm.invoke(prompt)
    return {"travel_leg_summary": msg.content, "distance_km": km}


def _build_user_location_subagent() -> StateGraph:
    """Nested graph (subagent) for GeoIP / origin + destination geocode + travel modes."""
    sub = StateGraph(TravelState)
    sub.add_node("resolve_origin", subagent_resolve_user_origin)
    sub.add_node("geocode_destination", subagent_geocode_destination)
    sub.add_node("travel_options", subagent_summarize_travel_options)
    sub.add_edge(START, "resolve_origin")
    sub.add_edge("resolve_origin", "geocode_destination")
    sub.add_edge("geocode_destination", "travel_options")
    sub.add_edge("travel_options", END)
    return sub


user_location_subagent = _build_user_location_subagent().compile()


# ── Main graph nodes ─────────────────────────────────────────────────────────────


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
    leg = state.get("travel_leg_summary") or "Travel options summary not available."
    user_response = interrupt(
        f"{state.get('user_geo_summary', '')}\n\n"
        f"How to get there (high level): {leg}\n\n"
        f"I've also pulled weather for {state['location']} "
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
    leg = state.get("travel_leg_summary") or ""
    dist = state.get("distance_km")
    dist_line = f"Approx. great-circle distance from inferred origin: {dist:.0f} km.\n" if dist else ""

    prompt = f"""You are an expert travel planner. Build a detailed, day-by-day itinerary.

DESTINATION: {state['location']}
DATES: {state['start_date']} to {state['end_date']} ({num_days} days)
{dist_line}GETTING THERE (high level — use this, do not add turn-by-turn driving):
{leg}

WEATHER: {state['weather_summary']}
TRAVELER PREFERENCES: {state['activity_preferences']}
RESEARCHED ATTRACTIONS:
{state['attractions']}

Write a warm, practical itinerary. For each day include:
- Morning, afternoon, and evening suggestions
- At least one recommendation tied directly to their stated preferences
- One practical tip (book ahead, best time to visit, nearest transit, etc.)

On **day 1**, briefly acknowledge arrival / transport (consistent with the travel summary above) without contradicting it.

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

builder.add_node("user_location_subagent", user_location_subagent)
builder.add_node("research_weather", research_weather)
builder.add_node("ask_preferences", ask_preferences)
builder.add_node("research_attractions", research_attractions)
builder.add_node("assemble_agenda", assemble_agenda)

builder.add_edge(START, "user_location_subagent")
builder.add_edge("user_location_subagent", "research_weather")
builder.add_edge("research_weather", "ask_preferences")
builder.add_edge("ask_preferences", "research_attractions")
builder.add_edge("research_attractions", "assemble_agenda")
builder.add_edge("assemble_agenda", END)

# Compile WITHOUT a checkpointer.
# LangSmith Deployment and `langgraph dev` both inject a durable checkpointer.
# For standalone Python testing, see test_local.py which adds MemorySaver.
graph = builder.compile()
