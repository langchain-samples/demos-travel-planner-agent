"""
Travel Planner Agent
====================
A LangGraph agent that plans a personalized travel itinerary.

Flow:
  1. **Parallel from START** (nothing that *needs* destination/dates runs until both finish):
     - ``collect_trip_details_hitl`` — **first human step**: ``interrupt()`` asks for the
       **destination** and dates; resume text is parsed into ``location``, ``start_date``,
       ``end_date``. We do **not** ask for origin here.
     - ``infer_user_origin_parallel`` — **GeoIP only** (independent of destination); runs
       in parallel with that interrupt. Does not read trip details.
  2. ``join_parallel_trip_prep`` — barrier.
  3. ``ask_origin_if_needed_hitl`` — if GeoIP did not yield coordinates, **only then**
     ``interrupt()`` to ask where the user is **departing from**, with an explanation.
  4. research_weather — weather at the destination for those dates
  5. ask_preferences — ``interrupt()`` for what the traveler enjoys
  6. travel_leg_geocode_destination + travel_leg_summarize_options — geocode + travel modes
  7. research_attractions — activities
  8. assemble_agenda — itinerary

Runs may start with **empty** graph input (Studio shows no misleading “origin” form).
Trip details and optional origin (when needed) come from **interrupts**, not the run form.

Deployment note:
  - When running via `langgraph dev` or LangSmith Deployment, the platform
    provides a PostgreSQL-backed checkpointer automatically. We compile the
    graph WITHOUT a checkpointer here so the platform can inject its own.
  - For standalone Python testing (test_local.py), compile with MemorySaver.

GeoIP note:
  - Default GeoIP reflects the **server egress IP** (deployment or your laptop),
    not the browser. If it fails or returns no coordinates, the graph asks the user
    for their departure city before continuing.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Mapping, NotRequired, TypedDict

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


class TravelInputState(TypedDict, total=False):
    """Studio/new-run input: intentionally **empty** so the UI does not imply we need origin.

    Destination, dates, and (if needed) departure location are collected via ``interrupt()``.
    Callers that invoke the graph programmatically may still pass optional ``origin_city``
    (and other ``TravelState`` keys) where the runtime merges extra keys into state.
    """

    pass


class TravelState(TypedDict, total=False):
    """Full checkpoint state. Core trip fields appear after the first HITL resume."""

    location: str
    start_date: str
    end_date: str
    origin_city: str
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


# ── Input normalization (LangSmith Studio) ─────────────────────────────────────

_TRIP_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "location": ("location", "Location", "destination", "Destination"),
    "start_date": ("start_date", "Start Date", "startDate", "StartDate"),
    "end_date": ("end_date", "End Date", "endDate", "EndDate"),
    "origin_city": ("origin_city", "Origin City", "originCity", "OriginCity"),
}


def _input_blobs(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten top-level state plus common nested envelopes Studio/APIs may use."""
    blobs: list[dict[str, Any]] = [dict(state)]
    for key in ("values", "input", "state"):
        inner = state.get(key)
        if isinstance(inner, dict):
            blobs.append(dict(inner))
    return blobs


def _first_non_empty(blobs: list[dict[str, Any]], keys: tuple[str, ...]) -> str | None:
    for blob in blobs:
        for k in keys:
            raw = blob.get(k)
            if raw is None:
                continue
            s = str(raw).strip()
            if s:
                return s
    return None


def _parse_trip_details_from_text(text: str) -> dict[str, str]:
    """Turn the user's free-text reply into canonical trip fields."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Trip details are empty. Please provide destination and dates.")

    prompt = f"""Extract trip details from the traveler's message.

Return ONLY a compact JSON object with exactly these string keys:
- "location": destination (city/region/country as they described it)
- "start_date": start date as YYYY-MM-DD if at all possible
- "end_date": end date as YYYY-MM-DD if at all possible

If a date is vague (e.g. "next June"), pick reasonable concrete YYYY-MM-DD and prefer the stated year if any.

Traveler message:
{cleaned}
"""
    raw = llm.invoke(prompt).content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    loc = str(data.get("location", "")).strip()
    sd = str(data.get("start_date", "")).strip()
    ed = str(data.get("end_date", "")).strip()
    if not loc or not sd or not ed:
        raise ValueError(
            "Could not read destination and both dates from your message. "
            "Please reply with where you're going and start/end dates "
            "(e.g. Kyoto, 2025-06-10 to 2025-06-17)."
        )
    return {"location": loc, "start_date": sd, "end_date": ed}


def collect_trip_details_hitl(state: TravelState) -> dict[str, Any]:
    """
    First human step: **destination and dates only** from this ``interrupt()`` (never origin).

    ``interrupt()`` is the first substantive step so a failed sibling task cannot
    cancel this node before the human prompt is raised (LangGraph cancels parallel
    tasks when any sibling raises a non-``GraphBubbleUp`` exception).

    Origin is inferred via GeoIP in parallel; if that fails, ``ask_origin_if_needed_hitl``
    asks for departure location and explains why.
    """
    user_reply = interrupt(
        "Where do you want to go, and when?\n\n"
        "Reply in one message with:\n"
        "• Your **destination** (city / region / country)\n"
        "• **Start date** and **end date** (calendar dates; YYYY-MM-DD is ideal)\n\n"
        "Example: \"Kyoto, Japan — June 10 through June 17, 2025\""
    )
    parsed = _parse_trip_details_from_text(str(user_reply))
    return parsed


# ── Parallel: infer home / egress location (GeoIP or origin_city) ─────────────


def infer_user_origin_parallel(state: TravelState) -> dict[str, Any]:
    """
    Infer where the traveler is starting from, **in parallel** with ``collect_trip_details_hitl``.

    Does **not** use destination or trip dates. Uses optional ``origin_city`` from run input /
    aliases, otherwise GeoIP (server egress IP).

    This node must **never raise**: a parallel exception cancels sibling tasks in LangGraph,
    which would prevent the trip-details ``interrupt()`` from running.
    """
    try:
        blobs = _input_blobs(state)
        hint = (state.get("origin_city") or "").strip()
        if not hint:
            found = _first_non_empty(blobs, _TRIP_FIELD_ALIASES["origin_city"])
            if found:
                hint = found.strip()
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
                    "Could not determine your location automatically (GeoIP lookup failed)."
                ),
            }
    except Exception:
        return {
            "user_geo_summary": (
                "Could not infer starting location automatically (unexpected error during GeoIP)."
            ),
        }


# ── Travel leg (post-HITL: destination geocode + modes) ─────────────────────────


def subagent_geocode_destination(state: TravelState) -> dict[str, Any]:
    """Geocode trip destination; tolerate missing `location` (partial Studio replays)."""
    loc = (state.get("location") or "").strip()
    if not loc:
        return {}
    coords = _nominatim_geocode(loc)
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
    dest = (state.get("location") or "").strip() or "your destination"

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


def join_parallel_trip_prep(state: TravelState) -> dict:
    """Join: trip details HITL + parallel GeoIP both completed."""
    return {}


def ask_origin_if_needed_hitl(state: TravelState) -> dict[str, Any]:
    """
    Second human step **only when** parallel GeoIP did not produce coordinates.

    If we already have ``user_lat`` / ``user_lon`` (GeoIP or a programmatic
    ``origin_city`` that geocoded in the parallel step), skip. Otherwise
    ``interrupt()`` and explain that automatic detection failed, then geocode the reply.
    """
    ulat, ulon = state.get("user_lat"), state.get("user_lon")
    if ulat is not None and ulon is not None:
        return {}

    user_reply = interrupt(
        "We couldn't automatically detect where you're traveling **from** "
        "(GeoIP only sees an approximate network location and sometimes fails).\n\n"
        "**Reply with the city or region you're departing from** "
        "(e.g. \"Austin, TX\" or \"London, UK\"). "
        "We'll use it for distance and how to get to your destination — "
        "you've already told us where you're going."
    )
    hint = str(user_reply).strip()
    if not hint:
        return {
            "user_geo_summary": (
                "Starting location unknown; travel distance and mode suggestions may be generic."
            ),
        }
    coords = _nominatim_geocode(hint)
    if coords:
        lat, lon = coords
        return {
            "origin_city": hint,
            "user_lat": lat,
            "user_lon": lon,
            "user_geo_summary": (
                f"Starting from {hint} (you provided this after automatic detection could not place you)."
            ),
        }
    return {
        "origin_city": hint,
        "user_geo_summary": (
            f"Starting from {hint!r} (geocoding failed; distance may be unknown)."
        ),
    }


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

# Register nodes in execution order so Studio’s graph layout matches the run.
builder.add_node("collect_trip_details_hitl", collect_trip_details_hitl)
builder.add_node("infer_user_origin_parallel", infer_user_origin_parallel)
builder.add_node("join_parallel_trip_prep", join_parallel_trip_prep)
builder.add_node("ask_origin_if_needed_hitl", ask_origin_if_needed_hitl)
builder.add_node("research_weather", research_weather)
builder.add_node("ask_preferences", ask_preferences)
builder.add_node("travel_leg_geocode_destination", subagent_geocode_destination)
builder.add_node("travel_leg_summarize_options", subagent_summarize_travel_options)
builder.add_node("research_attractions", research_attractions)
builder.add_node("assemble_agenda", assemble_agenda)

# First step: trip Q&A (HITL) in parallel with GeoIP-only origin inference.
builder.add_edge(START, "collect_trip_details_hitl")
builder.add_edge(START, "infer_user_origin_parallel")
builder.add_edge("collect_trip_details_hitl", "join_parallel_trip_prep")
builder.add_edge("infer_user_origin_parallel", "join_parallel_trip_prep")
builder.add_edge("join_parallel_trip_prep", "ask_origin_if_needed_hitl")
builder.add_edge("ask_origin_if_needed_hitl", "research_weather")
builder.add_edge("research_weather", "ask_preferences")
builder.add_edge("ask_preferences", "travel_leg_geocode_destination")
builder.add_edge("travel_leg_geocode_destination", "travel_leg_summarize_options")
builder.add_edge("travel_leg_summarize_options", "research_attractions")
builder.add_edge("research_attractions", "assemble_agenda")
builder.add_edge("assemble_agenda", END)

# Compile WITHOUT a checkpointer.
# LangSmith Deployment and `langgraph dev` both inject a durable checkpointer.
# For standalone Python testing, see test_local.py which adds MemorySaver.
graph = builder.compile()
