# Travel Planner Agent (LangGraph + LangSmith)

This repository is an **example project** for building a **human-in-the-loop agent** with [LangGraph](https://github.com/langchain-ai/langgraph) and running it through **LangSmith Deployment**, while using **`langgraph dev`** locally to spin up the API server and open **LangSmith Studio** for debugging, threads, and interrupts.

It is not a production travel product; it is a **reference pattern** you can copy: stateful graph, web search + LLM tools, `interrupt()` for HITL, and the same graph definition that works locally (Studio) and when deployed.

## What the agent does

The graph `travel_planner` (see `langgraph.json`) starts with **where & when** and **inferred origin** in **parallel**, then continues sequentially:

1. **Parallel (from `__start__`)**  
   - **`ingest_trip_input`** — Normalizes run input into **`location`**, **`start_date`**, **`end_date`** (and optional **`origin_city`**), including Studio label casing / nested `values`.  
   - **`infer_user_origin_parallel`** — **`origin_city`** or **GeoIP** ([ipapi.co](https://ipapi.co)); reads aliased keys so it does not depend on ingest finishing first.  
2. **`join_parallel_trip_prep`** — Barrier: both branches must complete before any trip planning continues.  
3. **`research_weather`** — Tavily + Claude for weather at the destination over the trip window.  
4. **`ask_preferences`** — **`interrupt()`** for what the traveler enjoys.  
5. **`travel_leg_geocode_destination`** + **`travel_leg_summarize_options`** — Geocode destination ([Nominatim](https://nominatim.org)), great-circle distance, Claude **high-level** travel modes (air / rail / rough drive times; **no turn-by-turn**).  
6. **`research_attractions`** — Tavily for activities aligned with preferences.  
7. **`assemble_agenda`** — Day-by-day itinerary (uses travel-leg summary on day one).

**Input** for a new run: **`location`**, **`start_date`**, **`end_date`**, and optionally **`origin_city`**. Other state is filled by the graph. The compiled graph has **no** checkpointer in code so **`langgraph dev`** and LangSmith Deployment inject storage.

### If Studio’s trace looks wrong

Restart **`langgraph dev`**, redeploy, or use a **new thread** after changing `agent/graph.py`. Old checkpoints can reflect an older graph shape.

## Prerequisites

- Python 3.11+ recommended (3.12/3.13 are a good default; very new Python versions may show dependency warnings).
- Accounts / keys: [Anthropic](https://www.anthropic.com/), [Tavily](https://tavily.com/), and [LangSmith](https://smith.langchain.com/) (for Studio against dev and for deployment).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set at least:

- `ANTHROPIC_API_KEY`
- `TAVILY_API_KEY`
- `LANGSMITH_API_KEY`

Optional: `ANTHROPIC_MODEL` (defaults to a current Sonnet id in code). See `.env.example` for tracing variables.

Install the in-memory API stack for the dev server if your environment is missing it:

```bash
pip install -U "langgraph-cli[inmem]"
```

Use the **same virtualenv** for both `pip` and `langgraph` so the CLI can import `langgraph-api`.

## Local dev: `langgraph dev` and LangSmith Studio

From the repo root (with `.env` present and venv activated):

```bash
langgraph dev
```

This reads `langgraph.json`, loads `./agent/graph.py:graph`, and starts the LangGraph API in development mode. The CLI prints a **Studio** link (and may open the browser). In Studio you can start threads, pass graph input, step through nodes, and complete the **interrupt** step as you would in a deployed agent.

If the default port is busy (often **2024**), either stop the other process or run:

```bash
langgraph dev --port 2025
```

## LangSmith Deployment

To run the same graph on LangSmith Deployment, connect this repo per the [LangSmith deployment docs](https://docs.smith.langchain.com/deployment) (Git integration or CLI). The `langgraph.json` graph id **`travel_planner`** is the assistant id you invoke from the SDK or UI.

## Scripts in this repo

| Script | Purpose |
|--------|--------|
| `python test_local.py` | Runs the graph in-process with `MemorySaver` to exercise streaming, interrupt, and `Command(resume=...)` without deployment. |
| `python test_deployed.py` | Same flow against a live deployment URL (`DEPLOYMENT_URL` + `LANGSMITH_API_KEY` in `.env`). |

## Project layout

```
agent/graph.py    # State schemas, nodes, StateGraph builder, compiled `graph`
langgraph.json    # Dependencies, graph id → graph path, `.env` path
requirements.txt  # Python dependencies
```

## License

Refer to repository policy for your org; this demo is intended for learning and as a LangChain / LangSmith example.
