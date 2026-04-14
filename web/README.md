# Travel Planner Web UI

React + TypeScript + Tailwind chat front-end for the deployed LangGraph agent. It talks to your LangSmith deployment through a **small local Express proxy** so `LANGSMITH_API_KEY` and `DEPLOYMENT_URL` stay off the browser.

## Setup

From the **repository root**, ensure `.env` includes:

- `DEPLOYMENT_URL` — your LangGraph deployment base URL (no trailing slash)
- `LANGSMITH_API_KEY` — same key you use for the deployment API

Optional:

- `LANGGRAPH_ASSISTANT_ID` — defaults to `travel_planner`
- `API_PORT` — proxy port (default **3847**; Vite proxies `/api` here in dev)

Install and run (from `web/`):

```bash
cd web
npm install
npm run dev
```

Open **http://127.0.0.1:5173**. Click **New trip** to create a thread and start a run with `{}`; answer the assistant’s prompts in order (destination → optional origin if GeoIP fails → preferences).

## Production / Vercel

The UI uses **relative** `/api` URLs so you can host the static build behind any origin.

- **Option A:** Deploy the Vite `dist/` as static assets and run the Express proxy (or equivalent) on a separate host; set your frontend’s public URL to proxy `/api` to that service.
- **Option B:** Re-implement `server/proxy.ts` as serverless routes (e.g. Vercel Node handlers) and set `DEPLOYMENT_URL` / `LANGSMITH_API_KEY` in the provider’s environment (never in the client bundle).

Branding assets live in `public/branding/` (copied from the repo `img/` logos).
