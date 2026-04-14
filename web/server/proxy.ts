/**
 * Local API server: loads repo-root `.env`, keeps LangSmith / deployment keys off the browser.
 * The React app calls `/api/*` (proxied by Vite in dev, or your host in production).
 */
import path from "node:path";
import { fileURLToPath } from "node:url";

import { Client } from "@langchain/langgraph-sdk";
import cors from "cors";
import dotenv from "dotenv";
import express from "express";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

dotenv.config({ path: path.resolve(__dirname, "../../.env") });

const DEPLOYMENT_URL = (process.env.DEPLOYMENT_URL ?? "").replace(/\/$/, "");
const API_KEY =
  process.env.LANGSMITH_API_KEY ??
  process.env.LANGGRAPH_API_KEY ??
  process.env.LANGCHAIN_API_KEY ??
  "";
const ASSISTANT_ID =
  process.env.LANGGRAPH_ASSISTANT_ID?.trim() || "travel_planner";
const PORT = Number(process.env.API_PORT ?? process.env.PORT ?? 3847);

function getClient(): Client {
  if (!DEPLOYMENT_URL) {
    throw new Error("DEPLOYMENT_URL is not set in .env");
  }
  if (!API_KEY) {
    throw new Error(
      "Set LANGSMITH_API_KEY (or LANGGRAPH_API_KEY) in .env for the deployment",
    );
  }
  return new Client({ apiUrl: DEPLOYMENT_URL, apiKey: API_KEY });
}

function sseWrite(
  res: express.Response,
  payload: Record<string, unknown>,
): void {
  res.write(`data: ${JSON.stringify(payload)}\n\n`);
}

/**
 * Interrupt payloads sometimes arrive nested under the node name in `updates`
 * events, not only as a top-level `__interrupt__` key. Scan recursively.
 */
function extractInterruptResumeValue(data: unknown): unknown | null {
  if (data == null || typeof data !== "object") return null;
  if (Array.isArray(data)) {
    for (const item of data) {
      const v = extractInterruptResumeValue(item);
      if (v !== null) return v;
    }
    return null;
  }
  const rec = data as Record<string, unknown>;
  if (Array.isArray(rec.__interrupt__) && rec.__interrupt__.length > 0) {
    const raw = rec.__interrupt__[0];
    if (raw != null && typeof raw === "object" && "value" in raw) {
      return (raw as { value: unknown }).value;
    }
    return raw;
  }
  for (const v of Object.values(rec)) {
    const found = extractInterruptResumeValue(v);
    if (found !== null) return found;
  }
  return null;
}

const app = express();
app.use(
  cors({
    origin: true,
    credentials: true,
  }),
);
app.use(express.json({ limit: "1mb" }));

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    deploymentConfigured: Boolean(DEPLOYMENT_URL && API_KEY),
    assistantId: ASSISTANT_ID,
  });
});

app.post("/api/threads", async (_req, res) => {
  try {
    const client = getClient();
    const thread = await client.threads.create();
    res.json({ threadId: thread.thread_id });
  } catch (e) {
    res.status(500).json({
      error: e instanceof Error ? e.message : String(e),
    });
  }
});

app.get("/api/threads/:threadId/state", async (req, res) => {
  try {
    const client = getClient();
    const state = await client.threads.getState(req.params.threadId);
    res.json({ values: state.values });
  } catch (e) {
    res.status(500).json({
      error: e instanceof Error ? e.message : String(e),
    });
  }
});

/** Stream graph updates until interrupt or completion. */
app.post("/api/runs/stream", async (req, res) => {
  const { threadId, input, command } = req.body as {
    threadId?: string;
    input?: Record<string, unknown> | null;
    command?: { resume?: unknown };
  };

  if (!threadId || typeof threadId !== "string") {
    res.status(400).json({ error: "threadId required" });
    return;
  }

  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders?.();

  try {
    const client = getClient();
    const stream = client.runs.stream(threadId, ASSISTANT_ID, {
      input: command ? undefined : input ?? {},
      command,
      streamMode: ["updates", "values"],
    });

    for await (const chunk of stream) {
      const ev = chunk.event;

      if (ev === "updates" || ev === "values") {
        const resume = extractInterruptResumeValue(chunk.data);
        if (resume !== null) {
          sseWrite(res, { type: "interrupt", value: resume });
          res.end();
          return;
        }
      }

      if (ev === "updates") {
        const data = chunk.data as Record<string, unknown> | null;
        if (!data || typeof data !== "object") continue;
        const nodeNames = Object.keys(data).filter((k) => !k.startsWith("__"));
        for (const name of nodeNames) {
          sseWrite(res, { type: "node", name });
        }
      }
    }

    sseWrite(res, { type: "done" });
    res.end();
  } catch (e) {
    sseWrite(res, {
      type: "error",
      message: e instanceof Error ? e.message : String(e),
    });
    res.end();
  }
});

app.listen(PORT, () => {
  console.log(
    `[travel-planner-web] API on http://127.0.0.1:${PORT} → deployment ${DEPLOYMENT_URL || "(missing)"}`,
  );
});
