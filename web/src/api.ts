const apiBase = "";

export type StreamEvent =
  | { type: "node"; name: string }
  | { type: "interrupt"; value: unknown }
  | { type: "done" }
  | { type: "error"; message: string };

export async function fetchHealth(): Promise<{
  ok: boolean;
  deploymentConfigured: boolean;
  assistantId: string;
}> {
  const r = await fetch(`${apiBase}/api/health`);
  if (!r.ok) throw new Error(`Health check failed: ${r.status}`);
  return r.json();
}

export async function createThread(): Promise<{ threadId: string }> {
  const r = await fetch(`${apiBase}/api/threads`, { method: "POST" });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error ?? `HTTP ${r.status}`);
  return data;
}

export async function fetchThreadState(
  threadId: string,
): Promise<{ values: Record<string, unknown> }> {
  const r = await fetch(`${apiBase}/api/threads/${encodeURIComponent(threadId)}/state`);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error ?? `HTTP ${r.status}`);
  return data;
}

/** POST /api/runs/stream — parse SSE. Returns whether the graph paused on interrupt. */
export async function streamRun(
  body: {
    threadId: string;
    input?: Record<string, unknown> | null;
    command?: { resume?: unknown };
  },
  onEvent: (ev: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<{ interrupted: boolean }> {
  const r = await fetch(`${apiBase}/api/runs/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || `Stream failed: ${r.status}`);
  }

  const reader = r.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";
  let interrupted = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const block of parts) {
      const line = block.trim();
      if (!line.startsWith("data:")) continue;
      const json = line.slice(5).trim();
      try {
        const ev = JSON.parse(json) as StreamEvent;
        onEvent(ev);
        if (ev.type === "interrupt") interrupted = true;
      } catch {
        /* ignore */
      }
    }
  }

  return { interrupted };
}
