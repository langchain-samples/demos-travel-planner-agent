import { useCallback, useEffect, useRef, useState } from "react";

import {
  createThread,
  fetchHealth,
  fetchThreadState,
  streamRun,
  type StreamEvent,
} from "./api";
import { MarkdownMessage } from "./MarkdownMessage";
import { TripOutputCard } from "./TripOutputCard";

type Theme = "dark" | "light";

type ChatRow =
  | { id: string; role: "user"; text: string }
  | { id: string; role: "assistant"; text: string }
  | {
      id: string;
      role: "trip_output";
      weather?: string;
      travelLeg?: string;
      agenda?: string;
    };

function pickStr(v: unknown): string | undefined {
  if (typeof v !== "string") return undefined;
  const t = v.trim();
  return t || undefined;
}

const LOGO = {
  dark: "/branding/logo-light.png",
  light: "/branding/logo-dark.png",
} as const;

function uid(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export default function App() {
  const [theme, setTheme] = useState<Theme>(() => {
    if (typeof localStorage === "undefined") return "dark";
    return (localStorage.getItem("travel-planner-theme") as Theme) || "dark";
  });
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [rows, setRows] = useState<ChatRow[]>([]);
  const [nodes, setNodes] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [awaitingResume, setAwaitingResume] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("travel-planner-theme", theme);
  }, [theme]);

  useEffect(() => {
    fetchHealth()
      .then((h) => setBackendOk(h.deploymentConfigured))
      .catch(() => setBackendOk(false));
  }, []);

  const appendAssistant = useCallback((text: string) => {
    setRows((r) => [...r, { id: uid(), role: "assistant", text }]);
  }, []);

  const appendUser = useCallback((text: string) => {
    setRows((r) => [...r, { id: uid(), role: "user", text }]);
  }, []);

  const runStream = useCallback(
    async (opts: {
      threadId: string;
      input?: Record<string, unknown> | null;
      command?: { resume?: unknown };
    }) => {
      setBusy(true);
      setError(null);
      setNodes([]);
      // During a run the graph is not waiting for resume; avoids stale true.
      setAwaitingResume(false);
      abortRef.current?.abort();
      abortRef.current = new AbortController();

      try {
        const { interrupted } = await streamRun(
          opts,
          (ev: StreamEvent) => {
            if (ev.type === "node") {
              setNodes((n) => (n.includes(ev.name) ? n : [...n, ev.name]));
            } else if (ev.type === "interrupt") {
              const text =
                typeof ev.value === "string"
                  ? ev.value
                  : JSON.stringify(ev.value, null, 2);
              appendAssistant(text);
              // awaitingResume is set from streamRun's return value so it isn't
              // overwritten if "done" is handled in the same synchronous batch.
            } else if (ev.type === "error") {
              setError(ev.message);
            }
            // "done" does not toggle awaitingResume — derived from `interrupted` below.
          },
          abortRef.current.signal,
        );

        setAwaitingResume(interrupted);

        if (!interrupted) {
          const st = await fetchThreadState(opts.threadId);
          const vals = st.values ?? {};
          const weather = pickStr(vals.weather_summary);
          const travelLeg = pickStr(vals.travel_leg_summary);
          const agenda = pickStr(vals.agenda);
          if (weather || travelLeg || agenda) {
            setRows((r) => [
              ...r,
              {
                id: uid(),
                role: "trip_output",
                weather,
                travelLeg,
                agenda,
              },
            ]);
          }
        }
      } catch (e) {
        if ((e as Error).name === "AbortError") {
          setAwaitingResume(false);
          return;
        }
        setError(e instanceof Error ? e.message : String(e));
        setAwaitingResume(false);
      } finally {
        setBusy(false);
        abortRef.current = null;
      }
    },
    [appendAssistant],
  );

  const startNewTrip = useCallback(async () => {
    setError(null);
    setRows([]);
    setNodes([]);
    setAwaitingResume(false);
    setInput("");
    setBusy(true);
    try {
      const { threadId: tid } = await createThread();
      setThreadId(tid);
      await runStream({ threadId: tid, input: {} });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [runStream]);

  const sendUser = useCallback(async () => {
    const text = input.trim();
    if (!text || !threadId || busy) return;
    appendUser(text);
    setInput("");
    await runStream({ threadId, command: { resume: text } });
  }, [appendUser, input, threadId, busy, runStream]);

  const isDark = theme === "dark";

  return (
    <div
      className={
        isDark
          ? "flex h-screen bg-black text-zinc-100"
          : "flex h-screen bg-zinc-50 text-zinc-900"
      }
    >
      <aside
        className={
          isDark
            ? "flex w-64 shrink-0 flex-col border-r border-zinc-800 bg-[#0b0b0b]"
            : "flex w-64 shrink-0 flex-col border-r border-zinc-200 bg-white"
        }
      >
        <div className="flex items-center gap-2 border-b border-zinc-800/50 p-4 dark:border-zinc-800">
          <img
            src={isDark ? LOGO.dark : LOGO.light}
            alt=""
            className="h-8 w-8 object-contain"
          />
          <span className="font-semibold tracking-tight">Travel Planner</span>
        </div>

        <div className="flex flex-1 flex-col gap-2 p-3">
          <button
            type="button"
            onClick={startNewTrip}
            disabled={busy || backendOk === false}
            className={
              isDark
                ? "rounded-lg bg-blue-600 px-3 py-2.5 text-sm font-medium text-white transition hover:bg-blue-500 disabled:opacity-40"
                : "rounded-lg bg-blue-600 px-3 py-2.5 text-sm font-medium text-white transition hover:bg-blue-500 disabled:opacity-40"
            }
          >
            New trip
          </button>
          <p
            className={
              isDark ? "px-1 text-xs text-zinc-500" : "px-1 text-xs text-zinc-500"
            }
          >
            Starts the LangGraph agent with empty input; destination and dates are
            asked in chat first.
          </p>
          {threadId ? (
            <p
              className={
                isDark
                  ? "mt-2 break-all px-1 font-mono text-[10px] text-zinc-600"
                  : "mt-2 break-all px-1 font-mono text-[10px] text-zinc-400"
              }
            >
              thread: {threadId.slice(0, 8)}…
            </p>
          ) : null}
        </div>

        <div
          className={
            isDark
              ? "mt-auto space-y-1 border-t border-zinc-800 p-3 text-xs text-zinc-500"
              : "mt-auto space-y-1 border-t border-zinc-200 p-3 text-xs text-zinc-500"
          }
        >
          <button
            type="button"
            onClick={() => setTheme(isDark ? "light" : "dark")}
            className={
              isDark
                ? "w-full rounded-md px-2 py-1.5 text-left hover:bg-zinc-800"
                : "w-full rounded-md px-2 py-1.5 text-left hover:bg-zinc-100"
            }
          >
            {isDark ? "Light mode" : "Dark mode"}
          </button>
          <a
            href="https://docs.langchain.com/langgraph-platform"
            target="_blank"
            rel="noreferrer"
            className="block rounded-md px-2 py-1.5 text-blue-500 hover:underline"
          >
            LangGraph Platform docs
          </a>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <header
          className={
            isDark
              ? "flex items-center justify-between border-b border-zinc-800 px-6 py-3"
              : "flex items-center justify-between border-b border-zinc-200 px-6 py-3"
          }
        >
          <h1 className="text-sm font-medium text-zinc-400">
            Plan a trip with human-in-the-loop tools
          </h1>
          {backendOk === false ? (
            <span className="text-xs text-amber-500">
              Configure DEPLOYMENT_URL + LANGSMITH_API_KEY in repo .env and run{" "}
              <code className="rounded bg-zinc-800 px-1">npm run dev</code>
            </span>
          ) : null}
        </header>

        <div
          className={`scroll-dark flex-1 overflow-y-auto px-4 py-6 md:px-8 ${
            isDark ? "" : ""
          }`}
        >
          <div className="mx-auto max-w-3xl space-y-6">
            {rows.length === 0 && !busy ? (
              <div
                className={
                  isDark
                    ? "rounded-xl border border-zinc-800 bg-[#121212] p-8 text-center text-zinc-400"
                    : "rounded-xl border border-zinc-200 bg-white p-8 text-center text-zinc-600"
                }
              >
                <p
                  className={
                    isDark ? "mb-4 text-lg text-zinc-200" : "mb-4 text-lg text-zinc-800"
                  }
                >
                  Ready when you are
                </p>
                <p className="mb-6 text-sm">
                  Connects to your LangGraph deployment via the local API (reads{" "}
                  <code className="text-blue-400">DEPLOYMENT_URL</code> from{" "}
                  <code className="text-blue-400">.env</code>).
                </p>
                <button
                  type="button"
                  onClick={startNewTrip}
                  disabled={backendOk === false}
                  className="rounded-full bg-blue-600 px-6 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-40"
                >
                  Start a new trip
                </button>
              </div>
            ) : null}

            {rows.map((row) =>
              row.role === "trip_output" ? (
                <article key={row.id} className="mr-8 flex justify-start">
                  <TripOutputCard
                    isDark={isDark}
                    weather={row.weather}
                    travelLeg={row.travelLeg}
                    agenda={row.agenda}
                  />
                </article>
              ) : (
                <article
                  key={row.id}
                  className={
                    row.role === "user"
                      ? "ml-8 flex justify-end"
                      : "mr-8 flex justify-start"
                  }
                >
                  <div
                    className={
                      row.role === "user"
                        ? isDark
                          ? "max-w-[90%] rounded-2xl rounded-br-md bg-blue-600 px-4 py-3 text-sm text-white"
                          : "max-w-[90%] rounded-2xl rounded-br-md bg-blue-600 px-4 py-3 text-sm text-white"
                        : isDark
                          ? "max-w-[95%] rounded-xl border border-zinc-800 bg-[#121212] px-4 py-4 text-sm text-zinc-200"
                          : "max-w-[95%] rounded-xl border border-zinc-200 bg-white px-4 py-4 text-sm text-zinc-800"
                    }
                  >
                    {row.role === "user" ? (
                      row.text
                    ) : (
                      <MarkdownMessage isDark={isDark}>{row.text}</MarkdownMessage>
                    )}
                  </div>
                </article>
              ),
            )}

            {busy && nodes.length > 0 ? (
              <p
                className={
                  isDark ? "text-center text-xs text-zinc-500" : "text-center text-xs text-zinc-500"
                }
              >
                Running: {nodes.slice(-4).join(" → ")}
                {nodes.length > 4 ? "…" : ""}
              </p>
            ) : null}

            {error ? (
              <p
                className={
                  isDark
                    ? "rounded-lg border border-red-900/50 bg-red-950/40 px-4 py-3 text-sm text-red-300"
                    : "rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
                }
              >
                {error}
              </p>
            ) : null}
          </div>
        </div>

        <div
          className={
            isDark
              ? "border-t border-zinc-800 bg-black px-4 py-4 md:px-8"
              : "border-t border-zinc-200 bg-zinc-50 px-4 py-4 md:px-8"
          }
        >
          <div className="mx-auto max-w-3xl">
            <div
              className={
                isDark
                  ? "flex items-center gap-2 rounded-full border border-zinc-700 bg-[#0b0b0b] px-4 py-2 shadow-lg shadow-black/20 focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500/30"
                  : "flex items-center gap-2 rounded-full border border-zinc-300 bg-white px-4 py-2 shadow-md focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500/30"
              }
            >
              <span className="text-zinc-500">+</span>
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void sendUser();
                  }
                }}
                disabled={!threadId || !awaitingResume || busy}
                placeholder={
                  !threadId
                    ? "Click “New trip” or “Start” first…"
                    : !awaitingResume
                      ? "Wait for the assistant prompt…"
                      : "Reply to the assistant…"
                }
                className={
                  isDark
                    ? "min-h-10 flex-1 bg-transparent text-sm text-zinc-100 outline-none placeholder:text-zinc-600 disabled:opacity-50"
                    : "min-h-10 flex-1 bg-transparent text-sm text-zinc-900 outline-none placeholder:text-zinc-400 disabled:opacity-50"
                }
              />
              <button
                type="button"
                onClick={() => void sendUser()}
                disabled={!threadId || !awaitingResume || busy || !input.trim()}
                className="rounded-full bg-blue-600 px-4 py-2 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-40"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
