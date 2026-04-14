import type { ReactNode } from "react";

import { MarkdownMessage } from "./MarkdownMessage";

type Props = {
  isDark: boolean;
  weather?: string;
  travelLeg?: string;
  agenda?: string;
};

function CollapsibleSection({
  title,
  isDark,
  children,
}: {
  title: string;
  isDark: boolean;
  children: ReactNode;
}) {
  const shell =
    isDark
      ? "trip-details rounded-lg border border-zinc-800 bg-black/40"
      : "trip-details rounded-lg border border-zinc-200 bg-zinc-50/80";

  return (
    <details open className={shell}>
      <summary
        className={
          isDark
            ? "flex cursor-pointer list-none items-center gap-2 px-3 py-2.5 text-sm font-medium text-zinc-200 [&::-webkit-details-marker]:hidden"
            : "flex cursor-pointer list-none items-center gap-2 px-3 py-2.5 text-sm font-medium text-zinc-800 [&::-webkit-details-marker]:hidden"
        }
      >
        <span
          aria-hidden
          className={
            isDark
              ? "trip-details-chevron text-[10px] text-zinc-500"
              : "trip-details-chevron text-[10px] text-zinc-400"
          }
        >
          {"\u25B8"}
        </span>
        {title}
      </summary>
      <div
        className={
          isDark
            ? "border-t border-zinc-800 px-3 py-3 text-sm text-zinc-200"
            : "border-t border-zinc-200 px-3 py-3 text-sm text-zinc-800"
        }
      >
        {children}
      </div>
    </details>
  );
}

/** Weather, travel leg, and itinerary from completed graph state — each section collapsible. */
export function TripOutputCard({ isDark, weather, travelLeg, agenda }: Props) {
  const outer =
    isDark
      ? "max-w-[95%] rounded-xl border border-zinc-800 bg-[#121212] p-4"
      : "max-w-[95%] rounded-xl border border-zinc-200 bg-white p-4";

  return (
    <div className={outer}>
      <p
        className={
          isDark
            ? "mb-3 text-xs font-medium uppercase tracking-wide text-zinc-500"
            : "mb-3 text-xs font-medium uppercase tracking-wide text-zinc-500"
        }
      >
        Trip results
      </p>
      <div className="space-y-2">
        {weather ? (
          <CollapsibleSection title="Weather report" isDark={isDark}>
            <MarkdownMessage isDark={isDark}>{weather}</MarkdownMessage>
          </CollapsibleSection>
        ) : null}
        {travelLeg ? (
          <CollapsibleSection title="Getting there" isDark={isDark}>
            <MarkdownMessage isDark={isDark}>{travelLeg}</MarkdownMessage>
          </CollapsibleSection>
        ) : null}
        {agenda ? (
          <CollapsibleSection title="Itinerary" isDark={isDark}>
            <MarkdownMessage isDark={isDark}>{agenda}</MarkdownMessage>
          </CollapsibleSection>
        ) : null}
      </div>
    </div>
  );
}
