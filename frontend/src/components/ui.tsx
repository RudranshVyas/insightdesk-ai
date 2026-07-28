import type { ReactNode } from "react";
import type { Strength } from "../lib/api";
import { STRENGTH } from "../lib/plain";

const TONE = {
  good: "text-good bg-good-bg border-good/20",
  caution: "text-caution bg-caution-bg border-caution/20",
  halt: "text-halt bg-halt-bg border-halt/20",
} as const;

export function Panel({
  children,
  className = "",
  as: Tag = "section",
}: {
  children: ReactNode;
  className?: string;
  as?: "section" | "div" | "article";
}) {
  return (
    <Tag
      className={`rounded-2xl border border-rule bg-paper-raised shadow-card ${className}`}
    >
      {children}
    </Tag>
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <h2 className="mb-3 font-mono text-[11px] uppercase tracking-[0.14em] text-ink-mute">
      {children}
    </h2>
  );
}

/**
 * Match quality, in words.
 *
 * The number behind it is real and lives one disclosure away. It is not shown
 * here because a decimal invites reading as a confidence score, which it is not.
 */
export function MatchBadge({ strength }: { strength: Strength }) {
  const s = STRENGTH[strength];
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[13px] font-medium ${TONE[s.tone]}`}
    >
      <Dot tone={s.tone} />
      {s.label}
    </span>
  );
}

function Dot({ tone }: { tone: "good" | "caution" | "halt" }) {
  const fill = { good: "bg-good", caution: "bg-caution", halt: "bg-halt" }[tone];
  return <span className={`h-1.5 w-1.5 rounded-full ${fill}`} />;
}

export function Notice({
  tone,
  title,
  children,
}: {
  tone: "good" | "caution" | "halt";
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className={`rounded-xl border px-4 py-3.5 ${TONE[tone]}`}>
      <p className="text-[14px] font-semibold">{title}</p>
      {children && <div className="mt-1 text-[13.5px] leading-relaxed opacity-90">{children}</div>}
    </div>
  );
}

/** Everything precise, tucked away. Present on every screen, never in the way. */
export function TechnicalDetails({
  children,
  summary = "Technical details",
}: {
  children: ReactNode;
  summary?: string;
}) {
  return (
    <details className="group rounded-xl border border-rule bg-paper-sunk/60">
      <summary className="flex items-center gap-2 px-4 py-2.5 font-mono text-[11px] uppercase tracking-[0.12em] text-ink-mute transition hover:text-ink-soft">
        <svg
          viewBox="0 0 12 12"
          className="h-2.5 w-2.5 transition-transform group-open:rotate-90"
          aria-hidden
        >
          <path d="M4 2l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.6" />
        </svg>
        {summary}
      </summary>
      <div className="border-t border-rule px-4 py-3.5">{children}</div>
    </details>
  );
}

export function Row({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-6 border-b border-rule/70 py-1.5 last:border-0">
      <span className="text-[12.5px] text-ink-mute">{k}</span>
      <span className="text-right font-mono text-[12px] text-ink-soft">{v}</span>
    </div>
  );
}

export function TicketRef({ id }: { id: string }) {
  return (
    <a
      href={`#t-${id}`}
      className="rounded-md bg-teal-light px-1.5 py-0.5 font-mono text-[11.5px] text-teal transition hover:bg-teal hover:text-paper"
    >
      {id}
    </a>
  );
}

/** A disabled capability states why. Never an empty chart, never a zero. */
export function Unavailable({ name, reason }: { name: string; reason: string | null }) {
  return (
    <Panel className="px-8 py-14 text-center">
      <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-ink-faint">
        {name} — unavailable
      </p>
      <p className="mx-auto mt-4 max-w-reading font-serif text-[19px] leading-relaxed text-ink-soft">
        {reason ?? "No reason was recorded."}
      </p>
      <p className="mx-auto mt-5 max-w-reading text-[13px] leading-relaxed text-ink-mute">
        This is switched off because the data behind it does not support it. Nothing is shown in
        its place — a blank chart or a zero would suggest a measurement that never happened.
      </p>
    </Panel>
  );
}

export function Loading({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 text-[14px] text-ink-mute">
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-teal opacity-60" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-teal" />
      </span>
      {label}
    </div>
  );
}
