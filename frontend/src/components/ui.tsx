// Shared presentational pieces. Each one encodes a rule from the spec, so they
// live here rather than being re-improvised per page.

import type { ReactNode } from "react";
import type { Strength } from "../lib/api";

export function Card({
  title,
  subtitle,
  children,
  right,
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  right?: ReactNode;
}) {
  return (
    <section className="rounded-xl border border-line bg-ink-800/60 backdropentity">
      {(title || right) && (
        <header className="flex items-start justify-between gap-4 border-b border-line px-5 py-3.5">
          <div>
            {title && <h2 className="text-sm font-semibold tracking-wide text-slate-200">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      <div className="px-5 py-4">{children}</div>
    </section>
  );
}

/**
 * Retrieval strength badge.
 *
 * Three values, never a percentage. The tooltip states what it is, because a
 * coloured badge invites the reader to treat it as a confidence score and it is
 * not one.
 */
export function StrengthBadge({
  strength,
  cosine,
  calibrated,
}: {
  strength: Strength;
  cosine?: number | null;
  calibrated?: boolean;
}) {
  const styles: Record<Strength, string> = {
    strong: "bg-emerald-500/10 text-emerald-300 ring-emerald-500/30",
    mixed: "bg-amber-500/10 text-amber-300 ring-amber-500/30",
    weak: "bg-rose-500/10 text-rose-300 ring-rose-500/30",
  };
  return (
    <span
      title="Backend-computed from raw dense cosine and rank agreement. Not a probability, not a similarity percentage, and not produced by a language model."
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wider ring-1 ${styles[strength]}`}
    >
      {strength}
      {typeof cosine === "number" && (
        <span className="font-mono text-[11px] font-normal opacity-70">cos {cosine.toFixed(3)}</span>
      )}
      {calibrated === false && (
        <span className="text-[10px] font-normal opacity-60">uncalibrated</span>
      )}
    </span>
  );
}

export function ModeBadge({ mode }: { mode: string }) {
  const label: Record<string, string> = {
    deterministic: "No model was called. Retrieved resolution notes rendered with citations.",
    llm: "A language model composed these steps from the retrieved evidence.",
    evidence_only: "Retrieval worked, but no case carried a usable resolution note.",
    disabled: "This capability is switched off by the manifest.",
  };
  return (
    <span
      title={label[mode] ?? mode}
      className="rounded-md bg-ink-700 px-2.5 py-1 font-mono text-[11px] text-slate-300 ring-1 ring-line"
    >
      mode: {mode}
    </span>
  );
}

/** Per-step citation chip. Clicking scrolls to the evidence it points at. */
export function CitationChip({ id }: { id: string }) {
  return (
    <a
      href={`#evidence-${id}`}
      className="rounded bg-sky-500/10 px-1.5 py-0.5 font-mono text-[11px] text-sky-300 ring-1 ring-sky-500/25 transition hover:bg-sky-500/20"
    >
      {id}
    </a>
  );
}

export function Banner({
  tone,
  title,
  children,
}: {
  tone: "warn" | "info" | "danger";
  title: string;
  children?: ReactNode;
}) {
  const styles = {
    warn: "border-amber-500/30 bg-amber-500/[0.07] text-amber-200",
    info: "border-sky-500/30 bg-sky-500/[0.07] text-sky-200",
    danger: "border-rose-500/30 bg-rose-500/[0.07] text-rose-200",
  }[tone];
  return (
    <div className={`rounded-lg border px-4 py-3 text-sm ${styles}`}>
      <p className="font-semibold">{title}</p>
      {children && <div className="mt-1 text-[13px] opacity-90">{children}</div>}
    </div>
  );
}

/**
 * What a disabled capability renders. Never an empty chart, never a zero —
 * a zero implies a measurement that did not happen.
 */
export function CapabilityDisabled({ name, reason }: { name: string; reason: string | null }) {
  return (
    <div className="rounded-xl border border-line bg-ink-800/60 px-6 py-10 text-center">
      <p className="font-mono text-xs uppercase tracking-widest text-slate-500">
        {name} — disabled
      </p>
      <p className="mx-auto mt-3 max-w-xl text-sm text-slate-300">
        {reason ?? "No reason was recorded."}
      </p>
      <p className="mx-auto mt-4 max-w-xl text-xs leading-relaxed text-slate-500">
        This capability is switched off because the dataset or configuration does not support it.
        No placeholder result is shown in its place — a zero here would imply a measurement that
        did not happen.
      </p>
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-sm text-slate-400">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-slate-600 border-t-slate-300" />
      {label ?? "Loading…"}
    </div>
  );
}

export function KeyValue({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-line/60 py-1.5 last:border-0">
      <span className="text-xs text-slate-500">{k}</span>
      <span className="text-right font-mono text-xs text-slate-300">{v}</span>
    </div>
  );
}
