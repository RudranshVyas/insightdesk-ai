// Plain-language layer.
//
// The API speaks precisely: `mixed`, `cos 0.574`, "Redacted 1 PII item(s) from
// the query before use: ['card']". That precision is correct and is preserved
// verbatim under "technical details" on every screen.
//
// It is not, however, what a support analyst with a customer waiting needs to
// read. This module translates. The rule it follows: **simplify the wording,
// never the meaning.** No mapping here upgrades a caveat into reassurance.

import type { Mode, Strength } from "./api";

export interface StrengthCopy {
  label: string;
  blurb: string;
  tone: "good" | "caution" | "halt";
}

export const STRENGTH: Record<Strength, StrengthCopy> = {
  strong: {
    label: "Close match",
    blurb: "Several past tickets describe closely similar problems.",
    tone: "good",
  },
  mixed: {
    label: "Partial match",
    blurb:
      "Past tickets are related but not a close match. Read them before acting.",
    tone: "caution",
  },
  weak: {
    label: "No close match",
    blurb:
      "Nothing in the past tickets is similar enough to suggest steps from. Nothing is suggested.",
    tone: "halt",
  },
};

export const MODE: Record<Mode, { label: string; blurb: string }> = {
  deterministic: {
    label: "From past tickets",
    blurb:
      "Support's recorded reply on each matching ticket, shown as written. No AI wrote them.",
  },
  llm: {
    label: "AI summary",
    blurb:
      "An AI wrote these steps from the matching tickets. Every step cites the ticket it came from.",
  },
  evidence_only: {
    label: "Similar tickets only",
    blurb:
      "These tickets match, but the data records only how support replied — not what finally fixed the problem.",
  },
  disabled: { label: "Unavailable", blurb: "This feature is switched off." },
};

/**
 * Rewrite a pipeline warning as a sentence an analyst can act on.
 *
 * Falls through to the original text when nothing matches — an unrecognised
 * warning must still be shown, not swallowed.
 */
export function humanize(warning: string): { text: string; kind: "privacy" | "safety" | "quality" } {
  const w = warning.toLowerCase();

  if (w.includes("redacted") && w.includes("pii")) {
    const kinds = [...warning.matchAll(/'([a-z_]+)'/g)].map((m) => m[1]);
    const named = kinds.length ? ` (${kinds.join(", ")})` : "";
    return {
      text: `Personal details in your text${named} were removed before searching. They were never stored or sent anywhere.`,
      kind: "privacy",
    };
  }

  if (w.includes("instruction-like text")) {
    return {
      text: "This text tries to give the system instructions. It was treated as part of the customer's message and ignored as a command.",
      kind: "safety",
    };
  }

  if (w.includes("injection") && w.includes("evidence")) {
    return {
      text: "One of the matching tickets contains text that tries to give instructions. Read that ticket with care.",
      kind: "safety",
    };
  }

  if (w.includes("strength is mixed")) {
    return {
      text: "The match is only partial, so this needs a human check before you act on it.",
      kind: "quality",
    };
  }

  if (w.includes("strength is weak")) {
    return {
      text: "Nothing matched closely enough to suggest steps.",
      kind: "quality",
    };
  }

  if (w.includes("template group")) {
    const n = warning.match(/Dropped (\d+)/)?.[1];
    return {
      text: `${n ?? "Some"} near-identical duplicate ticket${n === "1" ? " was" : "s were"} hidden. Repeats of one ticket look like agreement without being it.`,
      kind: "quality",
    };
  }

  if (w.includes("citation") && w.includes("not in the evidence set")) {
    return {
      text: "A suggested step referred to a ticket that does not exist. It was removed.",
      kind: "safety",
    };
  }

  if (w.includes("no valid citation")) {
    return {
      text: "A suggested step had no real ticket behind it and was removed.",
      kind: "safety",
    };
  }

  if (w.includes("certainty language")) {
    return {
      text: "The wording promised a guaranteed outcome. Past tickets cannot guarantee anything here.",
      kind: "quality",
    };
  }

  if (w.includes("deterministic")) {
    return {
      text: "The AI step was unavailable, so the recorded resolutions are shown as written instead.",
      kind: "quality",
    };
  }

  if (w.includes("unusable resolution notes")) {
    const n = warning.match(/Dropped (\d+)/)?.[1];
    return {
      text: `${n ?? "Some"} matching ticket${n === "1" ? "" : "s"} had no record of what was done, so ${n === "1" ? "it was" : "they were"} left out.`,
      kind: "quality",
    };
  }

  if (w.includes("truncated")) {
    return { text: "Your text was long and was shortened before searching.", kind: "quality" };
  }

  return { text: warning, kind: "quality" };
}

/** Strip the boilerplate the deterministic renderer adds, leaving the resolution. */
export function stripStepPreamble(text: string): string {
  return text.replace(/^In ticket \S+ \([^)]*\), support resolved a similar report as follows:\s*/i, "");
}

/** Placeholder tokens from the source data read as noise to an analyst. */
export function tidy(text: string): string {
  return text
    .replace(/<(name|tel_num|acc_num|email|address|datetime|company|url)>/gi, "—")
    .replace(/\\n/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();
}
