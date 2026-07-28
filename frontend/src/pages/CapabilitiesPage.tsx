import { useCapabilities } from "../lib/capabilities";
import { Loading, Notice, Panel, Row, SectionLabel, TechnicalDetails } from "../components/ui";
import type { CapabilityBlock } from "../lib/api";

/** Plain names for machine keys, so this page reads as prose rather than schema. */
const NAMES: Record<string, { label: string; what: string }> = {
  analytics: { label: "Ticket statistics", what: "Counts and breakdowns across tickets." },
  retrieval: { label: "Finding similar tickets", what: "Searching past tickets by meaning and by keyword." },
  resolution_generation: { label: "Showing what resolved them", what: "Surfacing the recorded fix for each match." },
  analyst_agent: { label: "Ask-a-question assistant", what: "Answering compound questions across the data." },
  clustering: { label: "Recurring issue groups", what: "Grouping tickets into recurring themes." },
  risk: { label: "Escalation risk scoring", what: "Predicting which tickets will escalate." },
  llm_provider: { label: "AI writing", what: "Using a language model to summarise resolutions." },
  ai_ops: { label: "Performance tracing", what: "Recording timing and behaviour of each request." },
};

const ORDER = Object.keys(NAMES);

export default function CapabilitiesPage() {
  const { caps, loading, error } = useCapabilities();

  if (loading)
    return (
      <Panel className="px-5 py-4">
        <Loading label="Reading what this system can do…" />
      </Panel>
    );

  if (error)
    return (
      <Notice tone="halt" title="Couldn't read the settings">
        <code className="font-mono text-[12px]">{error}</code>
      </Notice>
    );

  if (!caps) return null;

  const on = ORDER.filter((k) => (caps[k as keyof typeof caps] as CapabilityBlock)?.enabled);
  const off = ORDER.filter((k) => !(caps[k as keyof typeof caps] as CapabilityBlock)?.enabled);

  return (
    <div className="space-y-9">
      <header className="animate-rise">
        <h1 className="font-serif text-[34px] leading-[1.15] tracking-[-0.015em] text-ink sm:text-[40px]">
          What this system
          <span className="text-teal"> can and can't do</span>
        </h1>
        <p className="mt-3 max-w-reading text-[15.5px] leading-relaxed text-ink-soft">
          Each feature is switched on only if the ticket data actually supports it. Anything the
          data can't support turns itself off and says why — rather than showing you an empty chart
          or a zero.
        </p>
      </header>

      <section className="animate-rise stagger" style={{ ["--i" as string]: 1 }}>
        <SectionLabel>Working — {on.length}</SectionLabel>
        <div className="grid gap-3 sm:grid-cols-2">
          {on.map((k) => (
            <Panel key={k} className="px-4 py-3.5">
              <div className="flex items-start gap-2.5">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-good" />
                <div>
                  <p className="text-[14.5px] font-medium text-ink">{NAMES[k].label}</p>
                  <p className="mt-0.5 text-[13px] leading-relaxed text-ink-mute">{NAMES[k].what}</p>
                </div>
              </div>
            </Panel>
          ))}
        </div>
      </section>

      <section className="animate-rise stagger" style={{ ["--i" as string]: 2 }}>
        <SectionLabel>Switched off — {off.length}</SectionLabel>
        <div className="space-y-3">
          {off.map((k) => {
            const b = caps[k as keyof typeof caps] as CapabilityBlock;
            return (
              <Panel key={k} className="px-4 py-3.5">
                <div className="flex items-start gap-2.5">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-ink-faint" />
                  <div>
                    <p className="text-[14.5px] font-medium text-ink-soft">{NAMES[k].label}</p>
                    <p className="mt-1 max-w-reading text-[13.5px] leading-relaxed text-ink-mute">
                      {b?.reason}
                    </p>
                  </div>
                </div>
              </Panel>
            );
          })}
        </div>
      </section>

      <section className="animate-rise stagger" style={{ ["--i" as string]: 3 }}>
        <SectionLabel>The ticket data behind it</SectionLabel>
        <Panel className="px-5 py-4">
          <div className="grid gap-x-10 gap-y-1 sm:grid-cols-2">
            <Row k="Tickets searched" v={caps.retrieval.corpus_size_served?.toLocaleString() ?? "—"} />
            <Row k="Tickets audited" v={caps.retrieval.corpus_size_audited?.toLocaleString() ?? "—"} />
            <Row k="Total rows in dataset" v={caps.row_count?.toLocaleString() ?? "—"} />
            <Row k="AI writing" v={caps.llm_provider.enabled ? caps.llm_provider.provider ?? "on" : "not used"} />
          </div>

          {caps.retrieval.evaluation_status === "manual_set_not_yet_labeled" && (
            <p className="mt-4 rounded-xl border border-rule bg-paper-sunk/60 px-4 py-3 text-[13.5px] leading-relaxed text-ink-mute">
              <span className="font-medium text-ink-soft">No accuracy score is shown.</span> Measuring
              how often the search returns the right ticket needs a set of questions graded by a
              person, and that hasn't been done yet. An invented accuracy figure would be worse than
              none.
            </p>
          )}

          {caps.retrieval.relaxation && (
            <p className="mt-3 text-[13px] leading-relaxed text-ink-mute">{caps.retrieval.relaxation}</p>
          )}
        </Panel>
      </section>

      <section className="animate-rise stagger" style={{ ["--i" as string]: 4 }}>
        <SectionLabel>Statistics available</SectionLabel>
        <Panel className="px-5 py-4">
          <div className="flex flex-wrap gap-1.5">
            {(caps.analytics.available_metrics ?? []).map((m) => (
              <span
                key={m}
                className="rounded-full bg-good-bg px-2.5 py-1 font-mono text-[11.5px] text-good"
              >
                {m}
              </span>
            ))}
          </div>
          <div className="mt-4 space-y-1.5">
            {Object.entries(caps.analytics.unavailable_metrics ?? {}).map(([m, why]) => (
              <p key={m} className="text-[13px] leading-relaxed">
                <span className="font-mono text-[12px] text-ink-soft">{m}</span>
                <span className="text-ink-mute"> — {why}</span>
              </p>
            ))}
          </div>
        </Panel>
      </section>

      <TechnicalDetails summary="Provenance and versions">
        <Row k="source file sha256" v={caps.data_hash?.slice(0, 32) ?? "—"} />
        <Row k="manifest generated" v={caps.generated_at?.slice(0, 19).replace("T", " ") ?? "—"} />
        <Row k="retrieval evaluation status" v={caps.retrieval.evaluation_status ?? "—"} />
        <Row k="generation modes" v={caps.resolution_generation.available_modes?.join(", ") ?? "—"} />
        <Row k="tokens / cost" v={caps.llm_provider.enabled ? "measured per request" : "not_applicable"} />
        <p className="mt-3 border-t border-rule pt-3 text-[11.5px] leading-relaxed text-ink-faint">
          Every figure on this page traces to the one source file identified by the hash above. A
          different file produces different numbers and requires a new audit.
        </p>
      </TechnicalDetails>
    </div>
  );
}
