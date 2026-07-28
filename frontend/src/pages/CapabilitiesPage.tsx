import { useCapabilities } from "../lib/capabilities";
import { Banner, Card, KeyValue, Spinner } from "../components/ui";
import type { CapabilityBlock } from "../lib/api";

const SUBSYSTEMS = [
  "analytics",
  "retrieval",
  "resolution_generation",
  "analyst_agent",
  "clustering",
  "risk",
  "llm_provider",
  "ai_ops",
] as const;

/**
 * The manifest page. This is the project's thesis rendered as a screen: what the
 * system is allowed to claim, and why each disabled thing is disabled.
 */
export default function CapabilitiesPage() {
  const { caps, loading, error, reload } = useCapabilities();

  if (loading) return <Card><Spinner label="Reading the capability manifest…" /></Card>;
  if (error)
    return (
      <Banner tone="danger" title="Could not read the manifest">
        <code className="font-mono text-xs">{error}</code>
      </Banner>
    );
  if (!caps) return null;

  const on = SUBSYSTEMS.filter((s) => (caps[s] as CapabilityBlock)?.enabled);
  const off = SUBSYSTEMS.filter((s) => !(caps[s] as CapabilityBlock)?.enabled);

  return (
    <div className="space-y-5">
      <Card
        title="Capability manifest"
        subtitle="Generated from an audit of the dataset. It gates every route and pipeline stage."
        right={
          <button
            onClick={reload}
            className="rounded border border-line px-2.5 py-1 text-xs text-slate-400 hover:text-slate-200"
          >
            Reload
          </button>
        }
      >
        <p className="mb-4 text-[13px] leading-relaxed text-slate-400">
          A capability the data cannot support switches itself off and states why. It never returns
          a zero — a zero would imply a measurement that did not happen.
        </p>

        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <p className="mb-2 text-[11px] uppercase tracking-widest text-emerald-400/70">
              Enabled ({on.length})
            </p>
            <div className="space-y-1.5">
              {on.map((s) => (
                <div
                  key={s}
                  className="rounded-md border border-emerald-500/20 bg-emerald-500/[0.05] px-3 py-2 font-mono text-xs text-emerald-200"
                >
                  {s}
                </div>
              ))}
            </div>
          </div>
          <div>
            <p className="mb-2 text-[11px] uppercase tracking-widest text-slate-500">
              Disabled ({off.length})
            </p>
            <div className="space-y-1.5">
              {off.map((s) => {
                const b = caps[s] as CapabilityBlock;
                return (
                  <div key={s} className="rounded-md border border-line bg-ink-900/50 px-3 py-2">
                    <p className="font-mono text-xs text-slate-400">{s}</p>
                    <p className="mt-1 text-[12px] leading-relaxed text-slate-500">{b?.reason}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </Card>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Retrieval" subtitle="Corpus and evaluation status">
          <KeyValue k="corpus served" v={caps.retrieval.corpus_size_served ?? "—"} />
          <KeyValue k="corpus audited" v={caps.retrieval.corpus_size_audited ?? "—"} />
          <KeyValue k="evaluation status" v={caps.retrieval.evaluation_status ?? "—"} />
          <KeyValue
            k="generation modes"
            v={caps.resolution_generation.available_modes?.join(", ") ?? "—"}
          />
          {caps.retrieval.evaluation_status === "manual_set_not_yet_labeled" && (
            <p className="mt-3 rounded-lg bg-ink-900/70 px-3 py-2.5 text-[12px] leading-relaxed text-slate-500">
              No retrieval quality figure is shown because the human-graded query set has not been
              built. A fabricated Hit@K would be worse than an absent one.
            </p>
          )}
          {caps.retrieval.relaxation && (
            <p className="mt-3 text-[12px] leading-relaxed text-amber-200/70">
              Relaxation: {caps.retrieval.relaxation}
            </p>
          )}
        </Card>

        <Card title="Analytics metrics" subtitle="Each metric is gated independently">
          <p className="mb-2 text-[11px] uppercase tracking-widest text-emerald-400/70">Available</p>
          <div className="mb-4 flex flex-wrap gap-1.5">
            {(caps.analytics.available_metrics ?? []).map((m) => (
              <span
                key={m}
                className="rounded bg-emerald-500/10 px-2 py-0.5 font-mono text-[11px] text-emerald-300"
              >
                {m}
              </span>
            ))}
          </div>
          <p className="mb-2 text-[11px] uppercase tracking-widest text-slate-500">
            Unavailable, with reasons
          </p>
          <div className="space-y-1.5">
            {Object.entries(caps.analytics.unavailable_metrics ?? {}).map(([m, why]) => (
              <div key={m} className="text-[12px] leading-relaxed">
                <span className="font-mono text-slate-400">{m}</span>
                <span className="text-slate-600"> — {why}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card title="Provenance" subtitle="Every number on this screen traces to one file">
        <KeyValue k="rows in canonical dataset" v={caps.row_count ?? "—"} />
        <KeyValue k="source file sha256" v={caps.data_hash?.slice(0, 24) ?? "—"} />
        <KeyValue k="manifest generated" v={caps.generated_at?.slice(0, 19).replace("T", " ") ?? "—"} />
        <KeyValue
          k="llm provider"
          v={`${caps.llm_provider.provider ?? "none"}${
            caps.llm_provider.enabled ? "" : " (disabled)"
          }`}
        />
        <KeyValue
          k="tokens / cost"
          v={caps.llm_provider.enabled ? "measured per request" : "not_applicable"}
        />
      </Card>
    </div>
  );
}
