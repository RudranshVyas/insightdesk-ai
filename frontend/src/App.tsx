import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import SupportBriefPage from "./pages/SupportBriefPage";
import CapabilitiesPage from "./pages/CapabilitiesPage";
import { useCapabilities } from "./lib/capabilities";
import type { CapabilityBlock } from "./lib/api";

/**
 * Capability-aware navigation: a link whose capability is off is rendered
 * disabled with the reason as its tooltip, rather than leading to a page that
 * has nothing honest to show.
 */
const NAV = [
  { to: "/brief", label: "Support Brief", capability: "retrieval" as const },
  { to: "/capabilities", label: "Capabilities", capability: null },
];

export default function App() {
  const { caps, error } = useCapabilities();

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 border-b border-line bg-ink-900/85 backdrop-blur">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-6 gap-y-2 px-6 py-3.5">
          <div className="flex items-baseline gap-2.5">
            <span className="text-[15px] font-semibold tracking-tight text-slate-100">
              InsightDesk AI
            </span>
            <span className="text-[11px] text-slate-600">support intelligence</span>
          </div>

          <nav className="flex items-center gap-1">
            {NAV.map((item) => {
              const block = item.capability
                ? (caps?.[item.capability] as CapabilityBlock | undefined)
                : undefined;
              const off = item.capability ? !block?.enabled : false;
              return off ? (
                <span
                  key={item.to}
                  title={block?.reason ?? "capability disabled"}
                  className="cursor-not-allowed rounded-md px-3 py-1.5 text-sm text-slate-700 line-through"
                >
                  {item.label}
                </span>
              ) : (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    `rounded-md px-3 py-1.5 text-sm transition ${
                      isActive
                        ? "bg-ink-700 text-slate-100"
                        : "text-slate-400 hover:text-slate-200"
                    }`
                  }
                >
                  {item.label}
                </NavLink>
              );
            })}
          </nav>

          <div className="ml-auto flex items-center gap-2 text-[11px]">
            {caps?.retrieval?.enabled && (
              <span className="font-mono text-slate-600">
                {caps.retrieval.corpus_size_served?.toLocaleString()} cases
              </span>
            )}
            <span
              title={
                caps?.llm_provider?.enabled
                  ? "A provider is configured."
                  : "No provider configured. The whole application works without one."
              }
              className={`rounded px-2 py-0.5 font-mono ring-1 ${
                caps?.llm_provider?.enabled
                  ? "bg-sky-500/10 text-sky-300 ring-sky-500/25"
                  : "bg-ink-700 text-slate-400 ring-line"
              }`}
            >
              {caps?.llm_provider?.enabled ? "llm" : "no key needed"}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-6">
        {error && (
          <div className="mb-5 rounded-lg border border-rose-500/30 bg-rose-500/[0.07] px-4 py-3 text-sm text-rose-200">
            <p className="font-semibold">Cannot reach the API</p>
            <p className="mt-1 text-[13px] opacity-90">
              Start it with{" "}
              <code className="font-mono text-xs">
                uvicorn backend.app.main:app --port 8000
              </code>
              . <span className="opacity-70">({error})</span>
            </p>
          </div>
        )}

        <Routes>
          <Route path="/" element={<Navigate to="/brief" replace />} />
          <Route path="/brief" element={<SupportBriefPage />} />
          <Route path="/capabilities" element={<CapabilitiesPage />} />
          <Route
            path="*"
            element={<p className="text-sm text-slate-500">No such page.</p>}
          />
        </Routes>

        <footer className="mt-10 border-t border-line pt-4 text-[11px] leading-relaxed text-slate-600">
          Resolution suggestions are historical evidence, never guaranteed resolutions. Retrieval
          strength is computed by the backend from raw cosine and rank agreement — it is not a
          probability and not produced by a language model. Fused ranking scores are never shown as
          similarity percentages.
        </footer>
      </main>
    </div>
  );
}
