import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import SupportBriefPage from "./pages/SupportBriefPage";
import CapabilitiesPage from "./pages/CapabilitiesPage";
import { useCapabilities } from "./lib/capabilities";
import type { CapabilityBlock } from "./lib/api";

const NAV = [
  { to: "/find", label: "Find resolutions", capability: "retrieval" as const },
  { to: "/about", label: "What it can do", capability: null },
];

export default function App() {
  const { caps, error } = useCapabilities();

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-rule bg-paper/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-4xl items-center gap-6 px-6 py-4">
          <a href="/find" className="group flex items-baseline gap-2">
            <span className="font-serif text-[19px] font-semibold tracking-[-0.01em] text-ink">
              InsightDesk
            </span>
            <span className="h-1 w-1 rounded-full bg-teal transition group-hover:scale-150" />
          </a>

          <nav className="flex items-center gap-1">
            {NAV.map((item) => {
              const block = item.capability
                ? (caps?.[item.capability] as CapabilityBlock | undefined)
                : undefined;
              const off = item.capability ? caps != null && !block?.enabled : false;

              return off ? (
                <span
                  key={item.to}
                  title={block?.reason ?? "unavailable"}
                  className="cursor-not-allowed px-3 py-1.5 text-[14px] text-ink-faint line-through decoration-ink-faint/50"
                >
                  {item.label}
                </span>
              ) : (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    `relative px-3 py-1.5 text-[14px] transition ${
                      isActive ? "text-ink" : "text-ink-mute hover:text-ink-soft"
                    }`
                  }
                >
                  {({ isActive }) => (
                    <>
                      {item.label}
                      {isActive && (
                        <span className="absolute inset-x-3 -bottom-[17px] h-[2px] origin-left animate-sweep bg-teal" />
                      )}
                    </>
                  )}
                </NavLink>
              );
            })}
          </nav>

          {caps?.retrieval?.enabled && (
            <span className="ml-auto hidden text-[12.5px] text-ink-faint sm:block">
              {caps.retrieval.corpus_size_served?.toLocaleString()} resolved tickets
            </span>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-6 pb-24 pt-10">
        {error && (
          <div className="mb-8 rounded-xl border border-halt/20 bg-halt-bg px-4 py-3.5 text-halt">
            <p className="text-[14px] font-semibold">The service isn't running</p>
            <p className="mt-1 text-[13.5px] leading-relaxed opacity-90">
              Start it with{" "}
              <code className="rounded bg-halt/10 px-1.5 py-0.5 font-mono text-[12px]">
                uvicorn backend.app.main:app --port 8000
              </code>
            </p>
          </div>
        )}

        <Routes>
          <Route path="/" element={<Navigate to="/find" replace />} />
          <Route path="/find" element={<SupportBriefPage />} />
          <Route path="/about" element={<CapabilitiesPage />} />
          <Route
            path="*"
            element={<p className="text-[15px] text-ink-mute">That page doesn't exist.</p>}
          />
        </Routes>
      </main>

      <footer className="border-t border-rule">
        <div className="mx-auto max-w-4xl px-6 py-6">
          <p className="max-w-reading text-[12.5px] leading-relaxed text-ink-faint">
            Everything shown here comes from tickets that were already resolved. It is a record of
            what was done before, not advice and not a guarantee. A person decides what happens
            next.
          </p>
        </div>
      </footer>
    </div>
  );
}
