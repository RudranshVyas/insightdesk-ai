// Capability context.
//
// The manifest is fetched once and every page reads it. A page whose capability
// is disabled renders the reason instead of the page — it never renders an empty
// chart, a zero, or a "no data yet" placeholder, because those all imply a
// measurement that did not happen.

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api, type Capabilities } from "./api";

interface CapabilityState {
  caps: Capabilities | null;
  loading: boolean;
  error: string | null;
  reload: () => void;
}

const Ctx = createContext<CapabilityState>({
  caps: null,
  loading: true,
  error: null,
  reload: () => {},
});

export function CapabilityProvider({ children }: { children: ReactNode }) {
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .capabilities()
      .then((c) => !cancelled && (setCaps(c), setError(null)))
      .catch((e: Error) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [nonce]);

  return (
    <Ctx.Provider value={{ caps, loading, error, reload: () => setNonce((n) => n + 1) }}>
      {children}
    </Ctx.Provider>
  );
}

export const useCapabilities = () => useContext(Ctx);

/** Returns [enabled, reason]. Unknown or unloaded capabilities are treated as off. */
export function useCapability(name: keyof Capabilities): [boolean, string | null] {
  const { caps } = useCapabilities();
  const block = caps?.[name] as { enabled?: boolean; reason?: string | null } | undefined;
  if (!block) return [false, "capability manifest has not loaded"];
  return [Boolean(block.enabled), block.reason ?? null];
}
