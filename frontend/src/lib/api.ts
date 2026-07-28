// Typed client for the InsightDesk API.
//
// The types here mirror backend/app/schemas/brief.py deliberately. Two fields
// that exist in other systems and must never appear here: a confidence value on
// anything the model produced, and a similarity percentage derived from the
// fused RRF score.

export type Mode = "deterministic" | "llm" | "evidence_only" | "disabled";
export type Strength = "strong" | "mixed" | "weak";
export type StageStatus = "ok" | "skipped" | "failed" | "degraded";

export interface CapabilityBlock {
  enabled: boolean;
  reason: string | null;
  [k: string]: unknown;
}

export interface Capabilities {
  available?: boolean;
  row_count?: number;
  data_hash?: string | null;
  generated_at?: string | null;
  analytics: CapabilityBlock & {
    available_metrics?: string[];
    unavailable_metrics?: Record<string, string>;
    denominators?: Record<string, number>;
  };
  retrieval: CapabilityBlock & {
    corpus_size_served?: number;
    corpus_size_audited?: number;
    evaluation_status?: string;
    note?: string | null;
    relaxation?: string | null;
  };
  resolution_generation: CapabilityBlock & { available_modes?: string[] };
  analyst_agent: CapabilityBlock;
  clustering: CapabilityBlock;
  risk: CapabilityBlock & { target?: string | null; target_kind?: string | null };
  llm_provider: CapabilityBlock & { provider?: string; model?: string | null };
  ai_ops: CapabilityBlock & { exporter?: string | null };
}

export interface SuggestedStep {
  text: string;
  citation_ticket_ids: string[];
}

export interface EvidenceTicket {
  ticket_id: string;
  issue_subject: string | null;
  issue_excerpt: string;
  resolution_notes: string | null;
  product_area: string | null;
  issue_type: string | null;
  dense_rank: number | null;
  lexical_rank: number | null;
  dense_cosine: number | null;
  fusion_rank: number | null;
  matched_metadata: Record<string, boolean>;
  template_group_id: number | null;
  injection_flags: string[];
}

export interface RetrievalStrengthDetail {
  strength: Strength;
  top_cosine: number | null;
  margin: number | null;
  candidates_above_floor: number;
  rank_agreement: number;
  calibrated: boolean;
  reasons: string[];
  note: string;
}

export interface StageTrace {
  name: string;
  status: StageStatus;
  latency_ms: number;
  summary: string;
  warnings: string[];
}

export interface VersionStamp {
  artifact: Record<string, number>;
  index_version: number | null;
  index_data_hash: string | null;
  embedding_model: string | null;
  prompt_version: string | null;
  provider: string | null;
  provider_model: string | null;
}

export interface SupportBrief {
  request_id: string;
  mode: Mode;
  retrieval_strength: Strength;
  strength_detail: RetrievalStrengthDetail | null;
  similar_cases: EvidenceTicket[];
  suggested_steps: SuggestedStep[];
  relevance_explanation: string | null;
  risk_signal: null;
  manual_review_required: boolean;
  insufficient_evidence: boolean;
  warnings: string[];
  stage_trace: StageTrace[];
  versions: VersionStamp;
  disclaimer: string;
}

export interface DisabledBrief {
  request_id: string;
  mode: "disabled";
  capability: string;
  reason: string;
  detail: string;
}

export function isDisabled(b: SupportBrief | DisabledBrief): b is DisabledBrief {
  return b.mode === "disabled";
}

const BASE = "/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* non-JSON error body; the status text is what we have */
    }
    throw new Error(`${res.status} — ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  capabilities: () => request<Capabilities>("/capabilities"),

  supportBrief: (body: {
    issue_description: string;
    product_area?: string | null;
    issue_type?: string | null;
    top_k?: number;
  }) =>
    request<SupportBrief | DisabledBrief>("/support-brief", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  trace: (requestId: string) =>
    request<{
      request_id: string;
      mode: Mode;
      retrieval_strength: Strength;
      stage_trace: StageTrace[];
      versions: VersionStamp;
      provider_calls: number;
      note: string;
    }>(`/support-brief/${requestId}/trace`),
};
