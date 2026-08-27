export type PipelinePhase =
  | "retrieve"
  | "verify"
  | "conflict"
  | "search"
  | "correct"
  | "answer";

export interface PipelineStage {
  id: PipelinePhase;
  label: string;
  shortLabel: string;
  nodes: string[];
}

export const PIPELINE_STAGES: PipelineStage[] = [
  {
    id: "retrieve",
    label: "RETRIEVE",
    shortLabel: "R",
    nodes: ["classify_and_plan", "retrieve_documents", "assemble_evidence"],
  },
  {
    id: "verify",
    label: "VERIFY",
    shortLabel: "V",
    nodes: ["extract_verify_claims", "verify_answer_claims"],
  },
  {
    id: "conflict",
    label: "CONFLICT",
    shortLabel: "C",
    nodes: [],
  },
  {
    id: "search",
    label: "SEARCH",
    shortLabel: "S",
    nodes: ["search_web"],
  },
  {
    id: "correct",
    label: "CORRECT",
    shortLabel: "X",
    nodes: ["repair_claims"],
  },
  {
    id: "answer",
    label: "ANSWER",
    shortLabel: "A",
    nodes: ["generate_answer"],
  },
];

const NODE_TO_PHASE: Record<string, PipelinePhase> = Object.fromEntries(
  PIPELINE_STAGES.flatMap((stage) => stage.nodes.map((node) => [node, stage.id]))
) as Record<string, PipelinePhase>;

export function nodeToPhase(node: string): PipelinePhase {
  return NODE_TO_PHASE[node] ?? "retrieve";
}

export function phaseIndex(phase: PipelinePhase): number {
  return PIPELINE_STAGES.findIndex((s) => s.id === phase);
}

export type StageStatus = "idle" | "active" | "done" | "conflict" | "error";

export interface PhaseState {
  phase: PipelinePhase;
  status: StageStatus;
  label?: string;
  detail?: string;
  elapsedMs?: number;
}

export function buildPhaseStates(
  events: Array<{ node: string; label: string; detail?: string; status: string; elapsedMs?: number; nodeMs?: number }>,
  hasConflicts = false,
  isComplete = false
): PhaseState[] {
  const activeNodes = new Set(events.map((e) => e.node));
  const lastEvent = events[events.length - 1];
  const lastPhase = lastEvent ? nodeToPhase(lastEvent.node) : null;

  return PIPELINE_STAGES.map((stage) => {
    const touched = stage.nodes.some((n) => activeNodes.has(n));
    const isActive =
      !isComplete && lastPhase === stage.id && lastEvent?.status === "running";

    if (stage.id === "conflict") {
      if (hasConflicts) return { phase: stage.id, status: "conflict" as StageStatus, label: "CONFLICT DETECTED" };
      if (isComplete && !hasConflicts) return { phase: stage.id, status: "done" as StageStatus };
      return { phase: stage.id, status: "idle" as StageStatus };
    }

    if (isActive) {
      return {
        phase: stage.id,
        status: "active" as StageStatus,
        label: lastEvent?.label,
        detail: lastEvent?.detail,
        elapsedMs: lastEvent?.elapsedMs,
      };
    }

    if (touched || (isComplete && stage.id === "answer")) {
      const doneEv = [...events].reverse().find(
        (e) => stage.nodes.includes(e.node) && e.status === "done" && typeof e.nodeMs === "number"
      );
      return {
        phase: stage.id,
        status: "done" as StageStatus,
        elapsedMs: doneEv?.nodeMs,
      };
    }

    return { phase: stage.id, status: "idle" as StageStatus };
  });
}

export const OPERATIONAL_LABELS: Record<string, string> = {
  classify_and_plan: "CLASSIFY + PLAN",
  classify_query: "QUERY CLASSIFICATION",
  build_plan: "STRATEGY FORMULATION",
  retrieve_documents: "LOCAL KNOWLEDGE RETRIEVAL",
  search_web: "WEB SEARCH FALLBACK",
  assemble_evidence: "EVIDENCE ASSEMBLY",
  extract_verify_claims: "CLAIM EXTRACTION",
  generate_answer: "RESPONSE GENERATION",
  verify_answer_claims: "FACT VERIFICATION",
  repair_claims: "ANSWER CORRECTION",
};
