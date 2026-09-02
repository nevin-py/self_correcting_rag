"use client";

import { useEffect, useRef, useState } from "react";
import {
  BrainCircuit,
  Search,
  PenLine,
  ShieldCheck,
  RefreshCcw,
  Check,
  Loader2,
  ChevronDown,
  ChevronRight,
  ArrowDown,
  AlertTriangle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { PipelineEvent } from "@/stores/chatStore";

/**
 * Interactive view of the real agent graph:
 *
 *   plan ──► gather ──► generate ──► verify ──► done
 *                            ▲           │
 *                            └── repair ─┘   (≤ MAX_REPAIR_PASSES)
 *
 * Each node shows live status + timing and expands to its detail line.
 * A pass counter visualizes self-correction loops; a mini log keeps the
 * raw event stream for everything else (clarification/conversational).
 */

interface NodeSpec {
  id: string;
  label: string;
  sub: string;
  icon: React.ReactNode;
}

const GRAPH_NODES: NodeSpec[] = [
  { id: "classify_and_plan", label: "Plan", sub: "understand + rewrite query", icon: <BrainCircuit size={13} /> },
  { id: "gather_evidence", label: "Gather evidence", sub: "documents + web, reranked", icon: <Search size={13} /> },
  { id: "generate_answer", label: "Generate", sub: "cited draft answer", icon: <PenLine size={13} /> },
  { id: "verify_answer", label: "Verify", sub: "claim-by-claim fact check", icon: <ShieldCheck size={13} /> },
];

type NodeStatus = "pending" | "running" | "done" | "error";

interface NodeView extends NodeSpec {
  status: NodeStatus;
  detail?: string;
  nodeMs?: number;
  totalMs?: number;
  passes: number; // how many times this node ran (repair passes)
}

const SHORTCUTS: Record<string, { label: string; icon: React.ReactNode }> = {
  conversational_response: { label: "Direct reply (no retrieval)", icon: <PenLine size={13} /> },
  ask_clarification: { label: "Asked for clarification", icon: <AlertTriangle size={13} /> },
};

function fmtMs(ms?: number): string {
  if (typeof ms !== "number" || ms <= 0) return "";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

export default function PipelineTracker({
  events,
  isStreaming,
  hasConflicts = false,
  compact = false,
}: PipelineTrackerProps) {
  const [openNodes, setOpenNodes] = useState<Record<string, boolean>>({});
  const logRef = useRef<HTMLDivElement>(null);

  // Build per-node views from the raw event stream.
  const nodes: NodeView[] = GRAPH_NODES.map((spec) => {
    const evts = events.filter((e) => e.node === spec.id);
    const last = evts[evts.length - 1];
    const lastDone = [...evts].reverse().find((e) => e.status === "done");
    const status: NodeStatus = last
      ? last.status === "done"
        ? "done"
        : "running"
      : "pending";
    return {
      ...spec,
      status: isStreaming || evts.length > 0 ? status : "pending",
      detail: last?.detail,
      nodeMs: lastDone?.nodeMs,
      totalMs: lastDone?.elapsedMs,
      passes: evts.filter((e) => e.status === "done").length,
    };
  });

  // Anything outside the main graph (clarification, conversational, etc.)
  const extraEvents = events.filter(
    (e) => !GRAPH_NODES.some((n) => n.id === e.node)
  );

  const repairPasses = nodes.find((n) => n.id === "generate_answer")?.passes ?? 0;
  const started = events.length > 0;
  const finished =
    !isStreaming &&
    started &&
    (nodes.find((n) => n.id === "verify_answer")?.status === "done" ||
      extraEvents.some((e) => e.status === "done"));

  // Auto-scroll the mini log while streaming.
  useEffect(() => {
    if (isStreaming && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [events, isStreaming]);

  const toggle = (id: string) =>
    setOpenNodes((prev) => ({ ...prev, [id]: !prev[id] }));

  if (compact) {
    const running = events.find((e) => e.status === "running");
    const label = running
      ? `${GRAPH_NODES.find((n) => n.id === running.node)?.label ?? running.label}…`
      : finished
        ? "Pipeline complete"
        : started
          ? "Processing"
          : "STANDBY";
    return (
      <div className="flex items-center gap-2 border border-border bg-surface-inset px-2 py-1">
        <span className="label-caps">Pipeline</span>
        <span className="font-mono text-[10px] text-accent-bright">{label}</span>
        {isStreaming && <Loader2 size={10} className="animate-spin text-accent-bright" />}
      </div>
    );
  }

  return (
    <div className="border border-border bg-surface">
      <div className="flex items-center justify-between border-b border-border px-3 py-1.5">
        <span className="label-caps">Agent Pipeline</span>
        {isStreaming ? (
          <span className="flex items-center gap-1.5 font-mono text-[9px] text-accent-bright">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-accent-bright" />
            RUNNING
          </span>
        ) : finished ? (
          <span className="flex items-center gap-1.5 font-mono text-[9px] text-text-muted">
            <Check size={10} className="text-accent" /> COMPLETE
          </span>
        ) : (
          <span className="font-mono text-[9px] text-text-muted">IDLE</span>
        )}
      </div>

      {/* ── Graph ── */}
      <div className="flex flex-col gap-0 p-2">
        {nodes.map((node, i) => {
          const isOpen = openNodes[node.id] ?? node.status === "running";
          const inRepair = node.passes > 1;
          return (
            <div key={node.id}>
              <button
                onClick={() => toggle(node.id)}
                className={cn(
                  "group flex w-full items-center gap-2 border px-2 py-1.5 text-left transition-colors",
                  node.status === "running" && "border-accent-bright bg-accent-glow",
                  node.status === "done" && "border-border-strong bg-surface-raised",
                  node.status === "pending" && "border-border bg-void opacity-60",
                  node.status === "error" && "border-error bg-accent-glow"
                )}
              >
                <span
                  className={cn(
                    "flex h-6 w-6 shrink-0 items-center justify-center rounded-full border",
                    node.status === "running" && "border-accent-bright text-accent-bright",
                    node.status === "done" && "border-accent text-accent",
                    node.status === "pending" && "border-border text-text-muted"
                  )}
                >
                  {node.status === "running" ? (
                    <Loader2 size={13} className="animate-spin" />
                  ) : node.status === "done" ? (
                    <Check size={13} />
                  ) : (
                    node.icon
                  )}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-2">
                    <span
                      className={cn(
                        "font-mono text-[10px] uppercase tracking-wider",
                        node.status === "running" ? "text-accent-bright" : "text-text-primary"
                      )}
                    >
                      {node.label}
                    </span>
                    {inRepair && node.status === "done" && (
                      <span className="flex items-center gap-0.5 rounded-sm border border-accent/60 px-1 font-mono text-[8px] text-accent-bright">
                        <RefreshCcw size={7} /> ×{node.passes}
                      </span>
                    )}
                  </span>
                  <span className="block truncate font-mono text-[8px] text-text-muted">
                    {node.status === "running" ? node.detail || node.sub : node.sub}
                  </span>
                </span>
                <span className="shrink-0 font-mono text-[9px] text-text-muted">
                  {fmtMs(node.nodeMs)}
                </span>
                {isOpen ? (
                  <ChevronDown size={11} className="shrink-0 text-text-muted" />
                ) : (
                  <ChevronRight size={11} className="shrink-0 text-text-muted" />
                )}
              </button>

              {isOpen && node.detail && (
                <div className="border border-t-0 border-border bg-surface-inset px-8 py-1.5">
                  <p className="font-mono text-[9px] leading-relaxed text-text-secondary">
                    {node.detail}
                    {typeof node.totalMs === "number" && (
                      <span className="text-text-muted"> · total {fmtMs(node.totalMs)}</span>
                    )}
                  </p>
                </div>
              )}

              {/* Repair loop arrow between generate and verify */}
              {i === 2 && repairPasses > 1 && (
                <div className="my-0.5 ml-6 flex items-center gap-1.5 border-l border-dashed border-accent/50 pl-2">
                  <RefreshCcw size={9} className="text-accent" />
                  <span className="font-mono text-[8px] uppercase tracking-wider text-accent">
                    self-correction loop ×{repairPasses - 1}
                  </span>
                </div>
              )}
              {i < nodes.length - 1 && (
                <div className="ml-6 h-2 w-px bg-border" aria-hidden>
                  <ArrowDown size={0} />
                </div>
              )}
            </div>
          );
        })}

        {/* Out-of-graph outcomes (clarification / conversational) */}
        {extraEvents.map((evt) => {
          const meta = SHORTCUTS[evt.node];
          if (!meta) return null;
          return (
            <div
              key={evt.id}
              className={cn(
                "mt-1 flex items-center gap-2 border px-2 py-1.5",
                evt.status === "running"
                  ? "border-accent-bright bg-accent-glow"
                  : "border-border-strong bg-surface-raised"
              )}
            >
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-accent text-accent">
                {evt.status === "running" ? (
                  <Loader2 size={13} className="animate-spin" />
                ) : (
                  meta.icon
                )}
              </span>
              <span className="font-mono text-[10px] uppercase tracking-wider text-text-primary">
                {meta.label}
              </span>
            </div>
          );
        })}
      </div>

      {hasConflicts && !isStreaming && (
        <div className="border-t border-accent bg-accent-glow px-3 py-2">
          <p className="font-mono text-[10px] uppercase tracking-wider text-accent-bright">
            Conflict detected → resolved with caveats
          </p>
        </div>
      )}

      {/* ── Mini log ── */}
      {events.length > 0 && (
        <div
          ref={logRef}
          className="max-h-28 overflow-y-auto border-t border-border bg-surface-inset px-2 py-1.5"
        >
          {events.slice(-30).map((evt) => (
            <div key={evt.id} className="flex items-center gap-2 py-px">
              <span className="font-mono text-[8px] text-text-muted">
                {evt.timestamp.toLocaleTimeString([], { hour12: false })}
              </span>
              <span
                className={cn(
                  "h-1 w-1 rounded-full",
                  evt.status === "running" ? "bg-accent-bright" : "bg-text-muted"
                )}
              />
              <span className="truncate font-mono text-[9px] text-text-secondary">
                {evt.label}
                {evt.detail ? ` — ${evt.detail}` : ""}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

interface PipelineTrackerProps {
  events: PipelineEvent[];
  isStreaming: boolean;
  hasConflicts?: boolean;
  compact?: boolean;
}
