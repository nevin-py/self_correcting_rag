"use client";

import { cn } from "@/lib/utils";
import { PIPELINE_STAGES, buildPhaseStates, type PhaseState } from "@/lib/pipeline";
import { PipelineEvent } from "@/stores/chatStore";

interface PipelineTrackerProps {
  events: PipelineEvent[];
  isStreaming: boolean;
  hasConflicts?: boolean;
  compact?: boolean;
}

function StageBlock({ stage, state, isLast }: { stage: typeof PIPELINE_STAGES[0]; state: PhaseState; isLast: boolean }) {
  const statusStyles: Record<string, string> = {
    idle: "border-border text-text-muted bg-void",
    active: "border-accent-bright text-accent-bright bg-accent-glow pipeline-active",
    done: "border-border-strong text-text-secondary bg-surface-raised",
    conflict: "border-accent text-accent-bright bg-accent-glow",
    error: "border-error text-error bg-accent-glow",
  };

  return (
    <div className="flex items-stretch">
      <div
        className={cn(
          "flex min-w-0 flex-1 flex-col border px-2 py-1.5 transition-colors duration-[var(--duration-normal)]",
          statusStyles[state.status]
        )}
      >
        <span className="font-mono text-[9px] tracking-[0.15em]">{stage.label}</span>
        {state.status === "active" && state.label && (
          <span className="mt-0.5 truncate font-mono text-[8px] text-text-muted">{state.label}</span>
        )}
        {state.status === "conflict" && (
          <span className="mt-0.5 font-mono text-[8px] text-accent-bright">DETECTED</span>
        )}
      </div>
      {!isLast && (
        <div className="flex w-3 items-center justify-center bg-void">
          <span className={cn("font-mono text-[10px]", state.status === "done" || state.status === "active" ? "text-accent" : "text-border")}>
            →
          </span>
        </div>
      )}
    </div>
  );
}

export default function PipelineTracker({ events, isStreaming, hasConflicts = false, compact = false }: PipelineTrackerProps) {
  const phases = buildPhaseStates(events, hasConflicts, !isStreaming && events.length > 0);

  if (compact) {
    const activeIdx = phases.findIndex((p) => p.status === "active" || p.status === "conflict");
    const current = activeIdx >= 0 ? phases[activeIdx] : phases.find((p) => p.status === "done");
    if (!current && events.length === 0) return null;
    return (
      <div className="flex items-center gap-2 border border-border bg-surface-inset px-2 py-1">
        <span className="label-caps">Pipeline</span>
        <span className="font-mono text-[10px] text-accent-bright">
          {current ? PIPELINE_STAGES.find((s) => s.id === current.phase)?.label : "STANDBY"}
        </span>
      </div>
    );
  }

  return (
    <div className="border border-border bg-surface">
      <div className="flex items-center justify-between border-b border-border px-3 py-1.5">
        <span className="label-caps">Correction Pipeline</span>
        {isStreaming && (
          <span className="flex items-center gap-1.5 font-mono text-[9px] text-accent-bright">
            <span className="inline-block h-1.5 w-1.5 bg-accent-bright pipeline-active" />
            PROCESSING
          </span>
        )}
      </div>
      <div className="flex p-2">
        {PIPELINE_STAGES.map((stage, i) => (
          <StageBlock
            key={stage.id}
            stage={stage}
            state={phases[i]}
            isLast={i === PIPELINE_STAGES.length - 1}
          />
        ))}
      </div>
      {hasConflicts && !isStreaming && (
        <div className="border-t border-accent bg-accent-glow px-3 py-2">
          <p className="font-mono text-[10px] uppercase tracking-wider text-accent-bright">
            LOCAL KNOWLEDGE → CONFLICT DETECTED → WEB SEARCH → VERIFIED → ANSWER UPDATED
          </p>
        </div>
      )}
    </div>
  );
}
