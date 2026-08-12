"use client";

import { useMemo } from "react";
import { ExternalLink, ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import { useChatStore, Message } from "@/stores/chatStore";
import { Citation, Claim, Conflict } from "@/lib/api";
import PipelineTracker from "./PipelineTracker";
import { OPERATIONAL_LABELS } from "@/lib/pipeline";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/lib/utils";

function Section({
  title,
  count,
  defaultOpen = true,
  children,
}: {
  title: string;
  count?: number;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b border-border">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-3 py-2 text-left hover:bg-surface-raised"
      >
        <span className="label-caps">
          {title}
          {count !== undefined && <span className="ml-2 text-text-muted">[{count}]</span>}
        </span>
        {open ? <ChevronDown size={12} className="text-text-muted" /> : <ChevronRight size={12} className="text-text-muted" />}
      </button>
      {open && <div className="px-3 pb-3">{children}</div>}
    </div>
  );
}

function claimStatusVariant(status: Claim["status"]): "success" | "warning" | "error" | "default" {
  if (status === "verified") return "success";
  if (status === "partial" || status === "uncertain") return "warning";
  if (status === "contradicted") return "error";
  return "default";
}

function EvidenceItem({ citation, index }: { citation: Citation; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const isWeb = citation.source_type === "web";

  return (
    <div className="border border-border bg-surface-inset">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-start gap-2 px-2 py-2 text-left hover:bg-surface-raised"
      >
        <span className="font-mono text-[10px] text-accent-bright">[{index + 1}]</span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <Badge variant={isWeb ? "accent" : "mono"}>{citation.source_type}</Badge>
            <span className="truncate text-xs text-text-primary">{citation.source_name}</span>
          </div>
        </div>
        {citation.source_url && (
          <a
            href={citation.source_url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="shrink-0 text-text-muted hover:text-accent-bright"
          >
            <ExternalLink size={11} />
          </a>
        )}
      </button>
      {expanded && (
        <div className="border-t border-border px-2 py-2">
          <p className="text-xs leading-relaxed text-text-secondary select-text">{citation.text}</p>
          <div className="mt-2 flex flex-wrap gap-3 font-mono text-[9px] text-text-muted">
            {citation.source_date && <span>DATE {new Date(citation.source_date).toLocaleDateString()}</span>}
            <span>AUTH {(citation.authority_score * 100).toFixed(0)}%</span>
            <span>REC {(citation.recency_score * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}
    </div>
  );
}

function ClaimItem({ claim, citations }: { claim: Claim; citations?: Citation[] }) {
  const refs = claim.evidence_ids
    .map((id) => citations?.find((c) => c.evidence_id === id))
    .filter(Boolean) as Citation[];

  return (
    <div className="border border-border bg-surface-inset px-2 py-2">
      <div className="flex items-start gap-2">
        <Badge variant={claimStatusVariant(claim.status)}>{claim.status}</Badge>
        <p className="flex-1 text-xs leading-relaxed text-text-primary select-text">{claim.text}</p>
      </div>
      {refs.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {refs.map((ref, i) => (
            <span key={ref.evidence_id} className="font-mono text-[9px] text-accent-bright">
              [{citations?.indexOf(ref)! + 1 || i + 1}]
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function OperationLog({ events, isStreaming }: { events: ReturnType<typeof useChatStore.getState>["pipelineEvents"]; isStreaming: boolean }) {
  if (events.length === 0) {
    return <p className="font-mono text-[10px] text-text-muted">Awaiting query...</p>;
  }

  return (
    <div className="space-y-px">
      {events.map((evt) => (
        <div
          key={evt.id}
          className={cn(
            "flex items-start gap-2 border-l-2 px-2 py-1.5",
            evt.status === "running" ? "border-accent-bright bg-accent-glow" : "border-border bg-surface-inset"
          )}
        >
          <span className="font-mono text-[9px] text-text-muted">
            {evt.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
          </span>
          <div className="min-w-0 flex-1">
            <p className="font-mono text-[10px] uppercase tracking-wider text-text-primary">
              {OPERATIONAL_LABELS[evt.node] || evt.label}
            </p>
            {evt.detail && <p className="mt-0.5 truncate text-[10px] text-text-muted">{evt.detail}</p>}
          </div>
          {evt.status === "running" && isStreaming && (
            <span className="font-mono text-[9px] text-accent-bright pipeline-active">RUN</span>
          )}
        </div>
      ))}
    </div>
  );
}

function getSelectedMessage(messages: Message[], selectedId: string | null): Message | null {
  if (!selectedId) {
    const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
    return lastAssistant ?? null;
  }
  return messages.find((m) => m.id === selectedId) ?? null;
}

export default function ContextPanel() {
  const { messages, selectedMessageId, pipelineEvents, isStreaming, toggleRightPanel } = useChatStore();

  const selected = useMemo(
    () => getSelectedMessage(messages, selectedMessageId),
    [messages, selectedMessageId]
  );

  const hasConflicts = (selected?.conflicts?.length ?? 0) > 0;
  const hasWebSources = selected?.citations?.some((c) => c.source_type === "web") ?? false;

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-[var(--header-height)] items-center justify-between border-b border-border px-3">
        <div>
          <p className="label-caps">Analysis Panel</p>
          <p className="font-mono text-[9px] text-text-muted">Evidence · Retrieval · Operations</p>
        </div>
        <button
          onClick={toggleRightPanel}
          className="font-mono text-[10px] text-text-muted hover:text-text-primary lg:hidden"
        >
          [×]
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="p-2">
          <PipelineTracker
            events={pipelineEvents}
            isStreaming={isStreaming}
            hasConflicts={hasConflicts || (isStreaming && pipelineEvents.some((e) => e.node === "search_web"))}
          />
        </div>

        {hasWebSources && (
          <div className="mx-2 mb-2 border border-accent bg-accent-glow px-2 py-1.5">
            <p className="font-mono text-[9px] uppercase tracking-wider text-accent-bright">
              Web Search Fallback Active
            </p>
          </div>
        )}

        <Section title="Operations" count={pipelineEvents.length} defaultOpen={isStreaming}>
          <OperationLog events={pipelineEvents} isStreaming={isStreaming} />
        </Section>

        <Section title="Evidence" count={selected?.citations?.length}>
          {!selected?.citations?.length ? (
            <p className="font-mono text-[10px] text-text-muted">No evidence indexed</p>
          ) : (
            <div className="space-y-1">
              {selected.citations.map((c, i) => (
                <EvidenceItem key={c.evidence_id} citation={c} index={i} />
              ))}
            </div>
          )}
        </Section>

        <Section title="Verification" count={selected?.claims?.length}>
          {!selected?.claims?.length ? (
            <p className="font-mono text-[10px] text-text-muted">No claims extracted</p>
          ) : (
            <div className="space-y-1">
              {selected.claims.map((claim) => (
                <ClaimItem key={claim.claim_id} claim={claim} citations={selected.citations} />
              ))}
            </div>
          )}
        </Section>

        {selected?.conflicts && selected.conflicts.length > 0 && (
          <Section title="Conflicts" count={selected.conflicts.length}>
            {selected.conflicts.map((conflict, i) => (
              <ConflictItem key={i} conflict={conflict} />
            ))}
          </Section>
        )}

        {selected && (
          <div className="border-t border-border px-3 py-2">
            <div className="flex flex-wrap gap-2">
              {selected.finalStatus && <Badge variant="mono">{selected.finalStatus}</Badge>}
              {typeof selected.latencyMs === "number" && (
                <Badge variant="mono">{selected.latencyMs.toFixed(0)}ms</Badge>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ConflictItem({ conflict }: { conflict: Conflict }) {
  return (
    <div className="mb-1 border border-accent bg-accent-glow px-2 py-2">
      <p className="font-mono text-[9px] uppercase tracking-wider text-accent-bright">Conflict Resolved</p>
      <p className="mt-1 text-xs text-text-secondary">{conflict.reason}</p>
      <p className="mt-1 font-mono text-[9px] text-text-muted">Resolution: {conflict.resolution}</p>
    </div>
  );
}
