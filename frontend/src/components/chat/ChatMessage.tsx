"use client";

import { useState, useRef, useEffect } from "react";
import { Copy, Check, MoreHorizontal, PanelRightOpen } from "lucide-react";
import MarkdownRenderer from "./MarkdownRenderer";
import type { Citation, Claim, Conflict } from "@/lib/api";
import { resolveSourceUrl } from "@/lib/api";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/lib/utils";
import { useChatStore } from "@/stores/chatStore";

interface MessageProps {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  isLatest?: boolean;
  isStreaming?: boolean;
  meta?: { filename?: string; status?: string };
  citations?: Citation[];
  claims?: Claim[];
  conflicts?: Conflict[];
  finalStatus?: string;
  latencyMs?: number;
  estimatedCostUsd?: number;
  timestamp: Date;
}

function VerificationSummary({ claims, conflicts }: { claims?: Claim[]; conflicts?: Conflict[] }) {
  const failed = claims?.filter((c) => ["unverified", "contradicted", "uncertain"].includes(c.status)) ?? [];
  const hasConflicts = (conflicts?.length ?? 0) > 0;

  if (!claims?.length && !hasConflicts) return null;

  return (
    <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-border pt-2">
      <Badge variant="mono">{claims?.filter((c) => c.status === "verified").length ?? 0} verified</Badge>
      {failed.length > 0 && <Badge variant="mono">{failed.length} unverified</Badge>}
      {hasConflicts && <Badge variant="mono">conflict detected</Badge>}
    </div>
  );
}

function SourceChips({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <div className="mt-3 flex flex-wrap gap-1.5">
      <span className="label-caps mr-1 self-center">Sources</span>
      {citations.slice(0, 8).map((c, i) => {
        const chip = (
          <>
            <span className="text-accent-bright/90">[{i + 1}]</span>
            <span className="max-w-[220px] truncate">{c.source_name}</span>
          </>
        );
        const cls =
          "citation-chip max-w-[260px] gap-1.5 aria-[current=true]:border-accent-bright";
        return c.source_url ? (
          <a
            key={c.evidence_id}
            href={resolveSourceUrl(c.source_url)}
            target="_blank"
            rel="noopener noreferrer"
            className={cls}
            title={c.source_name}
          >
            {chip}
          </a>
        ) : (
          <span key={c.evidence_id} className={cls} title={c.source_name}>
            {chip}
          </span>
        );
      })}
    </div>
  );
}

export default function ChatMessage({
  id,
  role,
  content,
  isStreaming,
  meta,
  citations,
  claims,
  conflicts,
  finalStatus,
  latencyMs,
  estimatedCostUsd,
  timestamp,
}: MessageProps) {
  const [copied, setCopied] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const { selectedMessageId, openMessageAnalysis } = useChatStore();
  const isSelected = selectedMessageId === id;
  const isUser = role === "user";
  const isSystem = role === "system";

  useEffect(() => {
    if (!menuOpen) return;
    const onDoc = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [menuOpen]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (isSystem) {
    const failed = meta?.status === "failed";
    return (
      <div className="flex justify-center py-2 animate-msg-in">
        <div
          className={cn(
            "rounded-md border px-3 py-1.5 font-mono text-[10px] uppercase tracking-wider",
            failed
              ? "border-warning/60 bg-warning/10 text-warning"
              : "border-border bg-surface-raised text-text-secondary"
          )}
        >
          {content}
        </div>
      </div>
    );
  }

  const timeStr = timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  // Before the first streamed token arrives we show a calm pulse, not a spinner.
  const awaitingFirstToken = isStreaming && content.length === 0;

  return (
    <article
      onClick={() => {
        if (!isUser) openMessageAnalysis(id);
      }}
      className={cn(
        "animate-msg-in group relative w-full",
        isUser ? "flex justify-end" : ""
      )}
    >
      <div
        className={cn(
          "bubble min-w-0 px-4 py-3 select-text",
          "max-w-[min(100%,var(--bubble-max))]",
          isUser
            ? "bubble-user bg-surface-raised lg:ml-auto"
            : cn(
                "bg-surface border transition-colors cursor-pointer hover:border-accent/70",
                isSelected && "border-strong shadow-[0_0_10px_var(--accent-glow)]",
                isStreaming && !awaitingFirstToken && "border-accent-bright/60"
              )
        )}
      >
        {/* Meta header */}
        <div className="mb-2 flex items-center justify-between gap-3">
          <div className="flex items-baseline gap-2.5">
            <span className="font-mono text-[9px] uppercase tracking-[0.15em] text-text-secondary">
              {isUser ? "You" : "SCRAG"}
            </span>
            <span className="font-mono text-[9px] text-text-muted">{timeStr}</span>
            {meta?.filename && (
              <span className="truncate font-mono text-[9px] text-text-muted" title={meta.filename}>
                {meta.filename}
              </span>
            )}
          </div>
          {!isUser && (
            <div className="flex items-center gap-2">
              {finalStatus && <Badge variant="mono">{finalStatus}</Badge>}
              {typeof latencyMs === "number" && (
                <span className="font-mono text-[9px] text-text-muted">{(latencyMs / 1000).toFixed(1)}s</span>
              )}
              {typeof estimatedCostUsd === "number" && estimatedCostUsd > 0 && (
                <span className="font-mono text-[9px] text-text-muted">~${estimatedCostUsd.toFixed(4)}</span>
              )}
              <div className="relative" ref={menuRef}>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setMenuOpen((o) => !o);
                  }}
                  className="p-1 text-text-muted opacity-0 transition-opacity hover:text-text-primary focus-visible:opacity-100 group-hover:opacity-100"
                  title="Message actions"
                  aria-label="Message actions"
                >
                  <MoreHorizontal size={14} />
                </button>
                {menuOpen && (
                  <div className="absolute right-0 top-full z-20 mt-1 min-w-[160px] rounded-lg border border-border bg-surface py-1 shadow-xl">
                    <button
                      className="flex w-full items-center gap-2 px-3 py-2 text-left font-mono text-[10px] uppercase tracking-wider text-text-secondary hover:bg-surface-raised hover:text-text-primary"
                      onClick={(e) => {
                        e.stopPropagation();
                        openMessageAnalysis(id);
                        setMenuOpen(false);
                      }}
                    >
                      <PanelRightOpen size={12} />
                      Show analysis
                    </button>
                    <button
                      className="flex w-full items-center gap-2 px-3 py-2 text-left font-mono text-[10px] uppercase tracking-wider text-text-secondary hover:bg-surface-raised hover:text-text-primary"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleCopy();
                        setMenuOpen(false);
                      }}
                    >
                      {copied ? <Check size={12} className="text-success" /> : <Copy size={12} />}
                      Copy response
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Body — plain #F1EDEF while streaming, never glowing */}
        {isUser ? (
          <p className="text-[15px] leading-[1.65] text-text-primary">{content}</p>
        ) : awaitingFirstToken ? (
          /* Calm three-dot pulse during retrieval/generation warm-up */
          <div className="dot-pulse flex items-center gap-1.5 py-1" aria-label="Working on your answer">
            <span />
            <span />
            <span />
          </div>
        ) : (
          <>
            <MarkdownRenderer content={content} citations={citations} />
            {isStreaming && <span className="streaming-cursor" aria-hidden="true" />}
            <VerificationSummary claims={claims} conflicts={conflicts} />
            {citations && citations.length > 0 && <SourceChips citations={citations} />}
          </>
        )}
      </div>
    </article>
  );
}
