"use client";

import { useState, useRef, useEffect } from "react";
import { Copy, Check, MoreHorizontal, PanelRightOpen } from "lucide-react";
import MarkdownRenderer from "./MarkdownRenderer";
import type { Citation, Claim, Conflict } from "@/lib/api";
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
      {hasConflicts && <Badge variant="accent">Conflict Resolved</Badge>}
      {failed.length === 0 ? (
        <Badge variant="success">Verified</Badge>
      ) : (
        <Badge variant="warning">{failed.length} claim{failed.length > 1 ? "s" : ""} flagged</Badge>
      )}
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
    return (
      <div className="flex justify-center py-2">
        <div
          className={cn(
            "border px-3 py-1.5 font-mono text-[10px] uppercase tracking-wider",
            meta?.status === "failed"
              ? "border-error bg-accent-glow text-error"
              : "border-border bg-surface-raised text-text-secondary"
          )}
        >
          {content}
        </div>
      </div>
    );
  }

  const timeStr = timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  return (
    <article
      onClick={() => {
        if (!isUser) openMessageAnalysis(id);
      }}
      className={cn(
        "group relative border transition-colors",
        isUser
          ? "ml-8 border-border bg-surface-raised"
          : cn(
              "mr-0 border-border bg-surface cursor-pointer hover:border-border-strong",
              isSelected && "border-accent bg-accent-glow/30",
              isStreaming && "border-accent-bright"
            )
      )}
    >
      <div className="flex items-center justify-between border-b border-border px-3 py-1.5">
        <div className="flex items-center gap-3">
          <span className="font-mono text-[9px] uppercase tracking-[0.15em] text-text-muted">
            {isUser ? "Operator" : "System Response"}
          </span>
          <span className="font-mono text-[9px] text-text-muted">{timeStr}</span>
        </div>
        {!isUser && (
          <div className="flex items-center gap-2">
            {finalStatus && <Badge variant="mono">{finalStatus}</Badge>}
            {typeof latencyMs === "number" && (
              <span className="font-mono text-[9px] text-text-muted">{latencyMs.toFixed(0)}ms</span>
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
                className="opacity-0 transition-opacity group-hover:opacity-100 p-1 text-text-muted hover:text-text-primary"
                title="Message actions"
                aria-label="Message actions"
              >
                <MoreHorizontal size={14} />
              </button>
              {menuOpen && (
                <div className="absolute right-0 top-full z-20 mt-1 min-w-[160px] border border-border bg-surface shadow-lg">
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

      <div className={cn("px-4 py-3 select-text", isUser ? "text-sm text-text-primary" : "")}>
        {isUser ? (
          <p className="leading-relaxed">{content}</p>
        ) : (
          <>
            <MarkdownRenderer content={content} citations={citations} />
            <VerificationSummary claims={claims} conflicts={conflicts} />
            {citations && citations.length > 0 && (
              <div className="mt-3 space-y-1 border-t border-border pt-2">
                <p className="label-caps text-text-muted">Sources</p>
                <div className="flex flex-col gap-1">
                  {citations.slice(0, 8).map((c, i) => (
                    <div key={c.evidence_id} className="flex items-start gap-2 font-mono text-[10px]">
                      <span className="shrink-0 text-accent-bright">[{i + 1}]</span>
                      {c.source_url ? (
                        <a
                          href={c.source_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="truncate text-text-secondary hover:text-accent-bright"
                        >
                          {c.source_name}
                        </a>
                      ) : (
                        <span className="truncate text-text-secondary">{c.source_name}</span>
                      )}
                      <span className="shrink-0 text-text-muted">{c.source_type}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {isStreaming && (
        <div className="flex items-center gap-1 border-t border-accent px-3 py-1">
          <span className="inline-block h-1 w-1 bg-accent-bright pipeline-active" />
          <span className="font-mono text-[9px] uppercase tracking-wider text-accent-bright">Transmitting</span>
        </div>
      )}
    </article>
  );
}
