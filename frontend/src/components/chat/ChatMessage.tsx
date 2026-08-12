"use client";

import { useState } from "react";
import { Copy, Check } from "lucide-react";
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
  timestamp,
}: MessageProps) {
  const [copied, setCopied] = useState(false);
  const { selectedMessageId, setSelectedMessage } = useChatStore();
  const isSelected = selectedMessageId === id;
  const isUser = role === "user";
  const isSystem = role === "system";

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
      onClick={() => !isUser && setSelectedMessage(id)}
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
      {/* Header rail */}
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
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleCopy();
              }}
              className="opacity-0 transition-opacity group-hover:opacity-100"
              title="Copy response"
            >
              {copied ? (
                <Check size={11} className="text-success" />
              ) : (
                <Copy size={11} className="text-text-muted hover:text-text-primary" />
              )}
            </button>
          </div>
        )}
      </div>

      {/* Content */}
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
