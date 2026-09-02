"use client";

import { useRef, useState, useCallback, useMemo, useEffect } from "react";
import { Paperclip, ArrowUp, Loader2, X } from "lucide-react";
import { documentApi } from "@/lib/api";
import { useChatStore, estimateLocalTokens } from "@/stores/chatStore";
import ProviderSelect from "./ProviderSelect";
import { cn } from "@/lib/utils";
import { Kbd } from "@/components/ui/Kbd";

interface ComposerProps {
  sessionId: string;
  disabled?: boolean;
}

function ContextRing({ used, limit }: { used: number; limit: number }) {
  const pct = Math.min(1, limit > 0 ? used / limit : 0);
  const r = 14;
  const c = 2 * Math.PI * r;
  const dash = c * pct;
  const stroke =
    pct >= 0.9 ? "var(--color-error, #c45c2c)" : pct >= 0.7 ? "var(--color-accent, #c45c2c)" : "#9a9590";

  return (
    <div
      className="relative flex h-9 w-9 items-center justify-center"
      title={`Context ~${used.toLocaleString()} / ${limit.toLocaleString()} tokens (${(pct * 100).toFixed(0)}%)`}
    >
      <svg width="36" height="36" className="-rotate-90">
        <circle cx="18" cy="18" r={r} fill="none" stroke="currentColor" strokeWidth="2" className="text-border" />
        <circle
          cx="18"
          cy="18"
          r={r}
          fill="none"
          stroke={stroke}
          strokeWidth="2.5"
          strokeDasharray={`${dash} ${c}`}
          strokeLinecap="butt"
        />
      </svg>
      <span className="absolute font-mono text-[8px] text-text-muted">{(pct * 100).toFixed(0)}%</span>
    </div>
  );
}

function formatUsd(n: number): string {
  if (!n || n < 0.0001) return "$0.00";
  if (n < 0.01) return `$${n.toFixed(4)}`;
  return `$${n.toFixed(2)}`;
}

export default function Composer({ sessionId, disabled }: ComposerProps) {
  const {
    isStreaming,
    sendMessage,
    messages,
    chatCostUsd,
    allSessionsCostUsd,
    contextWindowTokens,
    refreshUsage,
  } = useChatStore();
  const [input, setInput] = useState("");
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    void refreshUsage();
  }, [sessionId, refreshUsage]);

  const contextUsed = useMemo(() => {
    const hist = messages.reduce((sum, m) => sum + estimateLocalTokens(m.content), 0);
    return hist + estimateLocalTokens(input);
  }, [messages, input]);

  const handleSend = async () => {
    if (!input.trim() || isStreaming || disabled) return;
    const msg = input.trim();
    setInput("");
    await sendMessage(msg);
    inputRef.current?.focus();
  };

  const handleFileUpload = useCallback(
    async (file: File) => {
      if (!sessionId || uploading) return;
      setUploading(true);
      setUploadedFile(null);
      try {
        await documentApi.upload(sessionId, file);
        useChatStore.getState().addSystemMessage(`Document ingested: ${file.name}`, {
          filename: file.name,
          status: "uploaded",
        });
        setUploadedFile(file.name);
      } catch (err) {
        // Surface the server's rejection reason (413 size, 429 budget, …)
        // instead of a generic message the user can't act on.
        const detail =
          err && typeof err === "object" && "response" in err
            ? (err.response as { data?: { detail?: string } })?.data?.detail
            : undefined;
        useChatStore.getState().addSystemMessage(
          `Ingestion failed: ${file.name}${detail ? ` — ${detail}` : ""}`,
          {
            filename: file.name,
            status: "failed",
          }
        );
      } finally {
        setUploading(false);
      }
    },
    [sessionId, uploading]
  );

  return (
    <div className="shrink-0 border-t border-border bg-surface">
      {uploadedFile && (
        <div className="flex items-center gap-2 border-b border-border px-4 py-2">
          <span className="label-caps">Ingested</span>
          <span className="font-mono text-xs text-text-primary">{uploadedFile}</span>
          <button onClick={() => setUploadedFile(null)} className="ml-auto text-text-muted hover:text-text-primary">
            <X size={12} />
          </button>
        </div>
      )}

      <div className="px-4 py-3">
        <div className="border border-border-strong bg-surface-inset">
          <div className="flex items-end gap-1 p-2">
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileUpload(file);
                e.target.value = "";
              }}
              accept=".txt,.pdf,.md,.html,.csv,.xlsx,.json,.py,.js"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading || isStreaming}
              className="border border-transparent p-2 text-text-muted hover:border-border hover:text-text-primary disabled:opacity-40"
              title="Upload to knowledge base"
            >
              {uploading ? <Loader2 size={14} className="animate-spin" /> : <Paperclip size={14} strokeWidth={1.5} />}
            </button>

            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Enter query — retrieval and verification will commence..."
              rows={1}
              disabled={isStreaming || disabled}
              className="max-h-40 min-h-[2.75rem] flex-1 resize-none bg-transparent py-2.5 text-base leading-relaxed text-text-primary placeholder:text-text-muted focus:outline-none disabled:opacity-40"
            />

            <div className="flex flex-col items-end gap-1 pb-0.5">
              <ContextRing used={contextUsed} limit={contextWindowTokens} />
            </div>

            <ProviderSelect />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming || disabled}
              className={cn(
                "send-active rounded-md border p-2.5",
                input.trim() && !isStreaming
                  ? "send-ready border-transparent text-white"
                  : "border-border bg-surface-inset text-text-muted"
              )}
              title="Submit query"
            >
              <ArrowUp size={14} strokeWidth={2} />
            </button>
          </div>
        </div>

        <div className="mt-2 flex items-center justify-between gap-3">
          <p className="font-mono text-[9px] text-text-muted">
            <Kbd>↵</Kbd> submit · <Kbd>⇧↵</Kbd> newline · <Kbd>⌘K</Kbd> command
          </p>
          <div className="flex flex-wrap items-center justify-end gap-3 font-mono text-[9px] text-text-muted">
            <span title="Estimated spend this chat (rough)">
              Chat {formatUsd(chatCostUsd)}
            </span>
            <span className="text-border">|</span>
            <span title="Estimated spend across all your sessions (rough)">
              All {formatUsd(allSessionsCostUsd)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
