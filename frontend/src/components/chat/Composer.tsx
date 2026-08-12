"use client";

import { useRef, useState, useCallback } from "react";
import { Paperclip, ArrowUp, Loader2, X } from "lucide-react";
import { documentApi } from "@/lib/api";
import { useChatStore } from "@/stores/chatStore";
import ProviderSelect from "./ProviderSelect";
import { cn } from "@/lib/utils";
import { Kbd } from "@/components/ui/Kbd";

interface ComposerProps {
  sessionId: string;
  disabled?: boolean;
}

export default function Composer({ sessionId, disabled }: ComposerProps) {
  const { isStreaming, sendMessage } = useChatStore();
  const [input, setInput] = useState("");
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

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
      } catch {
        useChatStore.getState().addSystemMessage(`Ingestion failed: ${file.name}`, {
          filename: file.name,
          status: "failed",
        });
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
              className="max-h-32 min-h-[2.5rem] flex-1 resize-none bg-transparent py-2 font-mono text-sm text-text-primary placeholder:text-text-muted focus:outline-none disabled:opacity-40"
            />

            <ProviderSelect />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming || disabled}
              className={cn(
                "border p-2 transition-colors",
                input.trim() && !isStreaming
                  ? "border-accent bg-accent text-text-primary hover:border-accent-bright hover:bg-accent-bright"
                  : "border-border text-text-muted"
              )}
              title="Submit query"
            >
              <ArrowUp size={14} strokeWidth={2} />
            </button>
          </div>
        </div>

        <div className="mt-2 flex items-center justify-between">
          <p className="font-mono text-[9px] text-text-muted">
            <Kbd>↵</Kbd> submit · <Kbd>⇧↵</Kbd> newline · <Kbd>⌘K</Kbd> command
          </p>
          <p className="font-mono text-[9px] text-text-muted">RETRIEVE → VERIFY → CORRECT → ANSWER</p>
        </div>
      </div>
    </div>
  );
}
