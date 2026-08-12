"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { useAuthStore } from "@/stores/authStore";
import { useChatStore, GraphStatus } from "@/stores/chatStore";
import { documentApi } from "@/lib/api";
import Sidebar from "@/components/layout/Sidebar";
import ChatMessage from "@/components/chat/ChatMessage";
import ProviderSelect from "@/components/chat/ProviderSelect";
import { Send, Loader2, Brain, Paperclip, FileText, X, StopCircle } from "lucide-react";

function GraphProgress({ status }: { status: GraphStatus | null }) {
  if (!status) return null;
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex items-center gap-3 px-3 py-2 rounded-[var(--radius-md)] bg-[var(--mountainside)]/50 border border-[var(--apres-ski)]/10 text-xs text-[var(--apres-ski)] max-w-md"
    >
      <Loader2 size={14} className="text-[var(--glacier)] animate-spin shrink-0" />
      <div className="min-w-0">
        <span className="font-medium text-[var(--slopes)] block truncate">{status.label}...</span>
        {status.detail && (
          <span className="text-[var(--apres-ski)]/70 block truncate">{status.detail}</span>
        )}
      </div>
    </motion.div>
  );
}

function StreamingIndicator() {
  return (
    <div className="flex gap-1.5 items-center px-2 py-1">
      {[0, 1, 2].map((i) => (
        <motion.div key={i} className="w-1.5 h-1.5 rounded-full bg-[var(--glacier)]"
          animate={{ opacity: [0.3, 1, 0.3] }} transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }} />
      ))}
    </div>
  );
}

function EmptyState({ onNewChat }: { onNewChat: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4">
      <div className="w-16 h-16 rounded-full bg-[var(--glacier)]/10 flex items-center justify-center mb-6">
        <Brain size={28} className="text-[var(--glacier)]" />
      </div>
      <h2 className="font-display text-2xl text-[var(--arctic)] mb-2">How can I help you today?</h2>
      <p className="text-[var(--apres-ski)] text-sm mb-8 max-w-md">
        Ask questions about your documents, or start a new conversation.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-lg w-full">
        {[
          "What are the key takeaways from my documents?",
          "Summarize the latest meeting notes",
          "Compare Q3 results with Q2",
          "What action items were assigned?",
        ].map((prompt) => (
          <button key={prompt} onClick={() => onNewChat()}
            className="p-3 text-left text-xs text-[var(--slopes)] bg-[var(--mountainside)] border border-[var(--apres-ski)]/10 rounded-[var(--radius-md)] hover:border-[var(--glacier)]/30 transition-colors">
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function ChatSessionPage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.sessionId as string;
  const { token, loadUser } = useAuthStore();
  const { messages, isStreaming, graphStatus, sidebarOpen, selectChat, sendMessage } = useChatStore();
  const [input, setInput] = useState("");
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const messagesEnd = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { loadUser(); }, [loadUser]);
  useEffect(() => { if (!token) router.replace("/login"); }, [token, router]);
  useEffect(() => { if (sessionId) selectChat(sessionId); }, [sessionId, selectChat]);
  useEffect(() => { messagesEnd.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);
  useEffect(() => { if (!isStreaming) inputRef.current?.focus(); }, [isStreaming]);

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;
    const msg = input.trim();
    setInput("");
    await sendMessage(msg);
  };

  const handleFileUpload = useCallback(async (file: File) => {
    if (!sessionId || uploading) return;
    setUploading(true);
    setUploadedFile(null);
    try {
      await documentApi.upload(sessionId, file);
      useChatStore.getState().addSystemMessage(
        `Uploaded "${file.name}" for analysis`,
        { filename: file.name, status: "uploaded" }
      );
      setUploadedFile(file.name);
    } catch (err) {
      console.error("Upload failed:", err);
      useChatStore.getState().addSystemMessage(
        `Failed to upload "${file.name}"`,
        { filename: file.name, status: "failed" }
      );
    } finally {
      setUploading(false);
    }
  }, [sessionId, uploading]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileUpload(file);
    e.target.value = "";
  };

  const handleNewChat = async () => {
    const { createChat } = useChatStore.getState();
    const chat = await createChat("New Chat");
    router.push(`/chat/${chat.chat_id}`);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[var(--midnight)]">
      <Sidebar />
      <div className={`flex-1 flex flex-col transition-all duration-[var(--duration-standard)] ${sidebarOpen ? "ml-[220px]" : "ml-[72px]"}`}>
        {/* Messages area */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <EmptyState onNewChat={handleNewChat} />
          ) : (
            <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
              <AnimatePresence>
                {messages.map((msg, i) => (
                  <ChatMessage
                    key={msg.id}
                    role={msg.role}
                    content={msg.content}
                    isLatest={i === messages.length - 1}
                    meta={msg.meta}
                    citations={msg.citations}
                    claims={msg.claims}
                    conflicts={msg.conflicts}
                    finalStatus={msg.finalStatus}
                    latencyMs={msg.latencyMs}
                  />
                ))}
              </AnimatePresence>

              {isStreaming && graphStatus && (
                <GraphProgress status={graphStatus} />
              )}

              {isStreaming && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-3">
                  <div className="w-7 h-7 rounded-full bg-[var(--glacier)]/10 flex items-center justify-center">
                    <StreamingIndicator />
                  </div>
                  <div className="bg-transparent border border-[var(--apres-ski)]/20 px-4 py-2.5 rounded-[var(--radius-md)] rounded-bl-[var(--radius-sm)]">
                    <StreamingIndicator />
                  </div>
                </motion.div>
              )}
              <div ref={messagesEnd} />
            </div>
          )}
        </div>

        {/* Composer */}
        <div className="border-t border-[var(--apres-ski)]/10 shrink-0 bg-[var(--midnight)]">
          <AnimatePresence>
            {uploadedFile && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
                className="max-w-3xl mx-auto px-4 pt-3">
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-[var(--radius-sm)] bg-[var(--glacier)]/10 border border-[var(--glacier)]/20">
                  <FileText size={14} className="text-[var(--glacier)]" />
                  <span className="text-xs text-[var(--glacier)]">{uploadedFile}</span>
                  <button onClick={() => setUploadedFile(null)} className="text-[var(--glacier)]/60 hover:text-[var(--glacier)] ml-1">
                    <X size={12} />
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="max-w-3xl mx-auto px-4 py-3">
            <div className="flex items-end gap-2 bg-[var(--mountainside)] rounded-[var(--radius-md)] border border-[var(--apres-ski)]/15 p-2">
              <input ref={fileInputRef} type="file" className="hidden" onChange={handleFileChange}
                accept=".txt,.pdf,.md,.html,.csv,.xlsx,.json,.py,.js" />
              <button onClick={() => fileInputRef.current?.click()} disabled={uploading}
                className="p-2 text-[var(--apres-ski)] hover:text-[var(--slopes)] transition-colors rounded-[var(--radius-sm)] hover:bg-[var(--midnight)]"
                title="Upload document">
                {uploading ? <Loader2 size={16} className="animate-spin" /> : <Paperclip size={16} />}
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
                placeholder="Message..."
                rows={1}
                disabled={isStreaming}
                className="flex-1 py-2 bg-transparent text-[var(--slopes)] placeholder-[var(--apres-ski)] text-sm focus:outline-none resize-none disabled:opacity-50 max-h-32"
              />

              <ProviderSelect />

              <button onClick={handleSend} disabled={!input.trim() || isStreaming}
                className={`p-2 rounded-[var(--radius-sm)] transition-all ${
                  isStreaming
                    ? "bg-[var(--error)]/20 text-[var(--error)] hover:bg-[var(--error)]/30"
                    : input.trim()
                      ? "bg-[var(--glacier)] text-[var(--midnight)] hover:opacity-90"
                      : "bg-[var(--apres-ski)]/20 text-[var(--apres-ski)]"
                }`}>
                {isStreaming ? <StopCircle size={16} /> : <Send size={16} />}
              </button>
            </div>
            <p className="text-center text-[10px] text-[var(--apres-ski)]/50 mt-2">
              Self-Correcting RAG — answers are verified against your documents
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
