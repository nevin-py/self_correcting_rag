"use client";

import { useEffect, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import { PanelRightOpen, PanelRightClose } from "lucide-react";
import { useAuthStore } from "@/stores/authStore";
import { nextSessionTitle, useChatStore } from "@/stores/chatStore";
import AppShell from "@/components/layout/AppShell";
import ContextPanel from "@/components/chat/ContextPanel";
import ChatMessage from "@/components/chat/ChatMessage";
import Composer from "@/components/chat/Composer";
import PipelineTracker from "@/components/chat/PipelineTracker";
import { Button } from "@/components/ui/Button";
import { Kbd } from "@/components/ui/Kbd";

function EmptyWorkspace() {
  const router = useRouter();

  const handleNew = async () => {
    const title = nextSessionTitle(useChatStore.getState().chats);
    const chat = await useChatStore.getState().createChat(title);
    router.push(`/chat/${chat.chat_id}`);
  };

  return (
    <div className="flex h-full flex-col items-center justify-center px-6 grid-bg">
      <div className="max-w-md border border-border bg-surface p-8">
        <p className="label-caps mb-2">Self-Correcting Knowledge Workspace</p>
        <h2 className="font-display text-2xl font-semibold tracking-tight text-text-primary">
          Initialize Session
        </h2>
        <p className="mt-3 text-sm leading-relaxed text-text-secondary">
          Query your indexed knowledge base. The system retrieves evidence, verifies claims,
          detects conflicts, and corrects answers before delivery.
        </p>
        <div className="mt-6 border border-border bg-surface-inset p-3">
          <p className="font-mono text-[9px] uppercase tracking-[0.2em] text-accent-bright">
            RETRIEVE → VERIFY → DETECT CONFLICT → SEARCH → CORRECT → ANSWER
          </p>
        </div>
        <Button variant="accent" size="lg" onClick={handleNew} className="mt-6 w-full">
          Open New Session
        </Button>
      </div>
    </div>
  );
}

export default function ChatSessionPage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.sessionId as string;
  const { token, loadUser } = useAuthStore();
  const {
    messages,
    isStreaming,
    pipelineEvents,
    rightPanelOpen,
    toggleRightPanel,
    selectChat,
  } = useChatStore();
  const messagesEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadUser();
  }, [loadUser]);

  useEffect(() => {
    if (!token) router.replace("/login");
  }, [token, router]);

  useEffect(() => {
    if (sessionId) selectChat(sessionId);
  }, [sessionId, selectChat]);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  const header = (
    <header className="flex h-[var(--header-height)] shrink-0 items-center justify-between border-b border-border bg-surface px-4">
      <div className="flex items-center gap-4 min-w-0">
        <div>
          <p className="label-caps">Active Session</p>
          <p className="truncate font-mono text-xs text-text-primary">{sessionId?.slice(0, 12)}…</p>
        </div>
        <div className="hidden md:block">
          <PipelineTracker events={pipelineEvents} isStreaming={isStreaming} compact />
        </div>
      </div>
      <div className="flex items-center gap-2">
        <span className="hidden sm:flex items-center gap-1 font-mono text-[9px] text-text-muted">
          <Kbd>⌘K</Kbd> command
        </span>
        <button
          onClick={toggleRightPanel}
          className="border border-border p-1.5 text-text-muted hover:border-border-strong hover:text-text-primary lg:hidden"
          aria-label="Toggle analysis panel"
        >
          {rightPanelOpen ? <PanelRightClose size={14} /> : <PanelRightOpen size={14} />}
        </button>
      </div>
    </header>
  );

  return (
    <AppShell header={header} rightPanel={<ContextPanel />}>
      <div className="flex flex-1 flex-col min-h-0">
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <EmptyWorkspace />
          ) : (
            <div className="mx-auto max-w-3xl space-y-3 px-4 py-4">
              {messages.map((msg, i) => (
                <ChatMessage
                  key={msg.id}
                  id={msg.id}
                  role={msg.role}
                  content={msg.content}
                  timestamp={msg.timestamp}
                  isLatest={i === messages.length - 1}
                  isStreaming={isStreaming && msg.id.startsWith("stream-")}
                  meta={msg.meta}
                  citations={msg.citations}
                  claims={msg.claims}
                  conflicts={msg.conflicts}
                  finalStatus={msg.finalStatus}
                  latencyMs={msg.latencyMs}
                  estimatedCostUsd={msg.estimatedCostUsd}
                />
              ))}
              <div ref={messagesEnd} />
            </div>
          )}
        </div>
        <Composer sessionId={sessionId} />
      </div>
    </AppShell>
  );
}
