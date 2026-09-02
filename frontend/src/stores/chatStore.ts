import { create } from "zustand";
import { chatApi, Citation, Claim, Conflict } from "@/lib/api";
import { PipelinePhase, nodeToPhase } from "@/lib/pipeline";

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  meta?: { filename?: string; status?: string };
  citations?: Citation[];
  claims?: Claim[];
  conflicts?: Conflict[];
  finalStatus?: string;
  latencyMs?: number;
  trajectory?: string;
  tokenEstimate?: number;
  estimatedCostUsd?: number;
}

function parseProvenance(raw: string | null | undefined): Partial<Message> {
  if (!raw) return {};
  try {
    const p = JSON.parse(raw);
    return {
      citations: p.citations,
      claims: p.claims,
      conflicts: p.conflicts,
      finalStatus: p.final_status,
      latencyMs: p.latency_ms,
      trajectory: p.trajectory,
    };
  } catch {
    return {};
  }
}

// ── In-flight stream persistence (page-refresh survival) ────────────────────
// The SSE stream dies when the tab reloads — the backend never persisted the
// partial answer. We mirror the streamed text into sessionStorage so a refresh
// mid-generation RESTORES the partial instead of losing it, marked as
// interrupted.
const streamKey = (chatId: string) => `scr-stream-${chatId}`;

function persistStreamProgress(chatId: string, content: string) {
  try {
    sessionStorage.setItem(streamKey(chatId), content);
  } catch {
    /* storage full / disabled — refresh simply loses the partial */
  }
}

function clearStreamProgress(chatId: string) {
  try {
    sessionStorage.removeItem(streamKey(chatId));
  } catch {
    /* ignore */
  }
}

function restoreInterruptedStream(chatId: string) {
  if (typeof window === "undefined") return;
  let partial: string | null = null;
  try {
    partial = sessionStorage.getItem(streamKey(chatId));
  } catch {
    return;
  }
  if (!partial?.trim()) return;
  const { messages } = useChatStore.getState();
  // Skip restore if the finished answer already arrived from the DB (stream
  // completed between refresh start and message load).
  if (messages.some((m) => m.role === "assistant" && m.content === partial)) {
    clearStreamProgress(chatId);
    return;
  }
  const msgs = [...messages];
  const last = msgs[msgs.length - 1];
  if (last && last.role === "assistant") {
    last.content = partial;
    last.finalStatus = "interrupted";
  } else {
    msgs.push({
      id: `interrupted-${Date.now()}`,
      role: "assistant",
      content: partial,
      timestamp: new Date(),
      finalStatus: "interrupted",
    });
  }
  useChatStore.setState({ messages: msgs });
  clearStreamProgress(chatId);
}

/** Rough char→token estimate for the context meter. */
export function estimateLocalTokens(text: string): number {
  return Math.max(0, Math.ceil((text || "").length / 4));
}

export interface GraphStatus {
  node: string;
  label: string;
  detail?: string;
  status: "running" | "done" | "error";
  elapsedMs?: number;
}

export interface PipelineEvent {
  id: string;
  node: string;
  label: string;
  detail?: string;
  phase: PipelinePhase;
  status: "running" | "done" | "error";
  elapsedMs?: number;
  nodeMs?: number;
  timestamp: Date;
}

export interface Chat {
  chat_id: string;
  title: string;
  created_at: string;
}

export function nextSessionTitle(chats: Chat[]): string {
  let max = 0;
  for (const c of chats) {
    const m = /^Session\s+(\d+)$/i.exec((c.title || "").trim());
    if (m) max = Math.max(max, parseInt(m[1], 10));
  }
  return `Session ${String(max + 1).padStart(3, "0")}`;
}

interface ChatState {
  chats: Chat[];
  activeChatId: string | null;
  messages: Message[];
  isStreaming: boolean;
  graphStatus: GraphStatus | null;
  pipelineEvents: PipelineEvent[];
  selectedMessageId: string | null;
  sidebarCollapsed: boolean;
  sidebarOpen: boolean;
  rightPanelOpen: boolean;
  chatCostUsd: number;
  allSessionsCostUsd: number;
  contextWindowTokens: number;

  fetchChats: () => Promise<void>;
  createChat: (title: string) => Promise<Chat>;
  selectChat: (chatId: string) => Promise<void>;
  deleteChat: (chatId: string) => Promise<void>;
  purgeAllChats: () => Promise<number>;
  sendMessage: (content: string) => Promise<void>;
  toggleSidebar: () => void;
  toggleRightPanel: () => void;
  setSelectedMessage: (id: string | null) => void;
  openMessageAnalysis: (id: string) => void;
  refreshUsage: () => Promise<void>;
  addSystemMessage: (content: string, meta?: { filename?: string; status?: string }) => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  chats: [],
  activeChatId: null,
  messages: [],
  isStreaming: false,
  graphStatus: null,
  pipelineEvents: [],
  selectedMessageId: null,
  sidebarCollapsed: false,
  sidebarOpen: true,
  rightPanelOpen: true,
  chatCostUsd: 0,
  allSessionsCostUsd: 0,
  contextWindowTokens: 128000,

  fetchChats: async () => {
    if (typeof window !== "undefined" && !localStorage.getItem("token")) return;
    try {
      const res = await chatApi.list({ limit: 50 });
      set({ chats: res.data.chats });
    } catch {
      // token may not be set yet
    }
  },

  createChat: async (title: string) => {
    try {
      const res = await chatApi.create(title);
      const chat = res.data;
      set((s) => {
        const exists = s.chats.some((c) => c.chat_id === chat.chat_id);
        return {
          chats: exists ? s.chats.map((c) => (c.chat_id === chat.chat_id ? chat : c)) : [chat, ...s.chats],
          activeChatId: chat.chat_id,
          messages: exists && s.activeChatId === chat.chat_id ? s.messages : [],
          pipelineEvents: [],
          selectedMessageId: null,
        };
      });
      return chat;
    } catch (err) {
      await get().fetchChats();
      throw err;
    }
  },

  selectChat: async (chatId: string) => {
    set({ activeChatId: chatId, messages: [], pipelineEvents: [], selectedMessageId: null, chatCostUsd: 0 });
    if (typeof window !== "undefined" && !localStorage.getItem("token")) return;
    try {
      const res = await chatApi.messages(chatId, { limit: 100 });
      const messages: Message[] = (
        res.data.messages as Array<{
          sequence: number;
          role: "user" | "assistant" | "system";
          content: string;
          created_at: string;
          provenance_json?: string | null;
          token_estimate?: number | null;
          estimated_cost_usd?: number | null;
        }>
      ).map((m) => {
        const prov = m.role === "assistant" ? parseProvenance(m.provenance_json) : {};
        return {
          id: `${m.sequence}`,
          role: m.role,
          content: m.content,
          timestamp: new Date(m.created_at),
          tokenEstimate: m.token_estimate ?? undefined,
          estimatedCostUsd: m.estimated_cost_usd ?? undefined,
          ...prov,
        };
      });
      const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
      set({
        messages,
        selectedMessageId: lastAssistant?.id ?? null,
      });
      restoreInterruptedStream(chatId);
      void get().refreshUsage();
    } catch {
      set({ messages: [] });
    }
  },

  deleteChat: async (chatId: string) => {
    await chatApi.delete(chatId);
    set((s) => ({
      chats: s.chats.filter((c) => c.chat_id !== chatId),
      activeChatId: s.activeChatId === chatId ? null : s.activeChatId,
      messages: s.activeChatId === chatId ? [] : s.messages,
    }));
  },

  purgeAllChats: async () => {
    const res = await chatApi.purge();
    set({
      chats: [],
      activeChatId: null,
      messages: [],
      pipelineEvents: [],
      selectedMessageId: null,
      chatCostUsd: 0,
    });
    return res.data.deleted;
  },

  sendMessage: async (content: string) => {
    const { activeChatId } = get();
    if (!activeChatId) return;
    if (get().isStreaming) return; // one query at a time — double-clicks cost real LLM money

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content,
      timestamp: new Date(),
    };
    set({
      messages: [...get().messages, userMsg],
      isStreaming: true,
      graphStatus: null,
      pipelineEvents: [],
      selectedMessageId: null,
    });

    try {
      const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const token = typeof window !== "undefined" ? localStorage.getItem("token") : "";
      const provider =
        typeof window !== "undefined" ? localStorage.getItem("llm_provider") || "auto" : "auto";
      const resp = await fetch(`${API_BASE}/api/v1/agent/chats/${activeChatId}/query_stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ message: content, provider }),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullAnswer = "";
      let pendingCitations: Citation[] = [];
      let pendingConflicts: Conflict[] = [];
      let provenance: {
        citations?: Citation[];
        claims?: Claim[];
        conflicts?: Conflict[];
        final_status?: string;
        latency_ms?: number;
        trajectory?: string;
      } | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let currentEvent = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith("data: ") && currentEvent) {
            try {
              const data = JSON.parse(line.slice(6)) as Record<string, unknown>;

              if (currentEvent === "status") {
                const node = String(data.node);
                const status = data.status === "done" ? "done" : "running";
                const elapsedMs = typeof data.elapsed_ms === "number" ? data.elapsed_ms : undefined;
                const nodeMs = typeof data.node_ms === "number" ? data.node_ms : undefined;
                const detail = data.detail ? String(data.detail) : undefined;
                const event: PipelineEvent = {
                  id: `evt-${Date.now()}-${node}-${status}`,
                  node,
                  label: String(data.label),
                  detail,
                  phase: nodeToPhase(node),
                  status,
                  elapsedMs,
                  nodeMs,
                  timestamp: new Date(),
                };
                set((s) => {
                  const pipelineEvents = [...s.pipelineEvents];
                  // Completion replaces its own running entry; running always appends.
                  const lastIdx = [...pipelineEvents].reverse().findIndex(
                    (e) => e.node === node && e.status === "running"
                  );
                  if (status === "done" && lastIdx >= 0) {
                    pipelineEvents[pipelineEvents.length - 1 - lastIdx] = event;
                  } else {
                    pipelineEvents.push(event);
                  }
                  return {
                    graphStatus: {
                      node,
                      label: event.label,
                      detail: event.detail,
                      status,
                      elapsedMs,
                    },
                    pipelineEvents,
                  };
                });
              } else if (currentEvent === "ping") {
                const elapsedMs = typeof data.elapsed_ms === "number" ? data.elapsed_ms : undefined;
                set((s) => (s.graphStatus ? { graphStatus: { ...s.graphStatus, elapsedMs } } : {}));
              } else if (currentEvent === "answer_reset") {
                // Repair pass: the previously streamed answer is replaced.
                fullAnswer = "";
                set((s) => {
                  const msgs = [...s.messages];
                  const lastMsg = msgs[msgs.length - 1];
                  if (lastMsg && lastMsg.role === "assistant" && lastMsg.id.startsWith("stream-")) {
                    lastMsg.content = "";
                    return { messages: msgs };
                  }
                  return s;
                });
               } else if (currentEvent === "token") {
                fullAnswer += String(data.content);
                persistStreamProgress(activeChatId, fullAnswer);
                set((s) => {
                  const msgs = [...s.messages];
                  const lastMsg = msgs[msgs.length - 1];
                  if (lastMsg && lastMsg.role === "assistant" && lastMsg.id.startsWith("stream-")) {
                    lastMsg.content = fullAnswer;
                    if (pendingCitations.length && !lastMsg.citations?.length) {
                      lastMsg.citations = pendingCitations;
                      lastMsg.conflicts = pendingConflicts;
                    }
                  } else {
                    const id = `stream-${Date.now()}`;
                    msgs.push({
                      id,
                      role: "assistant",
                      content: fullAnswer,
                      timestamp: new Date(),
                      citations: pendingCitations.length ? pendingCitations : undefined,
                      conflicts: pendingConflicts.length ? pendingConflicts : undefined,
                    });
                    return { messages: msgs, selectedMessageId: id };
                  }
                  return { messages: msgs };
                });
              } else if (currentEvent === "provenance") {
                const citations = (data.citations as Citation[]) || [];
                const conflicts = (data.conflicts as Conflict[]) || [];
                if (citations.length) pendingCitations = citations;
                if (conflicts.length) pendingConflicts = conflicts;
                set((s) => {
                  const msgs = [...s.messages];
                  const lastMsg = msgs[msgs.length - 1];
                  if (lastMsg && lastMsg.role === "assistant") {
                    lastMsg.citations = citations.length ? citations : lastMsg.citations;
                    lastMsg.conflicts = conflicts.length ? conflicts : lastMsg.conflicts;
                    return { messages: msgs, selectedMessageId: lastMsg.id };
                  }
                  return s;
                });
              } else if (currentEvent === "done") {
                clearStreamProgress(activeChatId);
                provenance = {
                  citations: ((data.citations as Citation[]) || []).length
                    ? (data.citations as Citation[])
                    : pendingCitations,
                  claims: (data.claims as Claim[]) || [],
                  conflicts: ((data.conflicts as Conflict[]) || []).length
                    ? (data.conflicts as Conflict[])
                    : pendingConflicts,
                  final_status: data.final_status ? String(data.final_status) : undefined,
                  latency_ms: typeof data.latency_ms === "number" ? data.latency_ms : undefined,
                  trajectory: data.trajectory ? String(data.trajectory) : undefined,
                };
                const msgId = `ai-${Date.now()}`;
                set((s) => {
                  const msgs = s.messages.filter((m) => !m.id.startsWith("stream-"));
                  msgs.push({
                    id: msgId,
                    role: "assistant",
                    content: String(data.answer),
                    timestamp: new Date(),
                    citations: provenance?.citations,
                    claims: provenance?.claims,
                    conflicts: provenance?.conflicts,
                    finalStatus: provenance?.final_status,
                    latencyMs: provenance?.latency_ms,
                    trajectory: provenance?.trajectory,
                    tokenEstimate: typeof data.token_estimate === "number" ? data.token_estimate : undefined,
                    estimatedCostUsd:
                      typeof data.estimated_cost_usd === "number" ? data.estimated_cost_usd : undefined,
                  });
                  return {
                    messages: msgs,
                    isStreaming: false,
                    graphStatus: null,
                    selectedMessageId: msgId,
                    rightPanelOpen: true,
                    pipelineEvents: s.pipelineEvents.map((e) => ({ ...e, status: "done" as const })),
                  };
                });
                void get().refreshUsage();
              } else if (currentEvent === "error") {
                clearStreamProgress(activeChatId); // the (partial) answer is persisted server-side
                set((s) => {
                  // Keep streamed answer + evidence; don't replace with a bare error bubble.
                  const msgs = [...s.messages];
                  const last = msgs[msgs.length - 1];
                  if (last && last.role === "assistant" && (last.content || last.citations?.length)) {
                    last.finalStatus = "error";
                    if (!last.content) {
                      last.content = String(data.detail || "Operation failed.");
                    }
                    return {
                      messages: msgs,
                      isStreaming: false,
                      graphStatus: null,
                      selectedMessageId: last.id,
                    };
                  }
                  return {
                    messages: [
                      ...msgs,
                      {
                        id: `err-${Date.now()}`,
                        role: "assistant",
                        content: String(data.detail || "Operation failed."),
                        timestamp: new Date(),
                      },
                    ],
                    isStreaming: false,
                    graphStatus: null,
                  };
                });
              }
            } catch {
              // malformed SSE chunk
            }
          }
        }
      }

      if (!provenance && fullAnswer) {
        const msgId = `ai-${Date.now()}`;
        set((s) => ({
          messages: s.messages
            .filter((m) => !m.id.startsWith("stream-"))
            .concat([
              {
                id: msgId,
                role: "assistant",
                content: fullAnswer,
                timestamp: new Date(),
                citations: pendingCitations.length ? pendingCitations : undefined,
                conflicts: pendingConflicts.length ? pendingConflicts : undefined,
                finalStatus: "partial",
              },
            ]),
          isStreaming: false,
          graphStatus: null,
          selectedMessageId: msgId,
        }));
      }
    } catch {
      try {
        const res = await chatApi.query(activeChatId, content);
        const data = res.data;
        const msgId = `ai-${Date.now()}`;
        const aiMsg: Message = {
          id: msgId,
          role: "assistant",
          content: data.answer,
          timestamp: new Date(),
          citations: data.citations,
          claims: data.claims,
          conflicts: data.conflicts,
          finalStatus: data.final_status,
          latencyMs: data.latency_ms,
        };
        set((s) => ({
          messages: [...s.messages, aiMsg],
          isStreaming: false,
          graphStatus: null,
          selectedMessageId: msgId,
        }));
      } catch {
        set((s) => ({
          messages: [
            ...s.messages,
            {
              id: `err-${Date.now()}`,
              role: "assistant",
              content: "Operation failed. Retry query.",
              timestamp: new Date(),
            },
          ],
          isStreaming: false,
          graphStatus: null,
        }));
      }
    }
  },

  toggleSidebar: () =>
    set((s) => ({
      sidebarCollapsed: !s.sidebarCollapsed,
      sidebarOpen: s.sidebarCollapsed,
    })),

  toggleRightPanel: () => set((s) => ({ rightPanelOpen: !s.rightPanelOpen })),

  setSelectedMessage: (id) => set({ selectedMessageId: id }),

  openMessageAnalysis: (id) =>
    set({ selectedMessageId: id, rightPanelOpen: true }),

  refreshUsage: async () => {
    const chatId = get().activeChatId;
    try {
      const [allRes, winRes] = await Promise.all([
        chatApi.userUsage(),
        chatApi.contextWindow().catch(() => ({ data: { context_window_tokens: 128000 } })),
      ]);
      let chatCost = 0;
      if (chatId) {
        try {
          const chatRes = await chatApi.chatUsage(chatId);
          chatCost = chatRes.data.estimated_cost_usd || 0;
        } catch {
          chatCost = 0;
        }
      }
      set({
        allSessionsCostUsd: allRes.data.estimated_cost_usd || 0,
        chatCostUsd: chatCost,
        contextWindowTokens: winRes.data.context_window_tokens || 128000,
      });
    } catch {
      // ignore usage fetch errors
    }
  },

  addSystemMessage: (content, meta) => {
    const msg: Message = {
      id: `sys-${Date.now()}`,
      role: "system",
      content,
      timestamp: new Date(),
      meta,
    };
    set((s) => ({ messages: [...s.messages, msg] }));
  },
}));
