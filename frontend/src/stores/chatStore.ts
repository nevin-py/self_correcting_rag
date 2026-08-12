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
}

export interface GraphStatus {
  node: string;
  label: string;
  detail?: string;
  status: "running" | "done" | "error";
}

export interface PipelineEvent {
  id: string;
  node: string;
  label: string;
  detail?: string;
  phase: PipelinePhase;
  status: "running" | "done" | "error";
  timestamp: Date;
}

export interface Chat {
  chat_id: string;
  title: string;
  created_at: string;
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

  fetchChats: () => Promise<void>;
  createChat: (title: string) => Promise<Chat>;
  selectChat: (chatId: string) => Promise<void>;
  deleteChat: (chatId: string) => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
  toggleSidebar: () => void;
  toggleRightPanel: () => void;
  setSelectedMessage: (id: string | null) => void;
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
    const res = await chatApi.create(title);
    const chat = res.data;
    set((s) => ({
      chats: [chat, ...s.chats],
      activeChatId: chat.chat_id,
      messages: [],
      pipelineEvents: [],
      selectedMessageId: null,
    }));
    return chat;
  },

  selectChat: async (chatId: string) => {
    set({ activeChatId: chatId, messages: [], pipelineEvents: [], selectedMessageId: null });
    if (typeof window !== "undefined" && !localStorage.getItem("token")) return;
    try {
      const res = await chatApi.messages(chatId, { limit: 50 });
      const messages: Message[] = (
        res.data.messages as Array<{
          sequence: number;
          role: "user" | "assistant" | "system";
          content: string;
          created_at: string;
        }>
      ).map((m) => ({
        id: `${m.sequence}`,
        role: m.role,
        content: m.content,
        timestamp: new Date(m.created_at),
      }));
      set({ messages });
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

  sendMessage: async (content: string) => {
    const { activeChatId, messages } = get();
    if (!activeChatId) return;

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content,
      timestamp: new Date(),
    };
    set({
      messages: [...messages, userMsg],
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
                const event: PipelineEvent = {
                  id: `evt-${Date.now()}-${node}`,
                  node,
                  label: String(data.label),
                  detail: data.detail ? String(data.detail) : undefined,
                  phase: nodeToPhase(node),
                  status: "running",
                  timestamp: new Date(),
                };
                set((s) => ({
                  graphStatus: {
                    node,
                    label: event.label,
                    detail: event.detail,
                    status: "running",
                  },
                  pipelineEvents: [...s.pipelineEvents, event],
                }));
              } else if (currentEvent === "token") {
                fullAnswer += String(data.content);
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
                  });
                  return {
                    messages: msgs,
                    isStreaming: false,
                    graphStatus: null,
                    selectedMessageId: msgId,
                    pipelineEvents: s.pipelineEvents.map((e) => ({ ...e, status: "done" as const })),
                  };
                });
              } else if (currentEvent === "error") {
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
