import { create } from "zustand";
import { chatApi, Citation, Claim, Conflict } from "@/lib/api";

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  meta?: { filename?: string; status?: string };
  // Structured provenance data returned by the agent
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
  sidebarOpen: boolean;

  fetchChats: () => Promise<void>;
  createChat: (title: string) => Promise<Chat>;
  selectChat: (chatId: string) => Promise<void>;
  deleteChat: (chatId: string) => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
  toggleSidebar: () => void;
  addSystemMessage: (content: string, meta?: { filename?: string; status?: string }) => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  chats: [],
  activeChatId: null,
  messages: [],
  isStreaming: false,
  graphStatus: null,
  sidebarOpen: true,

  fetchChats: async () => {
    if (typeof window !== "undefined" && !localStorage.getItem("token")) return;
    try {
      const res = await chatApi.list({ limit: 50 });
      set({ chats: res.data.chats });
    } catch {
      // Silently ignore — token may not be set yet
    }
  },

  createChat: async (title: string) => {
    const res = await chatApi.create(title);
    const chat = res.data;
    set((s) => ({ chats: [chat, ...s.chats], activeChatId: chat.chat_id, messages: [] }));
    return chat;
  },

  selectChat: async (chatId: string) => {
    set({ activeChatId: chatId, messages: [] });
    if (typeof window !== "undefined" && !localStorage.getItem("token")) return;
    try {
      const res = await chatApi.messages(chatId, { limit: 50 });
      const messages: Message[] = (res.data.messages as Array<{
        sequence: number;
        role: "user" | "assistant" | "system";
        content: string;
        created_at: string;
      }>).map((m) => ({
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

    const userMsg: Message = { id: `user-${Date.now()}`, role: "user", content, timestamp: new Date() };
    set({ messages: [...messages, userMsg], isStreaming: true, graphStatus: null });

    try {
      // Use streaming endpoint for real-time node status
      const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const token = typeof window !== "undefined" ? localStorage.getItem("token") : "";
      const provider = typeof window !== "undefined" ? localStorage.getItem("llm_provider") || "auto" : "auto";
      const resp = await fetch(`${API_BASE}/api/v1/agent/chats/${activeChatId}/query_stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": `Bearer ${token}` },
        body: JSON.stringify({ message: content, provider }),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullAnswer = "";
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

        // Parse SSE events
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
                set({ graphStatus: { node: String(data.node), label: String(data.label), detail: data.detail ? String(data.detail) : undefined, status: "running" } });
              } else if (currentEvent === "token") {
                fullAnswer += String(data.content);
                // Update the last assistant message with streaming content
                set((s) => {
                  const msgs = [...s.messages];
                  const lastMsg = msgs[msgs.length - 1];
                  if (lastMsg && lastMsg.role === "assistant" && lastMsg.id.startsWith("stream-")) {
                    lastMsg.content = fullAnswer;
                  } else {
                    msgs.push({ id: `stream-${Date.now()}`, role: "assistant", content: fullAnswer, timestamp: new Date() });
                  }
                  return { messages: msgs };
                });
              } else if (currentEvent === "done") {
                provenance = {
                  citations: (data.citations as Citation[]) || [],
                  claims: (data.claims as Claim[]) || [],
                  conflicts: (data.conflicts as Conflict[]) || [],
                  final_status: data.final_status ? String(data.final_status) : undefined,
                  latency_ms: typeof data.latency_ms === "number" ? data.latency_ms : undefined,
                  trajectory: data.trajectory ? String(data.trajectory) : undefined,
                };
                // Replace streaming message with final answer + provenance
                set((s) => {
                  const msgs = s.messages.filter((m) => !m.id.startsWith("stream-"));
                  msgs.push({
                    id: `ai-${Date.now()}`,
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
                  return { messages: msgs, isStreaming: false, graphStatus: null };
                });
              } else if (currentEvent === "error") {
                set((s) => ({
                  messages: [...s.messages, { id: `err-${Date.now()}`, role: "assistant", content: String(data.detail || "Something went wrong."), timestamp: new Date() }],
                  isStreaming: false, graphStatus: null,
                }));
              }
            } catch {}
          }
        }
      }

      // If the stream ended without a done event (e.g. connection dropped), commit whatever we have
      if (!provenance && fullAnswer) {
        set((s) => ({
          messages: s.messages.filter((m) => !m.id.startsWith("stream-")).concat([
            { id: `ai-${Date.now()}`, role: "assistant", content: fullAnswer, timestamp: new Date() }
          ]),
          isStreaming: false,
          graphStatus: null,
        }));
      }
    } catch {
      // Fallback to non-streaming if SSE fails
      try {
        const res = await chatApi.query(activeChatId, content);
        const data = res.data;
        const aiMsg: Message = {
          id: `ai-${Date.now()}`, role: "assistant", content: data.answer, timestamp: new Date(),
          citations: data.citations,
          claims: data.claims,
          conflicts: data.conflicts,
          finalStatus: data.final_status,
          latencyMs: data.latency_ms,
        };
        set((s) => ({ messages: [...s.messages, aiMsg], isStreaming: false, graphStatus: null }));
      } catch {
        set((s) => ({
          messages: [...s.messages, { id: `err-${Date.now()}`, role: "assistant", content: "Something went wrong. Please try again.", timestamp: new Date() }],
          isStreaming: false, graphStatus: null,
        }));
      }
    }
  },

  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

  addSystemMessage: (content, meta) => {
    const msg: Message = { id: `sys-${Date.now()}`, role: "system", content, timestamp: new Date(), meta };
    set((s) => ({ messages: [...s.messages, msg] }));
  },
}));
