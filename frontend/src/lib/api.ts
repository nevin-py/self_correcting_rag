import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: API_BASE,
  headers: { "Content-Type": "application/json" },
});

// Inject JWT token into every request
api.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

// Handle 401 — only clear token on auth endpoints, not on every 401
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const status = err.response?.status;
    const url = err.config?.url || "";

    // Only force-logout on auth endpoint failures (not on chat list, etc.)
    if (status === 401 && typeof window !== "undefined") {
      const isAuthEndpoint = url.includes("/auth/login") || url.includes("/auth/register");
      if (isAuthEndpoint) {
        localStorage.removeItem("token");
        window.location.href = new URL("/login", window.location.origin).href;
      }
      // For non-auth endpoints, let the caller handle the error
    }
    return Promise.reject(err);
  }
);

// ── Auth ────────────────────────────────────────────────────────────────────

export const authApi = {
  register: (email: string, password: string) =>
    api.post("/api/v1/auth/register", { email, password }),
  login: (email: string, password: string) =>
    api.post(
      "/api/v1/auth/login",
      new URLSearchParams({ username: email, password }),
      { headers: { "Content-Type": "application/x-www-form-urlencoded" } }
    ),
  refresh: () => api.post("/api/v1/auth/refresh"),
};

// ── Chats ───────────────────────────────────────────────────────────────────

export const chatApi = {
  list: (params?: { limit?: number; offset?: number }) =>
    api.get("/api/v1/agent/chats", { params }),
  create: (title: string) => api.post("/api/v1/agent/chats", { title }),
  get: (id: string) => api.get(`/api/v1/agent/chats/${id}`),
  delete: (id: string) => api.delete(`/api/v1/agent/chats/${id}`),
  query: (chatId: string, message: string, provider?: string) => {
    const providerPref = provider || localStorage.getItem("llm_provider") || "auto";
    return api.post(`/api/v1/agent/chats/${chatId}/query`, { message, provider: providerPref });
  },
  history: (chatId: string, params?: { limit?: number }) =>
    api.get(`/api/v1/agent/chats/${chatId}/history`, { params }),
  messages: (chatId: string, params?: { limit?: number }) =>
    api.get(`/api/v1/agent/chats/${chatId}/messages`, { params }),
};

// ── Documents ───────────────────────────────────────────────────────────────

export interface Citation {
  evidence_id: string;
  text: string;
  source_type: "document" | "web" | "llm" | "unknown";
  source_name: string;
  source_url?: string | null;
  source_date?: string | null;
  authority_score: number;
  recency_score: number;
}

export interface Claim {
  claim_id: string;
  text: string;
  status: "verified" | "partial" | "contradicted" | "unverified" | "uncertain";
  evidence_ids: string[];
  contradicting_evidence_ids: string[];
  reasoning: string;
}

export interface Conflict {
  evidence_a: string;
  evidence_b: string;
  reason: string;
  resolution: string;
  winner: string;
}

export interface QueryResponse {
  answer: string;
  chat_id: string;
  latency_ms: number;
  provider_used: string | null;
  final_status: string | null;
  claims: Claim[];
  citations: Citation[];
  conflicts: Conflict[];
}

export const documentApi = {
  upload: (chatId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return api.post(`/api/v1/documents/upload_file?chat_id=${chatId}`, form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
  status: (ingestionId: string) =>
    api.get(`/api/v1/documents/ingestions/${ingestionId}`),
};

// ── Memory ──────────────────────────────────────────────────────────────────

export const memoryApi = {
  getNodes: () => api.get("/api/v1/memory/nodes"),
  getPermanent: () => api.get("/api/v1/memory/permanent"),
};
