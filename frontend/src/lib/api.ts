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

// Handle 401 — try refresh once, then clear session on auth endpoints
api.interceptors.response.use(
  (res) => res,
  async (err) => {
    const status = err.response?.status;
    const url = err.config?.url || "";
    const original = err.config as { _retry?: boolean; url?: string; headers?: Record<string, string> } | undefined;
    if (!original) return Promise.reject(err);

    if (
      status === 401 &&
      typeof window !== "undefined" &&
      !original._retry &&
      !url.includes("/auth/login") &&
      !url.includes("/auth/register") &&
      !url.includes("/auth/refresh") &&
      !url.includes("/auth/verify-email")
    ) {
      const refresh = localStorage.getItem("refresh_token");
      if (refresh) {
        original._retry = true;
        try {
          const res = await api.post("/api/v1/auth/refresh", { refresh_token: refresh });
          localStorage.setItem("token", res.data.access_token);
          if (res.data.refresh_token) {
            localStorage.setItem("refresh_token", res.data.refresh_token);
          }
          original.headers = original.headers || {};
          original.headers.Authorization = `Bearer ${res.data.access_token}`;
          return api.request(original);
        } catch {
          localStorage.removeItem("token");
          localStorage.removeItem("refresh_token");
        }
      }
    }

    if (status === 401 && typeof window !== "undefined") {
      const isAuthEndpoint = url.includes("/auth/login") || url.includes("/auth/register");
      if (isAuthEndpoint) {
        localStorage.removeItem("token");
        localStorage.removeItem("refresh_token");
        window.location.href = new URL("/login", window.location.origin).href;
      }
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
  verifyEmail: (email: string, code: string) =>
    api.post("/api/v1/auth/verify-email", { email, code }),
  resendOtp: (email: string, purpose: "verify_email" | "reset_password" = "verify_email") =>
    api.post("/api/v1/auth/resend-otp", { email, purpose }),
  changePassword: (current_password: string, new_password: string) =>
    api.post("/api/v1/auth/change-password", { current_password, new_password }),
  forgotPassword: (email: string) =>
    api.post("/api/v1/auth/forgot-password", { email }),
  resetPassword: (email: string, code: string, new_password: string) =>
    api.post("/api/v1/auth/reset-password", { email, code, new_password }),
  refresh: (refresh_token: string) =>
    api.post("/api/v1/auth/refresh", { refresh_token }),
  logout: (refresh_token: string) =>
    api.post("/api/v1/auth/logout", { refresh_token }),
};

export interface ProviderSettings {
  provider: string;
  has_key: boolean;
  masked_key: string | null;
  has_fallback_key: boolean;
  masked_fallback_key: string | null;
  planner_model: string | null;
  generator_model: string | null;
  verifier_model: string | null;
  default_planner_model: string;
  default_generator_model: string;
  default_verifier_model: string;
  has_server_key: boolean;
}

export const settingsApi = {
  listProviders: () => api.get<{ providers: ProviderSettings[] }>("/api/v1/settings/providers"),
  upsertProvider: (
    provider: string,
    body: {
      api_key?: string;
      fallback_api_key?: string;
      clear_fallback?: boolean;
      planner_model?: string | null;
      generator_model?: string | null;
      verifier_model?: string | null;
    }
  ) => api.put(`/api/v1/settings/providers/${provider}`, body),
  deleteProvider: (provider: string) => api.delete(`/api/v1/settings/providers/${provider}`),
};

export interface MemoryChunk {
  id: string;
  document_preview: string;
  filename: string | null;
  chat_id: string | null;
  chunk_index: number | null;
  parent_id: string | null;
  file_hash: string | null;
  chunk_type: string | null;
}

export const memoryApi = {
  getChunks: (params?: { chat_id?: string; limit?: number; offset?: number; q?: string }) =>
    api.get<{
      collection: string;
      total: number;
      limit: number;
      offset: number;
      chunks: MemoryChunk[];
    }>("/api/v1/memory/chunks", { params }),
  getStats: () =>
    api.get<{
      collection: string;
      chunk_count: number;
      files: Array<{
        filename: string;
        chat_id: string | null;
        chunk_count: number;
        file_hash: string | null;
      }>;
    }>("/api/v1/memory/stats"),
};

// ── Chats ───────────────────────────────────────────────────────────────────

export const chatApi = {
  list: (params?: { limit?: number; offset?: number }) =>
    api.get("/api/v1/agent/chats", { params }),
  create: (title: string) => api.post("/api/v1/agent/chats", { title }),
  get: (id: string) => api.get(`/api/v1/agent/chats/${id}`),
  delete: (id: string) => api.delete(`/api/v1/agent/chats/${id}`),
  purge: () => api.post<{ deleted: number }>("/api/v1/agent/chats/purge"),
  query: (chatId: string, message: string, provider?: string) => {
    const providerPref = provider || localStorage.getItem("llm_provider") || "auto";
    return api.post(`/api/v1/agent/chats/${chatId}/query`, { message, provider: providerPref });
  },
  history: (chatId: string, params?: { limit?: number }) =>
    api.get(`/api/v1/agent/chats/${chatId}/history`, { params }),
  messages: (chatId: string, params?: { limit?: number }) =>
    api.get(`/api/v1/agent/chats/${chatId}/messages`, { params }),
  chatUsage: (chatId: string) =>
    api.get<{
      token_total: number;
      estimated_cost_usd: number;
      interaction_count: number;
      chat_id: string;
    }>(`/api/v1/agent/chats/${chatId}/usage`),
  userUsage: () =>
    api.get<{
      token_total: number;
      estimated_cost_usd: number;
      interaction_count: number;
    }>("/api/v1/agent/usage"),
  contextWindow: () =>
    api.get<{ context_window_tokens: number }>("/api/v1/agent/context-window"),
};

// ── Documents ───────────────────────────────────────────────────────────────

export interface Citation {
  evidence_id: string;
  text: string;
  source_type: "document" | "web" | "llm" | "unknown";
  source_name: string;
  source_url?: string | null;
  source_date?: string | null;
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
