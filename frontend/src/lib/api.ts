import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: API_BASE,
  headers: { "Content-Type": "application/json" },
  // Send the httpOnly refresh_token cookie on auth calls (and any future
  // cookie-based endpoints). No CSRF surface: auth uses bearer tokens, and
  // CORS only echoes exact allowed origins.
  withCredentials: true,
});

/** Citation/source links: the backend emits absolute http(s) URLs for web
 * sources and RELATIVE signed paths for uploaded documents. Prefix relative
 * ones with the API base so links open against the right origin. */
export function resolveSourceUrl(url?: string | null): string {
  if (!url) return "";
  if (url.startsWith("/") && !url.startsWith("//")) return `${API_BASE}${url}`;
  return url;
}

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

// Handle 401 — try refresh ONCE (single-flight, cross-tab), then clear session
//
// Single-flight matters doubly: refresh tokens are single-use (rotated
// server-side), AND the rotation is per-BROWSER (httpOnly cookie jar shared by
// all tabs). Per-tab promises don't serialize tabs — three tabs loading at once
// produced three sequential rotations (each legitimate on its own, but every
// other tab's stored access token goes stale, and the NEXT rotation from a
// stale tab trips reuse detection and kills the whole session).
//
// Two layers of coordination:
// 1. Web Locks API — only ONE tab rotates at a time, across tabs.
// 2. Freshness window — if another tab refreshed <10s ago, adopt its token
//    from localStorage instead of rotating again.
//
// Backend backstop: logout (and reuse detection) revoke ALL of the user's
// tokens, so a rotation that races a logout is worthless server-side.
let refreshInFlight: Promise<string | null> | null = null;
const REFRESH_FRESH_MS = 10_000;
const REFRESH_LOCK_NAME = "scrag-auth-refresh";

/** Broadcast auth events to every other tab of this origin. */
const authChannel: BroadcastChannel | null =
  typeof window !== "undefined" && "BroadcastChannel" in window
    ? new BroadcastChannel("scrag-auth")
    : null;

export function broadcastAuthEvent(type: string): void {
  try {
    authChannel?.postMessage({ type });
  } catch {
    // channel unavailable — non-fatal
  }
}

/** Subscribe other tabs to auth events (returns an unsubscribe fn). */
export function onAuthEvent(cb: (type: string) => void): () => void {
  if (!authChannel) return () => undefined;
  const handler = (ev: MessageEvent) => {
    if (ev.data?.type) cb(ev.data.type as string);
  };
  authChannel.addEventListener("message", handler);
  return () => authChannel.removeEventListener("message", handler);
}

/**
 * The ONE way any tab rotates the session. Serialized across tabs via Web
 * Locks, deduped via the freshness window. Returns the new access token or
 * null when the session is genuinely dead.
 */
export async function refreshAccessToken(): Promise<string | null> {
  if (refreshInFlight) return refreshInFlight;

  const rotate = async (): Promise<string | null> => {
    // Another tab may have JUST rotated — adopt its token instead of
    // burning another single-use rotation (the freshness window is short
    // enough that a legitimately-expired token path is unaffected).
    const refreshedAt = Number(localStorage.getItem("token_refreshed_at") || 0);
    const stored = localStorage.getItem("token");
    if (stored && Date.now() - refreshedAt < REFRESH_FRESH_MS) return stored;

    // Legacy localStorage refresh token still sent in the body so sessions
    // issued before the cookie rollout survive.
    const legacy = localStorage.getItem("refresh_token");
    try {
      // Bare axios call — going through `api` would re-enter this interceptor.
      const res = await axios.post(
        `${API_BASE}/api/v1/auth/refresh`,
        legacy ? { refresh_token: legacy } : {},
        { withCredentials: true }
      );
      localStorage.setItem("token", res.data.access_token);
      localStorage.setItem("token_refreshed_at", String(Date.now()));
      // Migrate off localStorage refresh tokens; the cookie is the source of
      // truth from here on.
      localStorage.removeItem("refresh_token");
      return res.data.access_token as string;
    } catch {
      localStorage.removeItem("token");
      localStorage.removeItem("refresh_token");
      localStorage.removeItem("token_refreshed_at");
      return null;
    }
  };

  refreshInFlight = (async () => {
    try {
      const locks = typeof navigator !== "undefined" ? (navigator as Navigator & { locks?: { request: (name: string, cb: () => Promise<string | null>) => Promise<string | null> } }).locks : undefined;
      if (locks?.request) {
        return await locks.request(REFRESH_LOCK_NAME, rotate);
      }
      return await rotate();
    } finally {
      // Release AFTER the storage writes so awaiters see a settled state.
      refreshInFlight = null;
    }
  })();
  return refreshInFlight;
}

api.interceptors.response.use(
  (res) => res,
  async (err) => {
    const status = err.response?.status;
    const url = err.config?.url || "";
    const original = err.config as { _retry?: boolean; url?: string; headers?: Record<string, string> } | undefined;
    if (!original) return Promise.reject(err);

    const isAuthEndpoint =
      url.includes("/auth/login") ||
      url.includes("/auth/register") ||
      url.includes("/auth/refresh") ||
      url.includes("/auth/verify-email") ||
      url.includes("/auth/resend-otp") ||
      url.includes("/auth/forgot-password") ||
      url.includes("/auth/reset-password") ||
      url.includes("/auth/logout");

    if (
      status === 401 &&
      typeof window !== "undefined" &&
      !original._retry &&
      !isAuthEndpoint
    ) {
      const newToken = await refreshAccessToken();
      if (newToken) {
        original._retry = true;
        original.headers = original.headers || {};
        original.headers.Authorization = `Bearer ${newToken}`;
        return api.request(original);
      }
    }

    if (status === 401 && typeof window !== "undefined") {
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
  me: () => api.get<{ user_id: string; email: string; email_verified: boolean; is_active: boolean }>("/api/v1/auth/me"),
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
  refresh: (refresh_token?: string) =>
    api.post("/api/v1/auth/refresh", refresh_token ? { refresh_token } : {}),
  logout: (refresh_token?: string) =>
    api.post("/api/v1/auth/logout", refresh_token ? { refresh_token } : {}),
};

export interface ProviderSettings {
  provider: string;
  has_key: boolean;
  masked_key: string | null;
  has_fallback_key: boolean;
  masked_fallback_key: string | null;
  client_family: string;
  base_url: string | null;
  default_base_url: string | null;
  default_family: string;
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
  revealProvider: (provider: string) =>
    api.get<{ provider: string; api_key: string | null; fallback_api_key: string | null }>(
      `/api/v1/settings/providers/${provider}/reveal`
    ),
  upsertProvider: (
    provider: string,
    body: {
      api_key?: string;
      fallback_api_key?: string;
      clear_fallback?: boolean;
      client_family?: string;
      base_url?: string | null;
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
  cite_key?: string | null;   // "E1", "E2"… — matches [E#] markers in the answer text
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
