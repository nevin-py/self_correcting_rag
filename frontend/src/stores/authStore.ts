import { create } from "zustand";
import { authApi, refreshAccessToken, broadcastAuthEvent, onAuthEvent, API_BASE } from "@/lib/api";

interface User {
  user_id: string;
  email: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  /** False until bootstrapAuth has proven (or rejected) a session with the API. */
  authReady: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<string>;
  verifyEmail: (email: string, code: string) => Promise<void>;
  logout: () => void;
  bootstrapAuth: () => Promise<void>;
}

function dbg(
  hypothesisId: string,
  location: string,
  message: string,
  data: Record<string, unknown>
) {
  // #region agent log
  fetch("http://127.0.0.1:7414/ingest/c9f169d4-33bb-4576-a7c6-7358a7e9745d", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "05b494" },
    body: JSON.stringify({
      sessionId: "05b494",
      hypothesisId,
      location,
      message,
      data,
      timestamp: Date.now(),
      runId: "login-ui",
    }),
  }).catch(() => undefined);
  // #endregion
}

/** Decode the JWT `sub` claim; never throws (a corrupt token must not crash the UI). */
function parseJwtSub(access: string): string | null {
  try {
    const payload = JSON.parse(atob(access.split(".")[1]));
    return typeof payload?.sub === "string" ? payload.sub : null;
  } catch {
    return null;
  }
}

function persistSession(
  access: string,
  email: string,
  set: (s: Partial<AuthState>) => void
) {
  bootstrapEpoch += 1;
  // Access token in localStorage (short-lived); the refresh token is an
  // httpOnly cookie set by the server — never stored in JS-readable storage.
  localStorage.setItem("token", access);
  set({
    user: { user_id: parseJwtSub(access) ?? "", email },
    token: access,
    isLoading: false,
    authReady: true,
  });
}

function clearSession(set: (s: Partial<AuthState>) => void) {
  localStorage.removeItem("token");
  localStorage.removeItem("refresh_token");
  set({ user: null, token: null, isLoading: false, authReady: true });
}

let bootstrapInFlight: Promise<void> | null = null;
let bootstrapEpoch = 0;

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  // Do not hydrate from localStorage for routing — a leftover JWT is not a session.
  token: null,
  isLoading: false,
  authReady: false,

  login: async (email, password) => {
    set({ isLoading: true });
    try {
      const res = await authApi.login(email, password);
      dbg("F", "authStore.ts:login", "login_ok", {
        hasAccess: Boolean(res.data?.access_token),
        origin: typeof window !== "undefined" ? window.location.origin : "",
        apiBase: API_BASE,
      });
      persistSession(res.data.access_token, email, set);
    } catch (err) {
      const status =
        err && typeof err === "object" && "response" in err
          ? (err as { response?: { status?: number } }).response?.status
          : undefined;
      const msg = err instanceof Error ? err.message : String(err);
      dbg("F", "authStore.ts:login", "login_fail", {
        status: status ?? null,
        msg,
        origin: typeof window !== "undefined" ? window.location.origin : "",
        apiBase: API_BASE,
      });
      set({ isLoading: false });
      throw err;
    }
  },

  register: async (email, password) => {
    set({ isLoading: true });
    try {
      await authApi.register(email, password);
      set({ isLoading: false });
      return email;
    } catch (err) {
      set({ isLoading: false });
      throw err;
    }
  },

  verifyEmail: async (email, code) => {
    set({ isLoading: true });
    try {
      const res = await authApi.verifyEmail(email, code);
      persistSession(res.data.access_token, email, set);
    } catch (err) {
      set({ isLoading: false });
      throw err;
    }
  },

  logout: () => {
    // Cookie-based logout: the browser sends the httpOnly refresh cookie. The
    // backend revokes ALL of the user's refresh tokens (see the logout route:
    // this closes the in-flight-rotation race that resurrected sessions), and
    // every other tab is told to drop its state immediately via BroadcastChannel.
    authApi.logout().catch(() => undefined);
    localStorage.removeItem("token_refreshed_at");
    broadcastAuthEvent("logout");
    clearSession(set);
  },

  bootstrapAuth: async () => {
    // Prove the stored session is still valid on the server — WITHOUT rotating
    // the refresh token. The previous implementation called /auth/refresh on
    // every cold load: one rotation per page load, and two open tabs rotating
    // concurrently trip reuse detection and kill each other's sessions.
    if (typeof window === "undefined") return;
    if (bootstrapInFlight) return bootstrapInFlight;
    const epoch = bootstrapEpoch;

    bootstrapInFlight = (async () => {
      const access = localStorage.getItem("token");
      dbg("G", "authStore.ts:bootstrapAuth", "start", {
        hasAccess: Boolean(access),
        epoch,
        origin: window.location.origin,
        apiBase: API_BASE,
      });

      // 1. Access token present → validate it against /auth/me (no rotation).
      if (access) {
        try {
          const res = await authApi.me();
          if (epoch !== bootstrapEpoch) return;
          persistSession(access, res.data.email, set);
          dbg("H", "authStore.ts:bootstrapAuth", "me_ok", {});
          return;
        } catch {
          dbg("H", "authStore.ts:bootstrapAuth", "me_fail_try_refresh", {});
        }
      }

      try {
        const newAccess = await refreshAccessToken();
        if (epoch !== bootstrapEpoch) {
          dbg("G", "authStore.ts:bootstrapAuth", "stale_skip_after_refresh", { epoch, current: bootstrapEpoch });
          return;
        }
        if (!newAccess) {
          dbg("G", "authStore.ts:bootstrapAuth", "refresh_empty_clear", { epoch });
          clearSession(set);
          return;
        }
        try {
          const me = await authApi.me();
          persistSession(newAccess, me.data.email, set);
        } catch {
          persistSession(newAccess, "", set);
        }
      } catch {
        if (epoch !== bootstrapEpoch) return;
        dbg("G", "authStore.ts:bootstrapAuth", "refresh_throw_clear", { epoch });
        clearSession(set);
      }
    })().finally(() => {
      bootstrapInFlight = null;
    });

    return bootstrapInFlight;
  },
}));

// Other tabs: on a logout broadcast, drop local state immediately instead of
// discovering it on the next 401 (the server has already revoked everything).
if (typeof window !== "undefined") {
  onAuthEvent((type) => {
    if (type === "logout") {
      localStorage.removeItem("token");
      localStorage.removeItem("refresh_token");
      localStorage.removeItem("token_refreshed_at");
      useAuthStore.setState({ user: null, token: null, isLoading: false, authReady: true });
    }
  });
}
