import { create } from "zustand";
import { authApi, refreshAccessToken, broadcastAuthEvent, onAuthEvent } from "@/lib/api";

interface User {
  user_id: string;
  email: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<string>;
  verifyEmail: (email: string, code: string) => Promise<void>;
  logout: () => void;
  loadUser: () => void;
  bootstrapAuth: () => Promise<void>;
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
  // Access token in localStorage (short-lived); the refresh token is an
  // httpOnly cookie set by the server — never stored in JS-readable storage.
  localStorage.setItem("token", access);
  set({ user: { user_id: parseJwtSub(access) ?? "", email }, token: access, isLoading: false });
}

function clearSession(set: (s: Partial<AuthState>) => void) {
  localStorage.removeItem("token");
  localStorage.removeItem("refresh_token");
  set({ user: null, token: null, isLoading: false });
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: typeof window !== "undefined" ? localStorage.getItem("token") : null,
  isLoading: false,

  login: async (email, password) => {
    set({ isLoading: true });
    try {
      const res = await authApi.login(email, password);
      persistSession(res.data.access_token, email, set);
    } catch (err) {
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

  loadUser: () => {
    if (typeof window === "undefined") return;
    const token = localStorage.getItem("token");
    if (token) {
      const sub = parseJwtSub(token);
      if (sub) {
        set({ user: { user_id: sub, email: "" }, token });
      } else {
        clearSession(set);
      }
    }
  },

  bootstrapAuth: async () => {
    // Prove the stored session is still valid on the server — WITHOUT rotating
    // the refresh token. The previous implementation called /auth/refresh on
    // every cold load: one rotation per page load, and two open tabs rotating
    // concurrently trip reuse detection and kill each other's sessions.
    if (typeof window === "undefined") return;
    const access = localStorage.getItem("token");
    // NOTE: the refresh token now lives in an httpOnly cookie — its presence
    // can't be checked from JS. When no access token exists we still try the
    // refresh round-trip (empty body); a missing cookie just fails and the
    // session is cleared.
    set({ isLoading: true });

    // 1. Access token present → validate it against /auth/me (no rotation).
    if (access) {
      try {
        const res = await authApi.me();
        persistSession(access, res.data.email, set);
        return;
      } catch {
        // invalid/expired access — fall through to a single refresh attempt
      }
    }

    // 2. Rotate once (httpOnly cookie carries the token), then read the email
    //    from /auth/me with the new token — through the SAME cross-tab-coordinated
    //    refresh the 401 interceptor uses (Web Locks + freshness window), so a
    //    multi-tab cold start performs at most ONE rotation for the whole browser.
    try {
      const newAccess = await refreshAccessToken();
      if (!newAccess) {
        clearSession(set);
        return;
      }
      set({ token: newAccess });
      try {
        const me = await authApi.me();
        set({ user: { user_id: me.data.user_id, email: me.data.email }, isLoading: false });
      } catch {
        set({ user: { user_id: parseJwtSub(newAccess) ?? "", email: "" }, isLoading: false });
      }
    } catch {
      clearSession(set);
    }
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
      useAuthStore.setState({ user: null, token: null, isLoading: false });
    }
  });
}
