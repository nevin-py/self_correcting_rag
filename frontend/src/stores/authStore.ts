import { create } from "zustand";
import { authApi } from "@/lib/api";

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
}

function persistSession(access: string, refresh: string | undefined, email: string, set: (s: Partial<AuthState>) => void) {
  localStorage.setItem("token", access);
  if (refresh) localStorage.setItem("refresh_token", refresh);
  const payload = JSON.parse(atob(access.split(".")[1]));
  set({ user: { user_id: payload.sub, email }, token: access, isLoading: false });
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: typeof window !== "undefined" ? localStorage.getItem("token") : null,
  isLoading: false,

  login: async (email, password) => {
    set({ isLoading: true });
    try {
      const res = await authApi.login(email, password);
      persistSession(res.data.access_token, res.data.refresh_token, email, set);
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
      persistSession(res.data.access_token, res.data.refresh_token, email, set);
    } catch (err) {
      set({ isLoading: false });
      throw err;
    }
  },

  logout: () => {
    const refresh = typeof window !== "undefined" ? localStorage.getItem("refresh_token") : null;
    if (refresh) {
      authApi.logout(refresh).catch(() => undefined);
    }
    localStorage.removeItem("token");
    localStorage.removeItem("refresh_token");
    set({ user: null, token: null });
  },

  loadUser: () => {
    if (typeof window === "undefined") return;
    const token = localStorage.getItem("token");
    if (token) {
      try {
        const payload = JSON.parse(atob(token.split(".")[1]));
        set({ user: { user_id: payload.sub, email: "" }, token });
      } catch {
        localStorage.removeItem("token");
        localStorage.removeItem("refresh_token");
      }
    }
  },
}));
