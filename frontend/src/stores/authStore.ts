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
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
  loadUser: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: typeof window !== "undefined" ? localStorage.getItem("token") : null,
  isLoading: false,

  login: async (email, password) => {
    set({ isLoading: true });
    try {
      const res = await authApi.login(email, password);
      const token = res.data.access_token;
      localStorage.setItem("token", token);

      // Decode JWT to get user info
      const payload = JSON.parse(atob(token.split(".")[1]));
      set({ user: { user_id: payload.sub, email }, token, isLoading: false });
    } catch (err) {
      set({ isLoading: false });
      throw err;
    }
  },

  register: async (email, password) => {
    set({ isLoading: true });
    try {
      await authApi.register(email, password);
      // Auto-login after registration
      const res = await authApi.login(email, password);
      const token = res.data.access_token;
      localStorage.setItem("token", token);
      const payload = JSON.parse(atob(token.split(".")[1]));
      set({ user: { user_id: payload.sub, email }, token, isLoading: false });
    } catch (err) {
      set({ isLoading: false });
      throw err;
    }
  },

  logout: () => {
    localStorage.removeItem("token");
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
      }
    }
  },
}));
