"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/authStore";

/** Wait for a server-validated session before treating the user as signed in. */
export function useRequireAuth() {
  const router = useRouter();
  const { token, authReady, bootstrapAuth } = useAuthStore();

  useEffect(() => {
    void bootstrapAuth();
  }, [bootstrapAuth]);

  useEffect(() => {
    if (!authReady) return;
    if (!token) router.replace("/login");
  }, [authReady, token, router]);

  return { token, authReady, isAuthenticated: authReady && !!token };
}
