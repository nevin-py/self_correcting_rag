"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/authStore";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";

export default function LoginPage() {
  const router = useRouter();
  const { login, isLoading } = useAuthStore();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    try {
      await login(email, password);
      router.push("/chat");
    } catch (err) {
      const status =
        err && typeof err === "object" && "response" in err
          ? (err.response as { status?: number; data?: { detail?: string } })?.status
          : undefined;
      const detail =
        err && typeof err === "object" && "response" in err
          ? (err.response as { data?: { detail?: string } })?.data?.detail
          : undefined;
      if (status === 403 && String(detail || "").toLowerCase().includes("not verified")) {
        router.push(`/verify-email?email=${encodeURIComponent(email)}`);
        return;
      }
      setError(detail || "Authentication failed");
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-void px-4 grid-bg">
      <div className="w-full max-w-[420px] border border-border bg-surface">
        <div className="border-b border-border px-6 py-4">
          <p className="label-caps">Access Terminal</p>
          <h1 className="font-display text-xl font-semibold text-text-primary">Sign In</h1>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4 p-6">
          <div>
            <label className="label-caps mb-1.5 block">Email</label>
            <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoFocus />
          </div>
          <div>
            <label className="label-caps mb-1.5 block">Password</label>
            <Input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          {error && (
            <p className="border border-error bg-accent-glow px-3 py-2 font-mono text-xs text-error">{error}</p>
          )}
          <Button type="submit" variant="accent" size="lg" disabled={isLoading} className="w-full">
            {isLoading ? "Authenticating..." : "Enter Workspace"}
          </Button>
        </form>

        <div className="border-t border-border px-6 py-4 text-center space-y-2">
          <p className="text-xs text-text-muted">
            <a href="/forgot-password" className="font-mono text-accent-bright hover:text-accent">
              Forgot password?
            </a>
          </p>
          <p className="text-xs text-text-muted">
            No access?{" "}
            <a href="/register" className="font-mono text-accent-bright hover:text-accent">
              Initialize account →
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
