"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { authApi } from "@/lib/api";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";

export default function ForgotPasswordPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await authApi.forgotPassword(email);
      router.push(`/reset-password?email=${encodeURIComponent(email)}`);
    } catch (err) {
      const detail =
        err && typeof err === "object" && "response" in err
          ? (err.response as { data?: { detail?: string } })?.data?.detail
          : undefined;
      setError(detail || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-void px-4 grid-bg">
      <div className="w-full max-w-[420px] border border-border bg-surface">
        <div className="border-b border-border px-6 py-4">
          <p className="label-caps">Access Terminal</p>
          <h1 className="font-display text-xl font-semibold text-text-primary">Forgot Password</h1>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4 p-6">
          <p className="text-xs text-text-secondary">We will email a reset code if the account exists.</p>
          <div>
            <label className="label-caps mb-1.5 block">Email</label>
            <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoFocus />
          </div>
          {error && (
            <p className="border border-error bg-accent-glow px-3 py-2 font-mono text-xs text-error">{error}</p>
          )}
          <Button type="submit" variant="accent" size="lg" disabled={loading} className="w-full">
            {loading ? "Sending..." : "Send Reset Code"}
          </Button>
        </form>
        <div className="border-t border-border px-6 py-4 text-center">
          <a href="/login" className="font-mono text-xs text-accent-bright hover:text-accent">
            ← Back to sign in
          </a>
        </div>
      </div>
    </div>
  );
}
