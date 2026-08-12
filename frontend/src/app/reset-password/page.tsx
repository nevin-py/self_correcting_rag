"use client";

import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { authApi } from "@/lib/api";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";

function ResetPasswordForm() {
  const router = useRouter();
  const params = useSearchParams();
  const [email, setEmail] = useState(params.get("email") || "");
  const [code, setCode] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await authApi.resetPassword(email, code.trim(), password);
      router.push("/login");
    } catch (err) {
      const detail =
        err && typeof err === "object" && "response" in err
          ? (err.response as { data?: { detail?: string } })?.data?.detail
          : undefined;
      setError(detail || "Reset failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-void px-4 grid-bg">
      <div className="w-full max-w-[420px] border border-border bg-surface">
        <div className="border-b border-border px-6 py-4">
          <p className="label-caps">Access Terminal</p>
          <h1 className="font-display text-xl font-semibold text-text-primary">Reset Password</h1>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4 p-6">
          <div>
            <label className="label-caps mb-1.5 block">Email</label>
            <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
          </div>
          <div>
            <label className="label-caps mb-1.5 block">Code</label>
            <Input
              value={code}
              onChange={(e) => setCode(e.target.value)}
              maxLength={6}
              required
              className="font-mono tracking-widest"
            />
          </div>
          <div>
            <label className="label-caps mb-1.5 block">New Password</label>
            <Input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              minLength={8}
              required
            />
          </div>
          {error && (
            <p className="border border-error bg-accent-glow px-3 py-2 font-mono text-xs text-error">{error}</p>
          )}
          <Button type="submit" variant="accent" size="lg" disabled={loading} className="w-full">
            {loading ? "Saving..." : "Reset Password"}
          </Button>
        </form>
      </div>
    </div>
  );
}

export default function ResetPasswordPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-void" />}>
      <ResetPasswordForm />
    </Suspense>
  );
}
