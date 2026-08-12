"use client";

import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuthStore } from "@/stores/authStore";
import { authApi } from "@/lib/api";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";

function VerifyEmailForm() {
  const router = useRouter();
  const params = useSearchParams();
  const { verifyEmail, isLoading } = useAuthStore();
  const [email, setEmail] = useState(params.get("email") || "");
  const [code, setCode] = useState("");
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    try {
      await verifyEmail(email, code.trim());
      router.push("/chat");
    } catch (err) {
      const detail =
        err && typeof err === "object" && "response" in err
          ? (err.response as { data?: { detail?: string } })?.data?.detail
          : undefined;
      setError(detail || "Verification failed");
    }
  };

  const handleResend = async () => {
    setError("");
    setInfo("");
    try {
      await authApi.resendOtp(email, "verify_email");
      setInfo("Code resent if the account exists.");
    } catch (err) {
      const detail =
        err && typeof err === "object" && "response" in err
          ? (err.response as { data?: { detail?: string } })?.data?.detail
          : undefined;
      setError(detail || "Could not resend code");
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-void px-4 grid-bg">
      <div className="w-full max-w-[420px] border border-border bg-surface">
        <div className="border-b border-border px-6 py-4">
          <p className="label-caps">Access Terminal</p>
          <h1 className="font-display text-xl font-semibold text-text-primary">Verify Email</h1>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4 p-6">
          <p className="text-xs text-text-secondary">
            Enter the 6-digit code sent to your email.
          </p>
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
              pattern="[0-9]{6}"
              required
              autoFocus
              className="font-mono tracking-widest"
            />
          </div>
          {error && (
            <p className="border border-error bg-accent-glow px-3 py-2 font-mono text-xs text-error">{error}</p>
          )}
          {info && <p className="font-mono text-xs text-text-secondary">{info}</p>}
          <Button type="submit" variant="accent" size="lg" disabled={isLoading} className="w-full">
            {isLoading ? "Verifying..." : "Verify & Enter"}
          </Button>
          <Button type="button" variant="ghost" size="md" onClick={handleResend} className="w-full">
            Resend code
          </Button>
        </form>
      </div>
    </div>
  );
}

export default function VerifyEmailPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-void" />}>
      <VerifyEmailForm />
    </Suspense>
  );
}
