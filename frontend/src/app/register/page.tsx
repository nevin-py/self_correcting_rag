"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/authStore";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";

export default function RegisterPage() {
  const router = useRouter();
  const { register, isLoading } = useAuthStore();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (password !== confirm) {
      setError("Password mismatch");
      return;
    }
    if (password.length < 8) {
      setError("Minimum 8 characters required");
      return;
    }
    try {
      const registeredEmail = await register(email, password);
      router.push(`/verify-email?email=${encodeURIComponent(registeredEmail)}`);
    } catch (err) {
      const res =
        err && typeof err === "object" && "response" in err
          ? (err.response as { data?: { detail?: unknown } })?.data
          : undefined;
      const detail = res?.detail;
      if (typeof detail === "string") setError(detail);
      else if (Array.isArray(detail))
        setError(
          detail
            .map((d) => (d && typeof d === "object" && "msg" in d ? String(d.msg) : String(d)))
            .join(". ")
        );
      else setError("Registration failed");
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-void px-4 grid-bg">
      <div className="w-full max-w-[420px] border border-border bg-surface">
        <div className="border-b border-border px-6 py-4">
          <p className="label-caps">Access Terminal</p>
          <h1 className="font-display text-xl font-semibold text-text-primary">Initialize Access</h1>
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
              minLength={8}
              required
            />
          </div>
          <div>
            <label className="label-caps mb-1.5 block">Confirm Password</label>
            <Input
              type="password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              required
            />
          </div>
          {error && (
            <p className="border border-error bg-accent-glow px-3 py-2 font-mono text-xs text-error">{error}</p>
          )}
          <Button type="submit" variant="accent" size="lg" disabled={isLoading} className="w-full">
            {isLoading ? "Creating..." : "Create Access"}
          </Button>
        </form>

        <div className="border-t border-border px-6 py-4 text-center">
          <p className="text-xs text-text-muted">
            Existing operator?{" "}
            <a href="/login" className="font-mono text-accent-bright hover:text-accent">
              Sign in →
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
