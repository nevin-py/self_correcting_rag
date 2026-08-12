"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { useAuthStore } from "@/stores/authStore";

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
    if (password !== confirm) { setError("Passwords do not match"); return; }
    if (password.length < 8) { setError("Password must be at least 8 characters"); return; }
    try {
      await register(email, password);
      router.push("/chat");
    } catch (err) {
      const res = err && typeof err === "object" && "response" in err
        ? (err.response as { data?: { detail?: unknown } })?.data
        : undefined;
      const detail = res?.detail;
      if (typeof detail === "string") setError(detail);
      else if (Array.isArray(detail)) setError(detail.map((d) => d && typeof d === "object" && "msg" in d ? String(d.msg) : String(d)).join(". "));
      else setError("Registration failed");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 bg-[var(--midnight)]">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        className="w-full max-w-[400px] p-8 rounded-[var(--radius-md)] bg-[var(--mountainside)] border border-[var(--apres-ski)]/10">
        <h1 className="font-display text-2xl text-[var(--arctic)] mb-6 text-center">Create Account</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs text-[var(--apres-ski)] mb-1.5 uppercase tracking-wider">Email</label>
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-3 rounded-[var(--radius-sm)] bg-[var(--midnight)] border border-[var(--apres-ski)]/20 focus:border-[var(--glacier)] focus:outline-none text-[var(--slopes)] text-sm transition-colors duration-[var(--duration-micro)]" required />
          </div>
          <div>
            <label className="block text-xs text-[var(--apres-ski)] mb-1.5 uppercase tracking-wider">Password</label>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} minLength={8}
              className="w-full px-4 py-3 rounded-[var(--radius-sm)] bg-[var(--midnight)] border border-[var(--apres-ski)]/20 focus:border-[var(--glacier)] focus:outline-none text-[var(--slopes)] text-sm transition-colors duration-[var(--duration-micro)]" required />
          </div>
          <div>
            <label className="block text-xs text-[var(--apres-ski)] mb-1.5 uppercase tracking-wider">Confirm Password</label>
            <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)}
              className="w-full px-4 py-3 rounded-[var(--radius-sm)] bg-[var(--midnight)] border border-[var(--apres-ski)]/20 focus:border-[var(--glacier)] focus:outline-none text-[var(--slopes)] text-sm transition-colors duration-[var(--duration-micro)]" required />
          </div>
          {error && <p className="text-[var(--error)] text-sm">{error}</p>}
          <button type="submit" disabled={isLoading}
            className="w-full py-3 rounded-[var(--radius-pill)] bg-[var(--glacier)] text-[var(--midnight)] font-semibold text-sm hover:opacity-90 transition-opacity disabled:opacity-50">
            {isLoading ? "Creating account..." : "Create Account"}
          </button>
        </form>
        <p className="text-center text-sm text-[var(--apres-ski)] mt-6">
          Already have an account? <a href="/login" className="text-[var(--glacier)] hover:underline">Sign in</a>
        </p>
      </motion.div>
    </div>
  );
}
