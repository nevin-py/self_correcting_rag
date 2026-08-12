"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { useAuthStore } from "@/stores/authStore";
import { Eye, EyeOff } from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const { login, isLoading } = useAuthStore();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    try {
      await login(email, password);
      router.push("/chat");
    } catch (err) {
      const detail = err && typeof err === "object" && "response" in err
        ? (err.response as { data?: { detail?: string } })?.data?.detail
        : undefined;
      setError(detail || "Invalid email or password");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 bg-[var(--midnight)]">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        className="w-full max-w-[400px] p-8 rounded-[var(--radius-md)] bg-[var(--mountainside)] border border-[var(--apres-ski)]/10">
        <h1 className="font-display text-2xl text-[var(--arctic)] mb-6 text-center">Welcome Back</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs text-[var(--apres-ski)] mb-1.5 uppercase tracking-wider">Email</label>
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-3 rounded-[var(--radius-sm)] bg-[var(--midnight)] border border-[var(--apres-ski)]/20 focus:border-[var(--glacier)] focus:outline-none text-[var(--slopes)] text-sm transition-colors duration-[var(--duration-micro)]" required />
          </div>
          <div>
            <label className="block text-xs text-[var(--apres-ski)] mb-1.5 uppercase tracking-wider">Password</label>
            <div className="relative">
              <input type={showPw ? "text" : "password"} value={password} onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 rounded-[var(--radius-sm)] bg-[var(--midnight)] border border-[var(--apres-ski)]/20 focus:border-[var(--glacier)] focus:outline-none text-[var(--slopes)] text-sm pr-10 transition-colors duration-[var(--duration-micro)]" required />
              <button type="button" onClick={() => setShowPw(!showPw)} className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--apres-ski)] hover:text-[var(--slopes)] transition-colors">
                {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>
          {error && <p className="text-[var(--error)] text-sm">{error}</p>}
          <button type="submit" disabled={isLoading}
            className="w-full py-3 rounded-[var(--radius-pill)] bg-[var(--glacier)] text-[var(--midnight)] font-semibold text-sm hover:opacity-90 transition-opacity disabled:opacity-50">
            {isLoading ? "Signing in..." : "Sign In"}
          </button>
        </form>
        <p className="text-center text-sm text-[var(--apres-ski)] mt-6">
          Don&apos;t have an account? <a href="/register" className="text-[var(--glacier)] hover:underline">Sign up</a>
        </p>
      </motion.div>
    </div>
  );
}
