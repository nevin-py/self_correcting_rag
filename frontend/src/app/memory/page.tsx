"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { useAuthStore } from "@/stores/authStore";
import MemoryGraph from "@/components/memory/MemoryGraph";
import { ArrowLeft, Brain, Search } from "lucide-react";

export default function MemoryPage() {
  const router = useRouter();
  const { token, loadUser } = useAuthStore();
  useEffect(() => { loadUser(); }, [loadUser]);
  useEffect(() => { if (!token) router.replace("/login"); }, [token, router]);

  return (
    <div className="min-h-screen bg-[var(--midnight)] flex flex-col">
      <header className="border-b border-[var(--apres-ski)]/10 px-6 h-14 flex items-center gap-4 shrink-0">
        <button onClick={() => router.push("/chat")} className="p-2 rounded-[var(--radius-sm)] hover:bg-[var(--mountainside)] transition-colors">
          <ArrowLeft size={18} className="text-[var(--apres-ski)]" />
        </button>
        <Brain size={18} className="text-[var(--glacier)]" />
        <h1 className="font-display text-[var(--arctic)] text-base">Permanent Memory</h1>
        <div className="flex-1" />
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-[var(--radius-pill)] bg-[var(--mountainside)] border border-[var(--apres-ski)]/10">
          <Search size={14} className="text-[var(--apres-ski)]" />
          <input placeholder="Filter memories..." className="bg-transparent text-sm text-[var(--slopes)] placeholder-[var(--apres-ski)] focus:outline-none w-40" />
        </div>
      </header>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.6 }} className="flex-1 flex items-center justify-center py-8">
        <MemoryGraph />
      </motion.div>
      <div className="text-center pb-6">
        <p className="text-[var(--apres-ski)] text-xs">Nodes represent concepts learned from your conversations. The graph breathes and drifts organically.</p>
      </div>
    </div>
  );
}
