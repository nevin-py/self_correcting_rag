"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { useAuthStore } from "@/stores/authStore";
import { useChatStore } from "@/stores/chatStore";
import Sidebar from "@/components/layout/Sidebar";
import { Settings, User, Brain, Palette, Trash2, Cpu } from "lucide-react";

const PROVIDERS = [
  { id: "auto", label: "Auto", desc: "Try Groq first, fall back to OpenRouter on rate limit" },
  { id: "groq", label: "Groq", desc: "Fast, free tier. Uses Llama 3.3 70B for planning, GPT-OSS 120B for generation" },
  { id: "openrouter", label: "OpenRouter", desc: "Paid. Uses MiMo-V2.5 for planning, Claude Haiku 4.5 for generation" },
] as const;

const TABS = [
  { id: "profile", label: "Profile", icon: User },
  { id: "provider", label: "Provider", icon: Cpu },
  { id: "memory", label: "Memory", icon: Brain },
  { id: "appearance", label: "Appearance", icon: Palette },
];

export default function SettingsPage() {
  const router = useRouter();
  const { token, user, loadUser } = useAuthStore();
  const { sidebarOpen } = useChatStore();
  const [activeTab, setActiveTab] = useState("profile");
  const [confirmClear, setConfirmClear] = useState(false);
  const [provider, setProvider] = useState<string>(() => {
    if (typeof window === "undefined") return "auto";
    return localStorage.getItem("llm_provider") || "auto";
  });

  useEffect(() => { loadUser(); }, [loadUser]);
  useEffect(() => { if (!token) router.replace("/login"); }, [token, router]);

  const handleProviderChange = (id: string) => {
    setProvider(id);
    localStorage.setItem("llm_provider", id);
  };

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className={`flex-1 flex flex-col transition-all duration-[var(--duration-standard)] ${sidebarOpen ? "ml-[220px]" : "ml-[72px]"}`}>
        <header className="h-14 border-b border-[var(--apres-ski)]/10 flex items-center px-6 gap-3 shrink-0">
          <Settings size={18} className="text-[var(--apres-ski)]" />
          <h1 className="font-display text-[var(--arctic)] text-base">Settings</h1>
        </header>
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-2xl mx-auto px-6 py-8">
            {/* Tabs */}
            <div className="flex gap-1 mb-8 border-b border-[var(--apres-ski)]/10">
              {TABS.map((tab) => (
                <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2.5 text-sm transition-colors duration-[var(--duration-micro)] border-b-2 -mb-px ${
                    activeTab === tab.id ? "border-[var(--glacier)] text-[var(--arctic)]" : "border-transparent text-[var(--apres-ski)] hover:text-[var(--slopes)]"
                  }`}>
                  <tab.icon size={16} /> {tab.label}
                </button>
              ))}
            </div>

            {activeTab === "profile" && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
                <div>
                  <label className="block text-xs text-[var(--apres-ski)] mb-1.5 uppercase tracking-wider">Email</label>
                  <input value={user?.email || ""} disabled className="w-full px-4 py-3 rounded-[var(--radius-sm)] bg-[var(--mountainside)] border border-[var(--apres-ski)]/10 text-[var(--slopes)] text-sm" />
                </div>
                <div>
                  <label className="block text-xs text-[var(--apres-ski)] mb-1.5 uppercase tracking-wider">User ID</label>
                  <input value={user?.user_id || ""} disabled className="w-full px-4 py-3 rounded-[var(--radius-sm)] bg-[var(--mountainside)] border border-[var(--apres-ski)]/10 text-[var(--apres-ski)] text-sm font-mono" />
                </div>
              </motion.div>
            )}

            {activeTab === "provider" && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
                <div className="p-4 rounded-[var(--radius-md)] bg-[var(--mountainside)] border border-[var(--apres-ski)]/10">
                  <h3 className="text-sm text-[var(--arctic)] font-medium mb-2">LLM Provider</h3>
                  <p className="text-xs text-[var(--apres-ski)] mb-4">Choose which AI provider powers your queries. This setting is saved in your browser.</p>
                  <div className="space-y-3">
                    {PROVIDERS.map((p) => (
                      <button key={p.id} onClick={() => handleProviderChange(p.id)}
                        className={`w-full text-left p-3 rounded-[var(--radius-sm)] border transition-colors duration-[var(--duration-micro)] ${
                          provider === p.id
                            ? "border-[var(--glacier)] bg-[var(--glacier)]/10"
                            : "border-[var(--apres-ski)]/10 hover:border-[var(--apres-ski)]/30"
                        }`}>
                        <div className="flex items-center justify-between">
                          <span className={`text-sm font-medium ${provider === p.id ? "text-[var(--glacier)]" : "text-[var(--arctic)]"}`}>{p.label}</span>
                          {provider === p.id && <span className="text-xs text-[var(--glacier)]">Active</span>}
                        </div>
                        <p className="text-xs text-[var(--apres-ski)] mt-1">{p.desc}</p>
                      </button>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === "memory" && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
                <div className="p-4 rounded-[var(--radius-md)] bg-[var(--mountainside)] border border-[var(--apres-ski)]/10">
                  <h3 className="text-sm text-[var(--arctic)] font-medium mb-2">Data Retention</h3>
                  <p className="text-xs text-[var(--apres-ski)] mb-4">Control how long your conversation history and memories are stored.</p>
                  {!confirmClear ? (
                    <button onClick={() => setConfirmClear(true)}
                      className="flex items-center gap-2 px-4 py-2 rounded-[var(--radius-sm)] border border-[var(--error)]/30 text-[var(--error)] text-sm hover:bg-[var(--error)]/10 transition-colors">
                      <Trash2 size={14} /> Clear All Data
                    </button>
                  ) : (
                    <div className="flex items-center gap-3">
                      <span className="text-[var(--error)] text-sm">Are you sure? This cannot be undone.</span>
                      <button onClick={() => setConfirmClear(false)} className="px-3 py-1.5 rounded-[var(--radius-sm)] bg-[var(--error)] text-white text-sm">Confirm</button>
                      <button onClick={() => setConfirmClear(false)} className="px-3 py-1.5 rounded-[var(--radius-sm)] border border-[var(--apres-ski)]/20 text-[var(--slopes)] text-sm">Cancel</button>
                    </div>
                  )}
                </div>
              </motion.div>
            )}

            {activeTab === "appearance" && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <p className="text-[var(--apres-ski)] text-sm">Dark mode is the default for v1. Light theme support coming soon.</p>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
