"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { useAuthStore } from "@/stores/authStore";
import { useChatStore } from "@/stores/chatStore";
import Sidebar from "@/components/layout/Sidebar";
import { Clock, MessageSquare, Search } from "lucide-react";

export default function HistoryPage() {
  const router = useRouter();
  const { token, loadUser } = useAuthStore();
  const { chats, sidebarOpen, fetchChats, selectChat } = useChatStore();

  useEffect(() => { loadUser(); fetchChats(); }, [loadUser, fetchChats]);
  useEffect(() => { if (!token) router.replace("/login"); }, [token, router]);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className={`flex-1 flex flex-col transition-all duration-[var(--duration-standard)] ${sidebarOpen ? "ml-[220px]" : "ml-[72px]"}`}>
        <header className="h-14 border-b border-[var(--apres-ski)]/10 flex items-center px-6 gap-3 shrink-0">
          <Clock size={18} className="text-[var(--apres-ski)]" />
          <h1 className="font-display text-[var(--arctic)] text-base">History</h1>
          <div className="flex-1" />
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-[var(--radius-pill)] bg-[var(--mountainside)] border border-[var(--apres-ski)]/10">
            <Search size={14} className="text-[var(--apres-ski)]" />
            <input placeholder="Search..." className="bg-transparent text-sm text-[var(--slopes)] placeholder-[var(--apres-ski)] focus:outline-none w-40" />
          </div>
        </header>
        <div className="flex-1 overflow-y-auto">
          {chats.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <Clock size={32} className="text-[var(--apres-ski)] mb-3" />
              <p className="text-[var(--arctic)] font-display mb-1">No conversations yet</p>
              <button onClick={() => router.push("/chat")} className="text-[var(--glacier)] text-sm hover:underline mt-1">Start one →</button>
            </div>
          ) : (
            <div className="divide-y divide-[var(--apres-ski)]/10">
              {chats.map((chat, i) => (
                <motion.div key={chat.chat_id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.03 }}
                  onClick={() => { selectChat(chat.chat_id); router.push(`/chat/${chat.chat_id}`); }}
                  className="flex items-center gap-4 px-6 py-4 hover:bg-[var(--mountainside)] cursor-pointer transition-colors duration-[var(--duration-micro)]">
                  <MessageSquare size={16} className="text-[var(--apres-ski)] shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-[var(--arctic)] truncate">{chat.title}</p>
                    <p className="text-xs text-[var(--apres-ski)]">{new Date(chat.created_at).toLocaleDateString()}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
