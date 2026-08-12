"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/authStore";
import { useChatStore } from "@/stores/chatStore";
import Sidebar from "@/components/layout/Sidebar";
import { Brain, Plus } from "lucide-react";

export default function ChatPage() {
  const router = useRouter();
  const { token, loadUser } = useAuthStore();
  const { sidebarOpen, createChat, fetchChats } = useChatStore();

  useEffect(() => { loadUser(); fetchChats(); }, [loadUser, fetchChats]);
  useEffect(() => { if (!token) router.replace("/login"); }, [token, router]);

  const handleNewChat = async () => {
    const chat = await createChat("New Chat");
    router.push(`/chat/${chat.chat_id}`);
  };

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className={`flex-1 flex flex-col items-center justify-center transition-all duration-[var(--duration-standard)] ${sidebarOpen ? "ml-[220px]" : "ml-[72px]"}`}>
        <Brain size={48} className="text-[var(--glacier)] mb-6 opacity-40" />
        <h2 className="font-display text-2xl text-[var(--arctic)] mb-2">Start a new conversation</h2>
        <p className="text-[var(--apres-ski)] text-sm mb-8">Ask questions about your documents</p>
        <button onClick={handleNewChat}
          className="flex items-center gap-2 px-6 py-3 rounded-[var(--radius-pill)] bg-[var(--glacier)] text-[var(--midnight)] font-semibold text-sm hover:opacity-90 transition-opacity">
          <Plus size={16} /> New Chat
        </button>
      </div>
    </div>
  );
}
