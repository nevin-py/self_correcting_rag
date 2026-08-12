"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { useChatStore } from "@/stores/chatStore";
import { useAuthStore } from "@/stores/authStore";
import { useRouter, usePathname } from "next/navigation";
import {
  MessageSquare,
  Brain,
  Clock,
  Settings,
  Plus,
  Trash2,
  LogOut,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";

const NAV_ITEMS = [
  { icon: MessageSquare, label: "Chat", href: "/chat" },
  { icon: Brain, label: "Memory", href: "/memory" },
  { icon: Clock, label: "History", href: "/history" },
  { icon: Settings, label: "Settings", href: "/settings" },
];

export default function Sidebar() {
  const router = useRouter();
  const pathname = usePathname();
  const { chats, activeChatId, fetchChats, createChat, selectChat, deleteChat } = useChatStore();
  const { logout } = useAuthStore();
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    fetchChats();
  }, [fetchChats]);

  const handleNewChat = async () => {
    const title = `Chat ${chats.length + 1}`;
    const chat = await createChat(title);
    router.push(`/chat/${chat.chat_id}`);
  };

  return (
    <aside
      className={`h-screen bg-[var(--midnight)] border-r border-[var(--apres-ski)]/20 flex flex-col fixed left-0 top-0 z-30 transition-all duration-[var(--duration-standard)] ${
        collapsed ? "w-[72px]" : "w-[220px]"
      }`}
    >
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-3 border-b border-[var(--apres-ski)]/20">
        {!collapsed && (
          <span className="font-display text-[var(--arctic)] text-sm font-medium tracking-wide">
            RAG
          </span>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1.5 rounded-[var(--radius-sm)] hover:bg-[var(--mountainside)] transition-colors duration-[var(--duration-micro)]"
        >
          {collapsed ? <PanelLeftOpen size={16} className="text-[var(--apres-ski)]" /> : <PanelLeftClose size={16} className="text-[var(--apres-ski)]" />}
        </button>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-2 px-2 space-y-0.5">
        {NAV_ITEMS.map((item) => {
          const active = pathname === item.href || pathname.startsWith(item.href + "/");
          return (
            <a
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-[var(--radius-sm)] transition-colors duration-[var(--duration-micro)] relative group ${
                active
                  ? "bg-[var(--mountainside)] text-[var(--arctic)]"
                  : "text-[var(--apres-ski)] hover:bg-[var(--mountainside)]/50 hover:text-[var(--slopes)]"
              }`}
            >
              {active && (
                <motion.div
                  layoutId="nav-indicator"
                  className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-full bg-[var(--glacier)]"
                  transition={{ type: "spring", stiffness: 500, damping: 30 }}
                />
              )}
              <item.icon size={18} />
              {!collapsed && <span className="text-sm">{item.label}</span>}
            </a>
          );
        })}

        {/* New Chat */}
        <button
          onClick={handleNewChat}
          className="flex items-center gap-3 px-3 py-2.5 rounded-[var(--radius-sm)] text-[var(--apres-ski)] hover:bg-[var(--mountainside)]/50 hover:text-[var(--slopes)] transition-colors duration-[var(--duration-micro)] w-full mt-2"
        >
          <Plus size={18} />
          {!collapsed && <span className="text-sm">New Chat</span>}
        </button>

        {/* Session list */}
        {!collapsed && chats.length > 0 && (
          <div className="mt-4 space-y-0.5">
            <p className="px-3 text-[10px] font-medium uppercase tracking-wider text-[var(--apres-ski)]/60 mb-1">
              Recent
            </p>
            {chats.slice(0, 10).map((chat) => (
              <div
                key={chat.chat_id}
                onClick={() => {
                  selectChat(chat.chat_id);
                  router.push(`/chat/${chat.chat_id}`);
                }}
                className={`group flex items-center gap-2 px-3 py-2 rounded-[var(--radius-sm)] cursor-pointer transition-colors duration-[var(--duration-micro)] ${
                  activeChatId === chat.chat_id
                    ? "bg-[var(--mountainside)] text-[var(--arctic)]"
                    : "text-[var(--apres-ski)] hover:bg-[var(--mountainside)]/50 hover:text-[var(--slopes)]"
                }`}
              >
                <MessageSquare size={14} className="shrink-0" />
                <span className="truncate text-xs flex-1">{chat.title}</span>
                <button
                  onClick={(e) => { e.stopPropagation(); deleteChat(chat.chat_id); }}
                  className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-[var(--error)]/20 hover:text-[var(--error)] transition"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
        )}
      </nav>

      {/* User */}
      <div className="p-3 border-t border-[var(--apres-ski)]/20">
        <button
          onClick={() => { logout(); router.push("/"); }}
          className="flex items-center gap-2 w-full px-3 py-2 rounded-[var(--radius-sm)] text-[var(--apres-ski)] hover:bg-[var(--mountainside)] hover:text-[var(--slopes)] transition-colors duration-[var(--duration-micro)] text-sm"
        >
          <LogOut size={16} />
          {!collapsed && <span>Sign Out</span>}
        </button>
      </div>
    </aside>
  );
}
