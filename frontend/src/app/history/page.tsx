"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/authStore";
import { useChatStore } from "@/stores/chatStore";
import AppShell from "@/components/layout/AppShell";
import { Input } from "@/components/ui/Input";

export default function HistoryPage() {
  const router = useRouter();
  const { token, loadUser } = useAuthStore();
  const { chats, fetchChats, selectChat } = useChatStore();
  const [filter, setFilter] = useState("");

  useEffect(() => {
    loadUser();
    fetchChats();
  }, [loadUser, fetchChats]);

  useEffect(() => {
    if (!token) router.replace("/login");
  }, [token, router]);

  const filtered = chats.filter((c) => c.title.toLowerCase().includes(filter.toLowerCase()));

  const header = (
    <header className="flex h-[var(--header-height)] items-center gap-4 border-b border-border px-4">
      <div>
        <p className="label-caps">Archive</p>
        <h1 className="font-display text-sm font-semibold text-text-primary">Session History</h1>
      </div>
      <div className="ml-auto w-48">
        <Input
          placeholder="Filter sessions..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="font-mono text-xs"
        />
      </div>
    </header>
  );

  return (
    <AppShell header={header} showRightPanel={false}>
      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <p className="font-mono text-xs text-text-muted">No sessions in archive</p>
          </div>
        ) : (
          <div>
            <div className="grid grid-cols-[1fr_120px_80px] border-b border-border bg-surface-raised px-4 py-2">
              <span className="label-caps">Title</span>
              <span className="label-caps">Date</span>
              <span className="label-caps">ID</span>
            </div>
            {filtered.map((chat) => (
              <button
                key={chat.chat_id}
                onClick={() => {
                  selectChat(chat.chat_id);
                  router.push(`/chat/${chat.chat_id}`);
                }}
                className="grid w-full grid-cols-[1fr_120px_80px] border-b border-border px-4 py-3 text-left transition-colors hover:bg-surface-raised"
              >
                <span className="truncate text-sm text-text-primary">{chat.title}</span>
                <span className="font-mono text-[10px] text-text-muted">
                  {new Date(chat.created_at).toLocaleDateString()}
                </span>
                <span className="font-mono text-[10px] text-text-muted">{chat.chat_id.slice(0, 6)}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </AppShell>
  );
}
