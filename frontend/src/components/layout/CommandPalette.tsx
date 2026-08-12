"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { useRouter } from "next/navigation";
import { Search, MessageSquare, Plus, Settings, Clock, Database } from "lucide-react";
import { useChatStore } from "@/stores/chatStore";
import { cn } from "@/lib/utils";
import { Kbd } from "@/components/ui/Kbd";

interface CommandItem {
  id: string;
  label: string;
  icon: React.ElementType;
  href?: string;
  chatId?: string;
  action?: "new-chat";
}

const STATIC_COMMANDS: CommandItem[] = [
  { id: "new", label: "New Session", icon: Plus, action: "new-chat" },
  { id: "chat", label: "Open Workspace", icon: MessageSquare, href: "/chat" },
  { id: "memory", label: "Memory Index", icon: Database, href: "/memory" },
  { id: "history", label: "Session Archive", icon: Clock, href: "/history" },
  { id: "settings", label: "Configuration", icon: Settings, href: "/settings" },
];

export default function CommandPalette() {
  const router = useRouter();
  const { chats, createChat, selectChat } = useChatStore();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(0);

  const items = useMemo(() => {
    const q = query.toLowerCase();
    const staticItems = STATIC_COMMANDS.filter((c) => c.label.toLowerCase().includes(q));
    const chatItems: CommandItem[] = chats
      .filter((c) => c.title.toLowerCase().includes(q))
      .map((c) => ({
        id: c.chat_id,
        label: c.title,
        icon: MessageSquare,
        chatId: c.chat_id,
      }));
    return [...staticItems, ...chatItems];
  }, [query, chats]);

  const execute = useCallback(
    async (index: number) => {
      const item = items[index];
      if (!item) return;

      if (item.action === "new-chat") {
        const chat = await createChat(`Session ${String(chats.length + 1).padStart(3, "0")}`);
        router.push(`/chat/${chat.chat_id}`);
      } else if (item.href) {
        router.push(item.href);
      } else if (item.chatId) {
        selectChat(item.chatId);
        router.push(`/chat/${item.chatId}`);
      }
      setOpen(false);
      setQuery("");
    },
    [items, createChat, chats.length, router, selectChat]
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((v) => !v);
        setQuery("");
        setSelected(0);
      }
      if (!open) return;
      if (e.key === "Escape") setOpen(false);
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelected((s) => Math.min(s + 1, items.length - 1));
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelected((s) => Math.max(s - 1, 0));
      }
      if (e.key === "Enter") {
        e.preventDefault();
        execute(selected);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, items.length, selected, execute]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-void/80 pt-[15vh]">
      <div className="w-full max-w-lg border border-border-strong bg-surface shadow-none">
        <div className="flex items-center gap-2 border-b border-border px-3 py-2">
          <Search size={14} className="text-text-muted" />
          <input
            autoFocus
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelected(0);
            }}
            placeholder="Search sessions, navigate..."
            className="flex-1 bg-transparent font-mono text-sm text-text-primary placeholder:text-text-muted focus:outline-none"
          />
          <Kbd>ESC</Kbd>
        </div>
        <div className="max-h-72 overflow-y-auto">
          {items.length === 0 ? (
            <p className="px-3 py-6 text-center font-mono text-xs text-text-muted">No matches</p>
          ) : (
            items.map((item, i) => (
              <button
                key={item.id}
                onClick={() => execute(i)}
                className={cn(
                  "flex w-full items-center gap-3 border-b border-border px-3 py-2 text-left text-xs transition-colors last:border-b-0",
                  i === selected
                    ? "bg-accent-glow text-text-primary"
                    : "text-text-secondary hover:bg-surface-raised"
                )}
              >
                <item.icon size={13} strokeWidth={1.5} />
                <span className="font-mono uppercase tracking-wider">{item.label}</span>
              </button>
            ))
          )}
        </div>
        <div className="flex items-center gap-4 border-t border-border px-3 py-2">
          <span className="flex items-center gap-1 font-mono text-[10px] text-text-muted">
            <Kbd>↑</Kbd><Kbd>↓</Kbd> navigate
          </span>
          <span className="flex items-center gap-1 font-mono text-[10px] text-text-muted">
            <Kbd>↵</Kbd> select
          </span>
        </div>
      </div>
    </div>
  );
}
