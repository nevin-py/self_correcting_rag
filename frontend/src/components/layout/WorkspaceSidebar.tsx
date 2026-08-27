"use client";

import { useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";
import {
  MessageSquare,
  Clock,
  Settings,
  Plus,
  Trash2,
  LogOut,
  PanelLeftClose,
  PanelLeftOpen,
  Database,
} from "lucide-react";
import { nextSessionTitle, useChatStore } from "@/stores/chatStore";
import { useAuthStore } from "@/stores/authStore";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/Button";

const NAV = [
  { icon: MessageSquare, label: "Workspace", href: "/chat" },
  { icon: Database, label: "Memory", href: "/memory" },
  { icon: Clock, label: "Archive", href: "/history" },
  { icon: Settings, label: "Config", href: "/settings" },
];

export default function WorkspaceSidebar() {
  const router = useRouter();
  const pathname = usePathname();
  const { chats, activeChatId, sidebarCollapsed, fetchChats, createChat, selectChat, deleteChat, toggleSidebar } =
    useChatStore();
  const { logout } = useAuthStore();

  useEffect(() => {
    fetchChats();
  }, [fetchChats]);

    const handleNewChat = async () => {
    const title = nextSessionTitle(chats);
    try {
      const chat = await createChat(title);
      router.push(`/chat/${chat.chat_id}`);
    } catch (err) {
      const detail =
        err && typeof err === "object" && "response" in err
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : undefined;
      window.alert(detail || "Could not create a session.");
    }
  };

  const width = sidebarCollapsed ? "var(--sidebar-collapsed)" : "var(--sidebar-width)";

  return (
    <aside
      className="fixed left-0 top-0 z-40 flex h-screen flex-col border-r border-border bg-surface"
      style={{ width }}
    >
      {/* Header */}
      <div className="flex h-[var(--header-height)] items-center justify-between border-b border-border px-3">
        {!sidebarCollapsed && (
          <div>
            <p className="font-display text-base font-bold tracking-[0.18em] text-text-primary">SCRAG</p>
            <p className="font-mono text-[9px] uppercase tracking-[0.2em] text-text-muted">Knowledge Terminal</p>
          </div>
        )}
        <button
          onClick={toggleSidebar}
          className="border border-transparent p-1.5 text-text-muted hover:border-border hover:text-text-primary"
          aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {sidebarCollapsed ? <PanelLeftOpen size={14} /> : <PanelLeftClose size={14} />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="border-b border-border p-2">
        {NAV.map((item) => {
          const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
          return (
            <a
              key={item.href}
              href={item.href}
              className={cn(
                "mb-px flex items-center gap-3 rounded-md px-3 py-2 text-xs transition-colors",
                active
                  ? "sidebar-item-active border border-transparent bg-surface-raised text-text-primary"
                  : "border border-transparent text-text-secondary hover:border-border hover:bg-surface-raised hover:text-text-primary"
              )}
            >
              <item.icon size={14} strokeWidth={1.5} />
              {!sidebarCollapsed && <span className="font-mono uppercase tracking-wider">{item.label}</span>}
            </a>
          );
        })}

        <button
          onClick={handleNewChat}
          className="mt-2 flex w-full items-center gap-3 border border-dashed border-border px-3 py-2 text-xs text-text-secondary hover:border-accent hover:text-accent-bright"
        >
          <Plus size={14} strokeWidth={1.5} />
          {!sidebarCollapsed && <span className="font-mono uppercase tracking-wider">New Session</span>}
        </button>
      </nav>

      {/* Session list */}
      {!sidebarCollapsed && (
        <div className="flex-1 overflow-y-auto p-2">
          <p className="label-caps mb-2 px-1">Active Sessions</p>
          {chats.length === 0 ? (
            <p className="px-2 py-4 font-mono text-[10px] text-text-muted">No sessions indexed</p>
          ) : (
            <div className="space-y-px">
              {chats.slice(0, 12).map((chat) => (
                <div
                  key={chat.chat_id}
                  onClick={() => {
                    selectChat(chat.chat_id);
                    router.push(`/chat/${chat.chat_id}`);
                  }}
                  className={cn(
                    "group flex cursor-pointer items-center gap-2 border px-2 py-2 transition-colors",
                    activeChatId === chat.chat_id
                      ? "border-border-strong bg-surface-raised text-text-primary"
                      : "border-transparent text-text-secondary hover:border-border hover:bg-surface-raised"
                  )}
                >
                  <span className="font-mono text-[9px] text-text-muted">
                    {chat.chat_id.slice(0, 4).toUpperCase()}
                  </span>
                  <span className="flex-1 truncate text-xs">{chat.title}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteChat(chat.chat_id);
                    }}
                    className="opacity-0 transition-opacity group-hover:opacity-100"
                    aria-label="Delete session"
                  >
                    <Trash2 size={11} className="text-error hover:text-accent-bright" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Footer */}
      <div className="border-t border-border p-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            logout();
            router.push("/");
          }}
          className="w-full justify-start"
        >
          <LogOut size={12} />
          {!sidebarCollapsed && "Sign Out"}
        </Button>
      </div>
    </aside>
  );
}

// Re-export as Sidebar for backward compatibility
export { WorkspaceSidebar as Sidebar };
