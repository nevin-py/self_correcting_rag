"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/authStore";
import { useChatStore } from "@/stores/chatStore";
import AppShell from "@/components/layout/AppShell";
import { Button } from "@/components/ui/Button";

export default function ChatPage() {
  const router = useRouter();
  const { token, loadUser } = useAuthStore();
  const { fetchChats } = useChatStore();

  useEffect(() => {
    loadUser();
    fetchChats();
  }, [loadUser, fetchChats]);

  useEffect(() => {
    if (!token) router.replace("/login");
  }, [token, router]);

  const handleNewChat = async () => {
    const chat = await useChatStore.getState().createChat("Session 001");
    router.push(`/chat/${chat.chat_id}`);
  };

  return (
    <AppShell showRightPanel={false}>
      <div className="flex h-full flex-col items-center justify-center grid-bg">
        <div className="border border-border bg-surface p-8 text-center">
          <p className="label-caps mb-2">Workspace</p>
          <h2 className="font-display text-xl font-semibold text-text-primary">No Active Session</h2>
          <p className="mt-2 text-sm text-text-secondary">Initialize a new knowledge session to begin.</p>
          <Button variant="accent" onClick={handleNewChat} className="mt-6">
            New Session
          </Button>
        </div>
      </div>
    </AppShell>
  );
}
