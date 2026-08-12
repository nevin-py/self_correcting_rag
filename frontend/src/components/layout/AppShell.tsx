"use client";

import { ReactNode } from "react";
import WorkspaceSidebar from "./WorkspaceSidebar";
import CommandPalette from "./CommandPalette";
import { useChatStore } from "@/stores/chatStore";
import { cn } from "@/lib/utils";

interface AppShellProps {
  children: ReactNode;
  header?: ReactNode;
  rightPanel?: ReactNode;
  showRightPanel?: boolean;
}

export default function AppShell({ children, header, rightPanel, showRightPanel = true }: AppShellProps) {
  const { sidebarCollapsed, rightPanelOpen } = useChatStore();
  const sidebarWidth = sidebarCollapsed ? "var(--sidebar-collapsed)" : "var(--sidebar-width)";

  return (
    <div className="flex h-screen overflow-hidden bg-void">
      <WorkspaceSidebar />
      <CommandPalette />

      <div
        className="flex flex-1 flex-col min-w-0 transition-[margin] duration-[var(--duration-normal)]"
        style={{ marginLeft: sidebarWidth }}
      >
        {header}

        <div className="flex flex-1 min-h-0">
          <main className={cn("flex min-w-0 flex-1 flex-col overflow-hidden", showRightPanel && rightPanelOpen && "border-r border-border")}>
            {children}
          </main>

          {showRightPanel && rightPanelOpen && rightPanel && (
            <>
              <aside className="hidden w-[var(--panel-width)] shrink-0 overflow-hidden bg-surface lg:flex lg:flex-col">
                {rightPanel}
              </aside>
              <div className="fixed inset-0 z-50 flex lg:hidden">
                <div className="flex-1 bg-void/70" onClick={() => useChatStore.getState().toggleRightPanel()} />
                <aside className="w-[min(100vw,var(--panel-width))] shrink-0 overflow-hidden bg-surface border-l border-border">
                  {rightPanel}
                </aside>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
