"use client";

import { cn } from "@/lib/utils";

export function Kbd({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <kbd
      className={cn(
        "inline-flex min-w-[1.25rem] items-center justify-center border border-border-strong bg-surface-inset px-1 py-0.5 font-mono text-[10px] text-text-muted",
        className
      )}
    >
      {children}
    </kbd>
  );
}
