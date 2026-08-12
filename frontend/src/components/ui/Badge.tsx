"use client";

import { cn } from "@/lib/utils";

type BadgeVariant = "default" | "accent" | "success" | "warning" | "error" | "mono";

const variants: Record<BadgeVariant, string> = {
  default: "border-border text-text-secondary bg-surface-raised",
  accent: "border-accent text-accent-bright bg-accent-glow",
  success: "border-success text-success bg-success/10",
  warning: "border-warning text-warning bg-warning/10",
  error: "border-error text-error bg-accent-glow",
  mono: "border-border-strong text-text-muted bg-void font-mono",
};

export function Badge({
  children,
  variant = "default",
  className,
}: {
  children: React.ReactNode;
  variant?: BadgeVariant;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 border px-1.5 py-0.5 text-[10px] font-mono uppercase tracking-wider",
        variants[variant],
        className
      )}
    >
      {children}
    </span>
  );
}
