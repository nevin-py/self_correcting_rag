"use client";

import { cn } from "@/lib/utils";
import { ButtonHTMLAttributes, forwardRef } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger" | "accent";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: "sm" | "md" | "lg";
}

const variants: Record<Variant, string> = {
  primary:
    "bg-text-primary text-void border-border hover:bg-text-secondary",
  secondary:
    "bg-surface-raised text-text-primary border-border hover:border-border-strong hover:bg-surface",
  ghost:
    "bg-transparent text-text-secondary border-transparent hover:text-text-primary hover:bg-surface-raised hover:border-border",
  danger:
    "bg-transparent text-error border-error/40 hover:bg-accent-glow hover:border-error",
  accent:
    "bg-accent text-text-primary border-accent hover:bg-accent-bright hover:border-accent-bright",
};

const sizes: Record<"sm" | "md" | "lg", string> = {
  sm: "px-2 py-1 text-[11px]",
  md: "px-3 py-2 text-xs",
  lg: "px-4 py-3 text-sm",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "secondary", size = "md", ...props }, ref) => (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center gap-2 border font-mono uppercase tracking-wider transition-colors duration-[var(--duration-fast)] disabled:opacity-40 disabled:pointer-events-none",
        variants[variant],
        sizes[size],
        className
      )}
      {...props}
    />
  )
);
Button.displayName = "Button";
