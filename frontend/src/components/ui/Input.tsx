"use client";

import { cn } from "@/lib/utils";
import { InputHTMLAttributes, forwardRef } from "react";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "w-full border border-border bg-surface-inset px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:border-accent-bright focus:outline-none transition-colors",
        className
      )}
      {...props}
    />
  )
);
Input.displayName = "Input";
