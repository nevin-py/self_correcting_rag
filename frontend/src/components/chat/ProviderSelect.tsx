"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";

const PROVIDERS = [
  { value: "auto", label: "AUTO" },
  { value: "groq", label: "GRQ" },
  { value: "openrouter", label: "OR" },
  { value: "google", label: "GGL" },
];

function getSavedProvider(): string {
  if (typeof window === "undefined") return "auto";
  return localStorage.getItem("llm_provider") || "auto";
}

export default function ProviderSelect() {
  const [provider, setProvider] = useState(getSavedProvider);

  const handleChange = (value: string) => {
    setProvider(value);
    if (typeof window !== "undefined") {
      localStorage.setItem("llm_provider", value);
    }
  };

  return (
    <select
      value={provider}
      onChange={(e) => handleChange(e.target.value)}
      className={cn(
        "border border-border bg-surface px-2 py-1.5 font-mono text-[10px] uppercase tracking-wider",
        "text-text-secondary hover:border-border-strong focus:border-accent-bright focus:outline-none"
      )}
      aria-label="LLM provider"
    >
      {PROVIDERS.map((p) => (
        <option key={p.value} value={p.value} className="bg-surface">
          {p.label}
        </option>
      ))}
    </select>
  );
}
