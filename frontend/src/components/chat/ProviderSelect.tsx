"use client";

import { useState } from "react";
import { Cpu } from "lucide-react";

const PROVIDERS = [
  { value: "auto", label: "Auto" },
  { value: "openrouter", label: "OpenRouter" },
  { value: "groq", label: "Groq" },
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
    <div className="flex items-center gap-2">
      <Cpu size={14} className="text-[var(--apres-ski)]" />
      <select
        value={provider}
        onChange={(e) => handleChange(e.target.value)}
        className="bg-transparent text-[11px] text-[var(--slopes)] border border-[var(--apres-ski)]/20 rounded-[var(--radius-sm)] px-2 py-1 focus:outline-none focus:border-[var(--glacier)]/40"
        aria-label="LLM provider"
      >
        {PROVIDERS.map((p) => (
          <option key={p.value} value={p.value} className="bg-[var(--mountainside)]">
            {p.label}
          </option>
        ))}
      </select>
    </div>
  );
}
