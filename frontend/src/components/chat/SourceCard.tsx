"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, ExternalLink } from "lucide-react";

interface Source {
  title: string;
  content: string;
  url?: string;
}

function parseSources(text: string): Source[] {
  // Parse "Source=0: Title\nContent: ..." format from Tavily
  const sources: Source[] = [];
  const parts = text.split(/Source=\d+:/);
  for (const part of parts) {
    const trimmed = part.trim();
    if (!trimmed) continue;
    const titleMatch = trimmed.match(/^(.+?)\nContent:\s*/);
    if (titleMatch) {
      sources.push({
        title: titleMatch[1].trim(),
        content: trimmed.slice(titleMatch[0].length).trim(),
      });
    }
  }
  return sources;
}

export default function SourceCard({ searchResults }: { searchResults: string }) {
  const [expanded, setExpanded] = useState(false);
  const sources = parseSources(searchResults);

  if (sources.length === 0) return null;

  return (
    <div className="mt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-[var(--apres-ski)] hover:text-[var(--slopes)] transition-colors"
      >
        <ExternalLink size={12} />
        <span>{sources.length} source{sources.length > 1 ? "s" : ""}</span>
        <ChevronDown size={12} className={`transition-transform ${expanded ? "rotate-180" : ""}`} />
      </button>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-2 space-y-2">
              {sources.map((source, i) => (
                <div key={i} className="p-2.5 rounded-[var(--radius-sm)] bg-[var(--midnight)] border border-[var(--apres-ski)]/10 text-xs">
                  <p className="text-[var(--arctic)] font-medium mb-1">{source.title}</p>
                  <p className="text-[var(--apres-ski)] line-clamp-3">{source.content}</p>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
