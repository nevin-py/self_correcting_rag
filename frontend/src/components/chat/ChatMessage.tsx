"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Copy, Check, User, Bot, FileText, Shield, AlertTriangle, BookOpen, ExternalLink, ChevronDown } from "lucide-react";
import MarkdownRenderer from "./MarkdownRenderer";
import type { Citation, Claim, Conflict } from "@/lib/api";

interface MessageProps {
  role: "user" | "assistant" | "system";
  content: string;
  isLatest?: boolean;
  meta?: { filename?: string; status?: string };
  citations?: Citation[];
  claims?: Claim[];
  conflicts?: Conflict[];
  finalStatus?: string;
  latencyMs?: number;
}

function StatusBadge({ status, claims }: { status?: string; claims?: Claim[] }) {
  if (!status) return null;
  const failed = claims?.filter((c) => ["unverified", "contradicted", "uncertain"].includes(c.status)) || [];
  const ok = failed.length === 0;
  return (
    <div className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-[var(--radius-sm)] text-[10px] border ${ok ? "bg-green-500/10 border-green-500/30 text-green-400" : "bg-amber-500/10 border-amber-500/30 text-amber-400"}`}>
      {ok ? <Shield size={10} /> : <AlertTriangle size={10} />}
      <span>{ok ? "Verified" : `${failed.length} claim${failed.length > 1 ? "s" : ""} need review`}</span>
    </div>
  );
}

function CollapsibleSection({ title, icon: Icon, children, defaultOpen = false }: {
  title: string;
  icon: React.ElementType;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="mt-3 border border-[var(--apres-ski)]/10 rounded-[var(--radius-md)] overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between gap-2 px-3 py-2 text-[11px] font-medium text-[var(--slopes)] bg-[var(--midnight)]/40 hover:bg-[var(--midnight)] transition-colors"
      >
        <span className="flex items-center gap-1.5"><Icon size={12} className="text-[var(--glacier)]" /> {title}</span>
        <ChevronDown size={12} className={`text-[var(--apres-ski)] transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-3 py-2.5">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ClaimPanel({ claims }: { claims?: Claim[] }) {
  if (!claims || claims.length === 0) return null;
  return (
    <CollapsibleSection title={`Claim verification (${claims.length})`} icon={Shield}>
      <div className="space-y-1.5">
        {claims.map((claim) => {
          const statusClasses: Record<string, string> = {
            verified: "text-green-400 border-green-500/30 bg-green-500/10",
            partial: "text-yellow-400 border-yellow-500/30 bg-yellow-500/10",
            contradicted: "text-red-400 border-red-500/30 bg-red-500/10",
            unverified: "text-[var(--apres-ski)] border-[var(--apres-ski)]/20 bg-[var(--midnight)]",
            uncertain: "text-orange-400 border-orange-500/30 bg-orange-500/10",
          };
          return (
            <div key={claim.claim_id} className={`text-[11px] px-2.5 py-1.5 rounded-[var(--radius-sm)] border ${statusClasses[claim.status] || statusClasses.unverified}`}>
              <p className="leading-relaxed">{claim.text}</p>
              {claim.reasoning && <p className="mt-1 opacity-80 text-[10px]">{claim.reasoning}</p>}
            </div>
          );
        })}
      </div>
    </CollapsibleSection>
  );
}

function CitationPanel({ citations }: { citations?: Citation[] }) {
  if (!citations || citations.length === 0) return null;
  return (
    <CollapsibleSection title={`Sources (${citations.length})`} icon={BookOpen} defaultOpen={true}>
      <div className="grid gap-2">
        {citations.map((c) => (
          <div key={c.evidence_id} className="p-2.5 rounded-[var(--radius-sm)] bg-[var(--midnight)] border border-[var(--apres-ski)]/10 text-xs">
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-center gap-1.5 text-[var(--arctic)] font-medium min-w-0">
                <span className="uppercase text-[9px] px-1 rounded bg-[var(--glacier)]/10 text-[var(--glacier)] shrink-0">{c.source_type}</span>
                <span className="truncate">{c.source_name}</span>
              </div>
              {c.source_url && (
                <a href={c.source_url} target="_blank" rel="noopener noreferrer" className="shrink-0 text-[var(--glacier)] hover:text-[var(--slopes)]">
                  <ExternalLink size={12} />
                </a>
              )}
            </div>
            <p className="mt-1 text-[var(--apres-ski)] line-clamp-3">{c.text}</p>
            <div className="mt-1.5 flex flex-wrap gap-3 text-[10px] text-[var(--apres-ski)]/60">
              {c.source_date && <span>{new Date(c.source_date).toLocaleDateString()}</span>}
              <span>Authority {(c.authority_score * 100).toFixed(0)}%</span>
              <span>Recency {(c.recency_score * 100).toFixed(0)}%</span>
            </div>
          </div>
        ))}
      </div>
    </CollapsibleSection>
  );
}

function ConflictPanel({ conflicts }: { conflicts?: Conflict[] }) {
  if (!conflicts || conflicts.length === 0) return null;
  return (
    <div className="mt-3 p-2.5 rounded-[var(--radius-sm)] border border-amber-500/30 bg-amber-500/10 text-xs">
      <p className="font-medium text-amber-400 flex items-center gap-1.5 mb-1">
        <AlertTriangle size={12} /> Conflicting evidence detected
      </p>
      <p className="text-[var(--apres-ski)]">{conflicts.length} conflict{conflicts.length > 1 ? "s" : ""} resolved using source authority.</p>
    </div>
  );
}

export default function ChatMessage(props: MessageProps) {
  const {
    role,
    content,
    citations,
    claims,
    conflicts,
    finalStatus,
    latencyMs,
  } = props;
  const [copied, setCopied] = useState(false);
  const isUser = role === "user";
  const isSystem = role === "system";

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (isSystem) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
        className="flex justify-center"
      >
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-[var(--radius-pill)] bg-[var(--mountainside)] border border-[var(--apres-ski)]/10 text-xs text-[var(--apres-ski)]">
          <FileText size={12} className="text-[var(--glacier)]" />
          <span>{content}</span>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
      className={`flex gap-3 group ${isUser ? "justify-end" : "justify-start"}`}
    >
      {!isUser && (
        <div className="w-7 h-7 rounded-full bg-[var(--glacier)]/10 flex items-center justify-center shrink-0 mt-1">
          <Bot size={14} className="text-[var(--glacier)]" />
        </div>
      )}

      <div className="max-w-[85%] md:max-w-[75%] relative">
        {isUser ? (
          <div className="bg-[var(--arctic)] text-[var(--midnight)] px-4 py-2.5 rounded-[var(--radius-md)] rounded-br-[var(--radius-sm)] text-sm">
            {content}
          </div>
        ) : (
          <div className="bg-transparent border border-[var(--apres-ski)]/20 px-4 py-3 rounded-[var(--radius-md)] rounded-bl-[var(--radius-sm)] text-sm">
            <div className="flex items-center gap-2 mb-2 flex-wrap">
              <StatusBadge status={finalStatus} claims={claims} />
              {typeof latencyMs === "number" && (
                <span className="text-[10px] text-[var(--apres-ski)]/50">{latencyMs.toFixed(0)}ms</span>
              )}
            </div>
            <MarkdownRenderer content={content} />
            <ConflictPanel conflicts={conflicts} />
            <ClaimPanel claims={claims} />
            <CitationPanel citations={citations} />
          </div>
        )}

        {!isUser && (
          <button
            onClick={handleCopy}
            className="absolute -top-2 -right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded bg-[var(--mountainside)] border border-[var(--apres-ski)]/20 hover:border-[var(--glacier)]/40"
            title="Copy answer"
          >
            {copied ? <Check size={12} className="text-green-400" /> : <Copy size={12} className="text-[var(--apres-ski)]" />}
          </button>
        )}
      </div>

      {isUser && (
        <div className="w-7 h-7 rounded-full bg-[var(--apres-ski)]/20 flex items-center justify-center shrink-0 mt-1">
          <User size={14} className="text-[var(--slopes)]" />
        </div>
      )}
    </motion.div>
  );
}
