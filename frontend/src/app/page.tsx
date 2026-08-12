"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/authStore";
import { Button } from "@/components/ui/Button";

const DEMO_STAGES = [
  { label: "RETRIEVE", text: "Query indexed HR handbook..." },
  { label: "VERIFY", text: "20 days paid leave — claim extracted" },
  { label: "CONFLICT", text: "Policy v2023 contradicts v2024 revision" },
  { label: "SEARCH", text: "Web fallback: official policy portal" },
  { label: "CORRECT", text: "Answer revised to 25 days" },
  { label: "ANSWER", text: "Verified response delivered" },
];

function PipelineDemo() {
  const [step, setStep] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setStep((s) => (s >= DEMO_STAGES.length ? 0 : s + 1));
    }, 1400);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="border border-border bg-surface">
      <div className="border-b border-border px-4 py-2">
        <p className="label-caps">Live Correction Sequence</p>
      </div>
      <div className="p-4 space-y-1">
        {DEMO_STAGES.slice(0, step).map((stage, i) => (
          <div
            key={stage.label}
            className={`flex items-start gap-3 border-l-2 px-3 py-2 ${
              i === step - 1 ? "border-accent-bright bg-accent-glow" : "border-border bg-surface-inset"
            }`}
          >
            <span className="font-mono text-[9px] uppercase tracking-wider text-accent-bright w-16 shrink-0">
              {stage.label}
            </span>
            <span className="font-mono text-[10px] text-text-secondary">{stage.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function HomePage() {
  const router = useRouter();
  const { token, loadUser } = useAuthStore();

  useEffect(() => {
    loadUser();
  }, [loadUser]);

  useEffect(() => {
    if (token) router.replace("/chat");
  }, [token, router]);

  return (
    <div className="min-h-screen bg-void">
      {/* Top bar */}
      <header className="flex items-center justify-between border-b border-border px-6 py-3">
        <div>
          <p className="font-display text-sm font-semibold tracking-tight text-text-primary">SCRAG</p>
          <p className="font-mono text-[9px] uppercase tracking-[0.2em] text-text-muted">
            Self-Correcting Knowledge Workspace
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={() => router.push("/login")}>
            Sign In
          </Button>
          <Button variant="accent" size="sm" onClick={() => router.push("/register")}>
            Initialize Access
          </Button>
        </div>
      </header>

      {/* Hero */}
      <main className="mx-auto max-w-6xl px-6 py-16">
        <div className="grid gap-12 lg:grid-cols-2 lg:items-start">
          <div>
            <p className="label-caps mb-4">Enterprise Knowledge Terminal</p>
            <h1 className="font-display text-4xl font-semibold leading-tight tracking-tight text-text-primary md:text-5xl">
              Retrieve.
              <br />
              Verify.
              <br />
              <span className="text-accent-bright">Correct.</span>
            </h1>
            <p className="mt-6 max-w-md text-sm leading-relaxed text-text-secondary">
              Not a chatbot. An information-analysis instrument that retrieves from your knowledge base,
              verifies every claim, detects contradictions, searches when local knowledge fails, and
              delivers corrected answers with full evidence provenance.
            </p>

            <div className="mt-8 border border-border bg-surface-inset p-4">
              <p className="font-mono text-[9px] uppercase tracking-[0.15em] text-text-muted">
                Operational Pipeline
              </p>
              <p className="mt-2 font-mono text-[10px] leading-relaxed text-accent-bright">
                RETRIEVE → VERIFY → DETECT CONFLICT → SEARCH → CORRECT → ANSWER
              </p>
            </div>

            <Button variant="accent" size="lg" onClick={() => router.push("/register")} className="mt-8">
              Enter Workspace
            </Button>
          </div>

          <PipelineDemo />
        </div>

        {/* Feature grid */}
        <div className="mt-20 grid gap-px border border-border bg-border sm:grid-cols-2 lg:grid-cols-4">
          {[
            { title: "Evidence Retrieval", desc: "Multi-source document indexing with authority scoring" },
            { title: "Claim Verification", desc: "Every assertion checked against indexed evidence" },
            { title: "Conflict Detection", desc: "Contradictions surfaced and resolved transparently" },
            { title: "Web Fallback", desc: "Automatic external search when local knowledge is stale" },
          ].map((f) => (
            <div key={f.title} className="bg-surface p-5">
              <p className="font-mono text-[10px] uppercase tracking-wider text-accent-bright">{f.title}</p>
              <p className="mt-2 text-xs leading-relaxed text-text-secondary">{f.desc}</p>
            </div>
          ))}
        </div>
      </main>

      <footer className="border-t border-border px-6 py-4">
        <p className="font-mono text-[9px] text-text-muted">
          SCRAG · Self-Correcting RAG · FastAPI · LangGraph · Next.js
        </p>
      </footer>
    </div>
  );
}
