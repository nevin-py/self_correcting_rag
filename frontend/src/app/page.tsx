"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { useAuthStore } from "@/stores/authStore";
import { ArrowRight, Brain, MessageSquare, Shield, Zap } from "lucide-react";

const DEMO_EXCHANGE = [
  { role: "user", text: "What's our company's leave policy?" },
  { role: "assistant", text: "According to the HR handbook, employees are entitled to 20 days of paid leave per year...", delay: 800 },
  { role: "system", text: "Checking 3 sources — one disagrees", delay: 200 },
  { role: "assistant", text: "Let me revise — the updated 2024 policy grants 25 days, not 20. The handbook was revised in Q3.", delay: 600 },
];

function AnimatedDemo() {
  const [visible, setVisible] = useState(0);

  const [resetKey, setResetKey] = useState(0);

  useEffect(() => {
    let i = 0;
    const timer = setInterval(() => {
      i++;
      if (i <= DEMO_EXCHANGE.length) setVisible(i);
      if (i >= DEMO_EXCHANGE.length + 2) {
        clearInterval(timer);
        setTimeout(() => {
          setVisible(0);
          setResetKey((k) => k + 1);
        }, 2000);
      }
    }, 1200);
    return () => clearInterval(timer);
  }, [resetKey]);

  return (
    <div className="w-full max-w-lg mx-auto bg-[var(--mountainside)] rounded-[var(--radius-md)] border border-[var(--apres-ski)]/20 p-6 space-y-4 h-[280px] overflow-hidden">
      <AnimatePresence>
        {DEMO_EXCHANGE.slice(0, visible).map((msg, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            {msg.role === "system" ? (
              <div className="text-xs text-[var(--apres-ski)] italic px-3 py-1">
                {msg.text}
              </div>
            ) : (
              <div
                className={`max-w-[80%] px-4 py-2.5 rounded-[var(--radius-md)] text-sm ${
                  msg.role === "user"
                    ? "bg-[var(--arctic)] text-[var(--midnight)] rounded-br-[var(--radius-sm)]"
                    : "bg-transparent border border-[var(--apres-ski)]/30 text-[var(--slopes)] rounded-bl-[var(--radius-sm)]"
                }`}
              >
                {msg.text}
              </div>
            )}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

export default function HomePage() {
  const router = useRouter();
  const { token, loadUser } = useAuthStore();

  useEffect(() => { loadUser(); }, [loadUser]);
  useEffect(() => { if (token) router.replace("/chat"); }, [token, router]);

  const features = [
    { icon: Brain, title: "Self-Correcting", desc: "Automatically detects and fixes hallucinations" },
    { icon: MessageSquare, title: "Multi-Turn Memory", desc: "Remembers context across messages" },
    { icon: Shield, title: "Verified Answers", desc: "Every claim checked against sources" },
    { icon: Zap, title: "Web Fallback", desc: "Searches Wikipedia and the web live" },
  ];

  return (
    <div className="min-h-screen flex flex-col">
      <header className="flex-1 flex flex-col items-center justify-center px-6 pt-20">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}>
          <h1 className="font-display text-5xl md:text-7xl text-[var(--arctic)] mb-4">
            Ask questions.<br />
            <span className="text-[var(--glacier)] italic">Get verified answers.</span>
          </h1>
          <p className="text-[var(--slopes)] text-lg max-w-xl mb-10">
            Grounded in your documents. Automatically self-corrects when it gets something wrong.
          </p>
          <div className="flex gap-4 justify-center">
            <button onClick={() => router.push("/register")} className="px-8 py-3 rounded-[var(--radius-pill)] bg-[var(--glacier)] text-[var(--midnight)] font-semibold text-sm hover:opacity-90 transition-opacity duration-[var(--duration-micro)] flex items-center gap-2">
              Get Started <ArrowRight size={16} />
            </button>
            <button onClick={() => router.push("/login")} className="px-8 py-3 rounded-[var(--radius-pill)] border border-[var(--apres-ski)] text-[var(--slopes)] font-semibold text-sm hover:border-[var(--arctic)] transition-colors duration-[var(--duration-micro)]">
              Sign In
            </button>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3, duration: 0.8 }} className="mt-16 mb-20">
          <AnimatedDemo />
        </motion.div>
      </header>

      <section className="py-16 px-6 border-t border-[var(--apres-ski)]/10">
        <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((f, i) => (
            <motion.div key={f.title} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 + i * 0.1, duration: 0.5 }} className="p-5 rounded-[var(--radius-md)] bg-[var(--mountainside)] border border-[var(--apres-ski)]/10">
              <f.icon className="w-8 h-8 text-[var(--glacier)] mb-3" />
              <h3 className="font-display text-[var(--arctic)] text-base mb-1">{f.title}</h3>
              <p className="text-[var(--apres-ski)] text-sm">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      <footer className="py-6 text-center text-[var(--apres-ski)] text-xs border-t border-[var(--apres-ski)]/10">
        Built with Next.js, FastAPI, and LangGraph
      </footer>
    </div>
  );
}
