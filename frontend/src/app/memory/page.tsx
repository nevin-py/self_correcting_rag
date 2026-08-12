"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/authStore";
import { memoryApi, type MemoryChunk } from "@/lib/api";
import AppShell from "@/components/layout/AppShell";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";

export default function MemoryPage() {
  const router = useRouter();
  const { token, loadUser } = useAuthStore();
  const [q, setQ] = useState("");
  const [query, setQuery] = useState("");
  const [chunks, setChunks] = useState<MemoryChunk[]>([]);
  const [total, setTotal] = useState(0);
  const [collection, setCollection] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadUser();
  }, [loadUser]);

  useEffect(() => {
    if (!token) router.replace("/login");
  }, [token, router]);

  useEffect(() => {
    if (!token) return;
    const t = setTimeout(() => setQuery(q.trim()), 250);
    return () => clearTimeout(t);
  }, [q, token]);

  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    setLoading(true);
    setError("");
    memoryApi
      .getChunks({ limit: 100, offset: 0, q: query || undefined })
      .then((res) => {
        if (cancelled) return;
        setChunks(res.data.chunks);
        setTotal(res.data.total);
        setCollection(res.data.collection);
      })
      .catch((err) => {
        if (cancelled) return;
        const detail =
          err && typeof err === "object" && "response" in err
            ? (err.response as { data?: { detail?: string } })?.data?.detail
            : undefined;
        setError(detail || "Could not load memory chunks");
        setChunks([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [token, query]);

  const groups = useMemo(() => {
    const map = new Map<string, MemoryChunk[]>();
    for (const chunk of chunks) {
      const key = `${chunk.filename || "untitled"}::${chunk.chat_id || "no-chat"}`;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(chunk);
    }
    return [...map.entries()];
  }, [chunks]);

  const header = (
    <header className="flex h-[var(--header-height)] items-center gap-4 border-b border-border px-4">
      <div>
        <p className="label-caps">Memory Index</p>
        <h1 className="font-display text-sm font-semibold text-text-primary">Chunk Grid</h1>
      </div>
      <div className="ml-auto w-56">
        <Input
          placeholder="Search chunks..."
          value={q}
          onChange={(e) => setQ(e.target.value)}
          className="font-mono text-xs"
        />
      </div>
    </header>
  );

  return (
    <AppShell header={header} showRightPanel={false}>
      <div className="flex flex-1 flex-col overflow-auto px-6 py-6">
        <div className="mb-4 flex items-center gap-3 text-xs text-text-muted">
          <Badge variant="mono">{collection || "—"}</Badge>
          <span className="font-mono">{loading ? "Loading…" : `${chunks.length} shown · ${total} total`}</span>
        </div>
        {error && (
          <p className="mb-4 border border-error bg-accent-glow px-3 py-2 font-mono text-xs text-error">
            {error}
          </p>
        )}
        {!loading && !error && chunks.length === 0 && (
          <p className="font-mono text-xs text-text-muted">
            No chunks yet. Upload documents in a chat to populate your collection.
          </p>
        )}
        <div className="space-y-8">
          {groups.map(([key, items]) => {
            const [filename, chatId] = key.split("::");
            return (
              <section key={key}>
                <div className="mb-3 flex flex-wrap items-baseline gap-2 border-b border-border pb-2">
                  <h2 className="font-display text-sm text-text-primary">{filename}</h2>
                  <span className="font-mono text-[10px] text-text-muted">chat {chatId}</span>
                  <span className="font-mono text-[10px] text-text-muted">{items.length} chunks</span>
                </div>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {items.map((chunk) => (
                    <article
                      key={chunk.id}
                      className="border border-border bg-surface p-3 transition-colors hover:border-border-strong"
                    >
                      <div className="mb-2 flex gap-2 font-mono text-[10px] text-text-muted">
                        <span>#{chunk.chunk_index ?? "—"}</span>
                        {chunk.chunk_type && <span>{chunk.chunk_type}</span>}
                      </div>
                      <p className="text-xs leading-relaxed text-text-secondary whitespace-pre-wrap">
                        {chunk.document_preview}
                      </p>
                    </article>
                  ))}
                </div>
              </section>
            );
          })}
        </div>
      </div>
    </AppShell>
  );
}
