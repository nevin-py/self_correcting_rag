"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation } from "@/lib/api";
import { resolveSourceUrl } from "@/lib/api";

function cleanLLMOutput(text: string): string {
  return text
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/?p>/gi, "\n")
    .replace(/<\/?strong>/gi, "**")
    .replace(/<\/?em>/gi, "*")
    .replace(/<\/?(?:div|span|table|tr|td|th|thead|tbody)[^>]*>/gi, "")
    .replace(/【\d+†[^】]*】/g, "")
    .replace(/\[\d+†[^\]]*\]/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/**
 * Rewrite bare [E1]-style citation markers into markdown links pointing at
 * the cited source URL, so readers can jump straight to the document/web
 * page. Markers without a URL are left as plain text (they still resolve in
 * the Evidence panel). We skip markers already followed by "(" (i.e. inside a
 * markdown link) and code spans are rare enough not to warrant a parser.
 */
function linkCitationMarkers(text: string, byKey: Map<string, Citation>, positional: Citation[]): string {
  if (byKey.size === 0 && positional.length === 0) return text;
  return text.replace(/\[([Ee]\d{1,3})\](?!\()/g, (match, key: string) => {
    const n = parseInt(key.slice(1), 10);
    const citation = byKey.get(key.toUpperCase()) ?? positional[n - 1];
    if (!citation?.source_url) return match;
    const url = resolveSourceUrl(citation.source_url).replace(/[)\s]+$/, "");
    return `[${key}](${url})`;
  });
}

const PROSE =
  "prose prose-invert max-w-none " +
  // Readability: body copy 15px/1.7, sans — the gothic display font never
  // touches message content, including markdown headings inside an answer.
  "prose-p:text-[15px] prose-p:leading-[1.7] prose-p:text-text-primary " +
  "prose-headings:font-body prose-headings:text-text-primary prose-headings:font-semibold " +
  "prose-h1:text-[19px] prose-h2:text-[17px] prose-h3:text-[16px] " +
  "prose-strong:text-text-primary prose-strong:font-semibold " +
  "prose-a:text-accent-bright prose-a:no-underline hover:prose-a:underline " +
  "prose-li:text-[15px] prose-li:leading-[1.65] prose-ul:my-2 prose-ol:my-2 " +
  "prose-blockquote:border-l-2 prose-blockquote:border-accent prose-blockquote:text-text-secondary ";

export default function MarkdownRenderer({
  content,
  citations,
}: {
  content: string;
  citations?: Citation[];
}) {
  // Map [E#] markers → citations. Keys are assigned per-turn ("E1", "E2", …)
  // by the backend and exposed on each citation as `cite_key`.
  const citeByKey = new Map<string, Citation>();
  (citations || []).forEach((c) => {
    if (c.cite_key) citeByKey.set(c.cite_key.toUpperCase(), c);
  });

  const cleaned = linkCitationMarkers(cleanLLMOutput(content), citeByKey, citations || []);

  return (
    <div className={PROSE}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => (
            <div className="my-3 overflow-x-auto border border-border">
              <table className="w-full text-sm">{children}</table>
            </div>
          ),
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              title={citations?.some((c) => c.source_url === href) ? "Open cited source ↗" : undefined}
            >
              {children}
            </a>
          ),
          code: ({ className, children, ...props }) => {
            const isInline = !className;
            if (isInline) {
              return (
                <code className="font-mono text-xs" {...props}>
                  {children}
                </code>
              );
            }
            return (
              <div className="my-3 overflow-hidden rounded-md border border-border bg-surface-inset">
                <div className="h-px w-full border-t border-accent/70" aria-hidden="true" />
                <pre className="overflow-x-auto p-3">
                  <code className="font-mono text-[13.5px] leading-relaxed text-text-secondary" {...props}>
                    {children}
                  </code>
                </pre>
              </div>
            );
          },
          p: ({ children }) => <p className="leading-relaxed">{children}</p>,
        }}
      >
        {cleaned}
      </ReactMarkdown>
    </div>
  );
}
