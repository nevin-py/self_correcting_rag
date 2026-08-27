import type { Metadata } from "next";
import { Cinzel, Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

// Gothic display voice — reserved for branding, headers, empty states ONLY.
const cinzel = Cinzel({
  subsets: ["latin"],
  variable: "--font-display-loaded",
  weight: ["400", "600", "700"],
});

// Body — clean, highly legible sans for ALL message content.
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-body-loaded",
  weight: ["400", "500", "600", "700"],
});

// Code / retrieved snippets.
const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono-loaded",
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "SCRAG — Self-Correcting Knowledge Workspace",
  description:
    "Verify, retrieve, correct. Agentic RAG with a hallucination gate. Created by Nevin Sunil Oommen.",
  authors: [{ name: "Nevin Sunil Oommen" }],
  creator: "Nevin Sunil Oommen",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${cinzel.variable} ${inter.variable} ${jetbrainsMono.variable} font-body antialiased`}
        style={{ ["--font-body-loaded" as string]: "var(--font-body-loaded)" }}
      >
        {children}
      </body>
    </html>
  );
}
