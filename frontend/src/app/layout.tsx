import type { Metadata } from "next";
import { Space_Grotesk, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-display-loaded",
  weight: ["400", "500", "600", "700"],
});

const ibmPlexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  variable: "--font-mono-loaded",
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "SCRAG — Self-Correcting Knowledge Workspace",
  description:
    "Enterprise retrieval, verification, and correction terminal. Created by Nevin Sunil Oommen.",
  authors: [{ name: "Nevin Sunil Oommen" }],
  creator: "Nevin Sunil Oommen",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${spaceGrotesk.variable} ${ibmPlexMono.variable} font-body antialiased`}
        style={{ ["--font-body-loaded" as string]: "var(--font-display-loaded)" }}
      >
        {children}
      </body>
    </html>
  );
}
