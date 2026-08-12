"use client";

import { useEffect, useRef, useState, useMemo } from "react";
import { motion } from "framer-motion";

interface MemoryNode {
  id: string;
  label: string;
  group: string;
  strength: number;
  x?: number;
  y?: number;
}

const SAMPLE_NODES: MemoryNode[] = [
  { id: "1", label: "Project Goals", group: "core", strength: 20 },
  { id: "2", label: "Team Members", group: "people", strength: 16 },
  { id: "3", label: "Meeting Notes", group: "events", strength: 14 },
  { id: "4", label: "Technical Specs", group: "docs", strength: 18 },
  { id: "5", label: "Deadlines", group: "events", strength: 12 },
  { id: "6", label: "API Design", group: "docs", strength: 15 },
  { id: "7", label: "Budget", group: "core", strength: 13 },
  { id: "8", label: "Alice", group: "people", strength: 11 },
  { id: "9", label: "Bob", group: "people", strength: 11 },
  { id: "10", label: "Q4 Review", group: "events", strength: 14 },
  { id: "11", label: "Database Schema", group: "docs", strength: 16 },
  { id: "12", label: "User Research", group: "core", strength: 13 },
  { id: "13", label: "Wireframes", group: "docs", strength: 12 },
  { id: "14", label: "Sprint 5", group: "events", strength: 10 },
  { id: "15", label: "Auth Flow", group: "docs", strength: 14 },
];

const GROUP_COLORS: Record<string, string> = {
  core: "#8FD6DE",
  people: "#B3B7BA",
  events: "#6C6D74",
  docs: "#D3D1CE",
};

export default function MemoryGraph({ compact = false }: { compact?: boolean }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const size = compact ? 300 : 600;

  const nodes = useMemo(() => {
    const groups = [...new Set(SAMPLE_NODES.map((n) => n.group))];
    return SAMPLE_NODES.map((node) => {
      const groupIdx = groups.indexOf(node.group);
      const nodesInGroup = SAMPLE_NODES.filter((n) => n.group === node.group);
      const idxInGroup = nodesInGroup.indexOf(node);
      const count = nodesInGroup.length;
      const radius = (size * 0.15) + groupIdx * (size * 0.1);
      const angle = (idxInGroup / count) * Math.PI * 2 - Math.PI / 2;
      return {
        ...node,
        x: size / 2 + radius * Math.cos(angle),
        y: size / 2 + radius * Math.sin(angle),
      };
    });
  }, [size]);

  useEffect(() => {
    if (compact) return;
    let frame: number;
    let t = 0;
    const animate = () => {
      t += 0.015;
      const svg = svgRef.current;
      if (!svg) return;
      svg.querySelectorAll("circle[data-node]").forEach((circle, i) => {
        const bx = parseFloat(circle.getAttribute("data-x") || "0");
        const by = parseFloat(circle.getAttribute("data-y") || "0");
        circle.setAttribute("cx", String(bx + Math.sin(t + i * 0.7) * 4));
        circle.setAttribute("cy", String(by + Math.cos(t + i * 0.5) * 4));
      });
      frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [nodes, compact]);

  return (
    <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      className={`relative ${compact ? "w-[300px] h-[300px]" : "w-full aspect-square max-w-[600px] mx-auto"}`}>
      <svg width="0" height="0" className="absolute">
        <defs>
          <filter id="gooey">
            <feGaussianBlur in="SourceGraphic" stdDeviation={compact ? 4 : 8} result="blur" />
            <feColorMatrix in="blur" mode="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 20 -9" result="goo" />
            <feComposite in="SourceGraphic" in2="goo" operator="atop" />
          </filter>
        </defs>
      </svg>
      <svg ref={svgRef} viewBox={`0 0 ${size} ${size}`} className="w-full h-full" style={{ filter: "url(#gooey)" }}>
        {nodes.map((node, i) =>
          nodes.slice(i + 1).map((other) => {
            if (node.group !== other.group || !node.x || !other.x) return null;
            const dist = Math.hypot((node.x || 0) - (other.x || 0), (node.y || 0) - (other.y || 0));
            if (dist > size * 0.35) return null;
            return <line key={`${node.id}-${other.id}`} x1={node.x} y1={node.y} x2={other.x} y2={other.y} stroke={GROUP_COLORS[node.group]} strokeWidth="1" strokeOpacity="0.15" />;
          })
        )}
        {nodes.map((node) => {
          if (node.x == null || node.y == null) return null;
          const color = hovered === node.id ? "#8FD6DE" : GROUP_COLORS[node.group] || "#6C6D74";
          return (
            <g key={node.id}>
              <circle data-node="true" data-x={node.x} data-y={node.y} cx={node.x} cy={node.y} r={node.strength} fill={color} fillOpacity={hovered === node.id ? 1 : 0.7} stroke={hovered === node.id ? "#D3D1CE" : "none"} strokeWidth="2" className="cursor-pointer" onMouseEnter={() => setHovered(node.id)} onMouseLeave={() => setHovered(null)} />
              {!compact && <text x={node.x} y={node.y + node.strength + 14} textAnchor="middle" fill="#6C6D74" fontSize="10" className="pointer-events-none select-none font-body">{node.label}</text>}
            </g>
          );
        })}
      </svg>
      {!compact && (
        <div className="absolute bottom-2 left-2 flex gap-3 text-[10px]">
          {Object.entries(GROUP_COLORS).map(([g, c]) => (
            <div key={g} className="flex items-center gap-1"><div className="w-2 h-2 rounded-full" style={{ backgroundColor: c }} /><span className="text-[var(--apres-ski)] capitalize">{g}</span></div>
          ))}
        </div>
      )}
    </motion.div>
  );
}
