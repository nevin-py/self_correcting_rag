"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/authStore";
import { useChatStore } from "@/stores/chatStore";
import { authApi, settingsApi, type ProviderSettings } from "@/lib/api";
import AppShell from "@/components/layout/AppShell";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/lib/utils";

const ROUTING_PROVIDERS = [
  { id: "auto", label: "AUTO", desc: "Uses whichever user or server keys exist (Groq → Google → OpenRouter)" },
  { id: "groq", label: "GROQ", desc: "Require Groq key (user Settings or server env)" },
  { id: "openrouter", label: "OPENROUTER", desc: "Require OpenRouter key" },
  { id: "google", label: "GOOGLE", desc: "Require Google AI Studio key" },
] as const;

const KEY_PROVIDERS = ["openrouter", "google", "groq"] as const;

const TABS = [
  { id: "profile", label: "Profile" },
  { id: "provider", label: "Provider" },
  { id: "memory", label: "Retention" },
  { id: "system", label: "System" },
];

export default function SettingsPage() {
  const router = useRouter();
  const { token, user, loadUser } = useAuthStore();
  const [activeTab, setActiveTab] = useState("profile");
  const [confirmClear, setConfirmClear] = useState(false);
  const [purgeMsg, setPurgeMsg] = useState("");
  const [purgeErr, setPurgeErr] = useState("");
  const [purging, setPurging] = useState(false);
  const [provider, setProvider] = useState<string>(() => {
    if (typeof window === "undefined") return "auto";
    return localStorage.getItem("llm_provider") || "auto";
  });
  const [providerRows, setProviderRows] = useState<ProviderSettings[]>([]);
  const [editProvider, setEditProvider] = useState<(typeof KEY_PROVIDERS)[number]>("openrouter");
  const [apiKey, setApiKey] = useState("");
  const [fallbackKey, setFallbackKey] = useState("");
  const [plannerModel, setPlannerModel] = useState("");
  const [generatorModel, setGeneratorModel] = useState("");
  const [verifierModel, setVerifierModel] = useState("");
  const [providerMsg, setProviderMsg] = useState("");
  const [providerErr, setProviderErr] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [pwMsg, setPwMsg] = useState("");
  const [pwErr, setPwErr] = useState("");

  useEffect(() => {
    loadUser();
  }, [loadUser]);

  useEffect(() => {
    if (!token) router.replace("/login");
  }, [token, router]);

  useEffect(() => {
    if (!token) return;
    settingsApi
      .listProviders()
      .then((res) => {
        setProviderRows(res.data.providers);
      })
      .catch(() => setProviderRows([]));
  }, [token]);

  useEffect(() => {
    const row = providerRows.find((p) => p.provider === editProvider);
    if (!row) return;
    setPlannerModel(row.planner_model || row.default_planner_model);
    setGeneratorModel(row.generator_model || row.default_generator_model);
    setVerifierModel(row.verifier_model || row.default_verifier_model);
    setApiKey("");
    setFallbackKey("");
  }, [editProvider, providerRows]);

  const handleProviderChange = (id: string) => {
    setProvider(id);
    localStorage.setItem("llm_provider", id);
  };

  const saveProviderKeys = async () => {
    setProviderMsg("");
    setProviderErr("");
    try {
      const body: Record<string, unknown> = {
        planner_model: plannerModel,
        generator_model: generatorModel,
        verifier_model: verifierModel,
      };
      if (apiKey.trim()) body.api_key = apiKey.trim();
      if (fallbackKey.trim()) body.fallback_api_key = fallbackKey.trim();
      const res = await settingsApi.upsertProvider(editProvider, body);
      setProviderRows((prev) =>
        prev.map((p) => (p.provider === editProvider ? res.data : p))
      );
      setApiKey("");
      setFallbackKey("");
      setProviderMsg("Saved (keys stored encrypted; only masked values are shown).");
    } catch (err) {
      const detail =
        err && typeof err === "object" && "response" in err
          ? (err.response as { data?: { detail?: string } })?.data?.detail
          : undefined;
      setProviderErr(detail || "Save failed");
    }
  };

  const clearProvider = async () => {
    setProviderMsg("");
    setProviderErr("");
    try {
      await settingsApi.deleteProvider(editProvider);
      const res = await settingsApi.listProviders();
      setProviderRows(res.data.providers);
      setProviderMsg("Provider keys cleared.");
    } catch (err) {
      const detail =
        err && typeof err === "object" && "response" in err
          ? (err.response as { data?: { detail?: string } })?.data?.detail
          : undefined;
      setProviderErr(detail || "Clear failed");
    }
  };

  const purgeAll = async () => {
    setPurgeMsg("");
    setPurgeErr("");
    setPurging(true);
    try {
      const deleted = await useChatStore.getState().purgeAllChats();
      setPurgeMsg(`Deleted ${deleted} session(s) and indexed memory for this account.`);
      setConfirmClear(false);
    } catch (err) {
      const detail =
        err && typeof err === "object" && "response" in err
          ? (err.response as { data?: { detail?: string } })?.data?.detail
          : undefined;
      setPurgeErr(detail || "Purge failed");
    } finally {
      setPurging(false);
    }
  };

  const changePassword = async () => {
    setPwMsg("");
    setPwErr("");
    try {
      await authApi.changePassword(currentPassword, newPassword);
      setCurrentPassword("");
      setNewPassword("");
      setPwMsg("Password updated.");
    } catch (err) {
      const detail =
        err && typeof err === "object" && "response" in err
          ? (err.response as { data?: { detail?: string } })?.data?.detail
          : undefined;
      setPwErr(detail || "Password change failed");
    }
  };

  const selectedRow = providerRows.find((p) => p.provider === editProvider);

  const header = (
    <header className="flex h-[var(--header-height)] items-center border-b border-border px-4">
      <div>
        <p className="label-caps">Configuration</p>
        <h1 className="font-display text-sm font-semibold text-text-primary">System Settings</h1>
      </div>
    </header>
  );

  return (
    <AppShell header={header} showRightPanel={false}>
      <div className="mx-auto max-w-2xl flex-1 overflow-y-auto px-6 py-6">
        <div className="flex border-b border-border">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                "border-b-2 px-4 py-2 font-mono text-[10px] uppercase tracking-wider transition-colors -mb-px",
                activeTab === tab.id
                  ? "border-accent text-text-primary"
                  : "border-transparent text-text-muted hover:text-text-secondary"
              )}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="mt-6">
          {activeTab === "profile" && (
            <div className="space-y-4">
              <div className="space-y-4 border border-border bg-surface p-4">
                <div>
                  <label className="label-caps mb-1.5 block">Operator Email</label>
                  <Input value={user?.email || ""} disabled />
                </div>
                <div>
                  <label className="label-caps mb-1.5 block">Operator ID</label>
                  <Input value={user?.user_id || ""} disabled className="font-mono text-xs" />
                </div>
              </div>
              <div className="space-y-3 border border-border bg-surface p-4">
                <p className="label-caps">Change Password</p>
                <Input
                  type="password"
                  placeholder="Current password"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                />
                <Input
                  type="password"
                  placeholder="New password (min 8)"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  minLength={8}
                />
                {pwErr && <p className="font-mono text-xs text-error">{pwErr}</p>}
                {pwMsg && <p className="font-mono text-xs text-text-secondary">{pwMsg}</p>}
                <Button variant="accent" size="md" onClick={changePassword}>
                  Update Password
                </Button>
              </div>
            </div>
          )}

          {activeTab === "provider" && (
            <div className="space-y-6">
              <div className="space-y-2">
                <p className="label-caps mb-2">Default routing preference</p>
                {ROUTING_PROVIDERS.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => handleProviderChange(p.id)}
                    className={cn(
                      "w-full border p-4 text-left transition-colors",
                      provider === p.id
                        ? "border-accent bg-accent-glow"
                        : "border-border bg-surface hover:border-border-strong"
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs uppercase tracking-wider text-text-primary">
                        {p.label}
                      </span>
                      {provider === p.id && <Badge variant="accent">Active</Badge>}
                    </div>
                    <p className="mt-1 text-xs text-text-secondary">{p.desc}</p>
                  </button>
                ))}
              </div>

              <div className="border border-border bg-surface p-4 space-y-3">
                <p className="label-caps">Your API keys & models</p>
                <p className="text-xs text-text-muted">
                  OpenRouter, Google, and Groq keys are encrypted at rest. Tavily and Nomic remain
                  system-configured.
                </p>
                <div className="flex gap-2">
                  {KEY_PROVIDERS.map((p) => (
                    <button
                      key={p}
                      onClick={() => setEditProvider(p)}
                      className={cn(
                        "border px-3 py-1 font-mono text-[10px] uppercase",
                        editProvider === p ? "border-accent text-accent" : "border-border text-text-muted"
                      )}
                    >
                      {p}
                    </button>
                  ))}
                </div>
                {selectedRow && (
                  <p className="font-mono text-[10px] text-text-muted">
                    Saved key: {selectedRow.has_key ? selectedRow.masked_key : "none"}
                    {selectedRow.has_server_key ? " · server fallback available" : ""}
                  </p>
                )}
                <Input
                  type="password"
                  placeholder="Primary API key (leave blank to keep)"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                />
                <Input
                  type="password"
                  placeholder="Optional fallback API key"
                  value={fallbackKey}
                  onChange={(e) => setFallbackKey(e.target.value)}
                />
                <Input
                  placeholder="Planner model"
                  value={plannerModel}
                  onChange={(e) => setPlannerModel(e.target.value)}
                />
                <Input
                  placeholder="Generator model"
                  value={generatorModel}
                  onChange={(e) => setGeneratorModel(e.target.value)}
                />
                <Input
                  placeholder="Verifier model"
                  value={verifierModel}
                  onChange={(e) => setVerifierModel(e.target.value)}
                />
                {providerErr && <p className="font-mono text-xs text-error">{providerErr}</p>}
                {providerMsg && <p className="font-mono text-xs text-text-secondary">{providerMsg}</p>}
                <div className="flex gap-2">
                  <Button variant="accent" size="md" onClick={saveProviderKeys}>
                    Save
                  </Button>
                  <Button variant="ghost" size="md" onClick={clearProvider}>
                    Clear keys
                  </Button>
                </div>
              </div>
            </div>
          )}

          {activeTab === "memory" && (
            <div className="border border-border bg-surface p-4">
              <p className="text-sm text-text-secondary">
                Control retention of conversation history and indexed memories.
              </p>
              {!confirmClear ? (
                <Button variant="danger" size="md" onClick={() => setConfirmClear(true)} className="mt-4">
                  Purge All Data
                </Button>
              ) : (
                <div className="mt-4 flex items-center gap-3 border border-error bg-accent-glow p-3">
                  <span className="text-xs text-error">Deletes all chats, messages, and indexed files for this account.</span>
                  <Button variant="accent" size="sm" onClick={purgeAll} disabled={purging}>
                    {purging ? "Purging…" : "Confirm"}
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => setConfirmClear(false)} disabled={purging}>
                    Cancel
                  </Button>
                </div>
              )}
              {purgeErr && <p className="mt-3 font-mono text-xs text-error">{purgeErr}</p>}
              {purgeMsg && <p className="mt-3 font-mono text-xs text-text-secondary">{purgeMsg}</p>}
            </div>
          )}

          {activeTab === "system" && (
            <div className="border border-border bg-surface p-4 space-y-3">
              <div className="flex justify-between border-b border-border pb-2">
                <span className="label-caps">Interface</span>
                <Badge variant="mono">Terminal Dark</Badge>
              </div>
              <div className="flex justify-between border-b border-border pb-2">
                <span className="label-caps">Version</span>
                <span className="font-mono text-xs text-text-muted">SCRAG v1.0</span>
              </div>
              <p className="text-xs text-text-muted">
                Self-Correcting Enterprise Knowledge Workspace. Retrieval, verification, conflict detection,
                and answer correction pipeline.
              </p>
            </div>
          )}
        </div>
      </div>
    </AppShell>
  );
}
