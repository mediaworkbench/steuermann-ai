"use client";

import { useState, createContext, useContext, useEffect } from "react";
import { usePathname } from "next/navigation";
import { Toaster } from "@/components/ui/sonner";
import { AppSidebar } from "./AppSidebar";
import { Header } from "./Header";
import { AppShell } from "@/components/product/AppShell";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { useProfile } from "@/hooks/useProfile";
import { useI18n } from "@/hooks/useI18n";
import { useConversations } from "@/hooks/useConversations";
import { ChatSessionProvider } from "@/context/ChatSessionContext";
import { WorkspacePanelProvider } from "@/context/WorkspacePanelContext";
import { ProviderHealthProvider } from "@/context/ProviderHealthContext";
import { ProviderOfflineBanner } from "@/components/product/ProviderOfflineBanner";
import type { Conversation } from "@/lib/types";

const WORKSPACE_OPEN_KEY = "workspace.panelOpen";

// ── Context so any child can access conversation state ───────────────

interface ConversationContextValue {
  conversations: Conversation[];
  activeId: string | null;
  activeConversation: Conversation | null;
  revision: number;
  setActiveId: (id: string | null, conv?: Conversation | null) => void;
  create: (title?: string, language?: string) => Promise<Conversation | null>;
  update: (id: string, updates: { title?: string; pinned?: boolean; language?: string }) => Promise<Conversation | null>;
  remove: (id: string) => Promise<boolean>;
  rename: (id: string, title: string) => Promise<Conversation | null>;
  bulkDelete: (ids: string[]) => Promise<void>;
  bulkPin: (ids: string[], pinned: boolean) => Promise<void>;
  refresh: () => Promise<void>;
  loading: boolean;
  workspaceSidebarOpen: boolean;
  setWorkspaceSidebarOpen: React.Dispatch<React.SetStateAction<boolean>>;
}

const ConversationContext = createContext<ConversationContextValue | null>(null);

export function useConversationContext() {
  const ctx = useContext(ConversationContext);
  if (!ctx) throw new Error("useConversationContext must be used within LayoutShell");
  return ctx;
}

export function LayoutShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  // Standalone full-screen gate pages — rendered without the app shell (no sidebar/chat)
  // so the user cannot navigate into the app from them.
  if (pathname.startsWith("/login") || pathname.startsWith("/change-password")) {
    return (
      <>
        {children}
        <Toaster
          position="top-right"
          closeButton
          toastOptions={{ duration: 6000 }}
        />
      </>
    );
  }

  return <AuthenticatedLayoutShell>{children}</AuthenticatedLayoutShell>;
}

function AuthenticatedLayoutShell({ children }: { children: React.ReactNode }) {
  const [workspaceSidebarOpen, setWorkspaceSidebarOpen] = useState(false);
  const pathname = usePathname();
  const profile = useProfile();
  const { t } = useI18n();
  const convState = useConversations();

  const chatTitle = convState.activeConversation?.title ?? profile.appName ?? t("chat.aiAgent");
  const isChat = pathname === "/";

  useEffect(() => {
    if (!profile.loading && profile.appName) {
      document.title = profile.appName;
    }
  }, [profile.loading, profile.appName]);

  // Restore + persist the workspace panel open state (a non-risky UI pref).
  // Hydrate in an effect so the first client render still matches SSR. The
  // restore effect runs before the persist effect, so the stored value is read
  // before it can be overwritten.
  useEffect(() => {
    try {
      if (localStorage.getItem(WORKSPACE_OPEN_KEY) === "true") setWorkspaceSidebarOpen(true);
    } catch {
      /* localStorage unavailable */
    }
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem(WORKSPACE_OPEN_KEY, String(workspaceSidebarOpen));
    } catch {
      /* ignore persistence failures */
    }
  }, [workspaceSidebarOpen]);

  return (
    <ProviderHealthProvider>
    <SidebarProvider>
    <ConversationContext.Provider value={{ ...convState, workspaceSidebarOpen, setWorkspaceSidebarOpen }}>
      <AppSidebar />
      {/* min-w-0: SidebarInset is a flex child of the sidebar wrapper and defaults to
          min-width:auto. Without this, wide non-wrapping content (long code tokens,
          tables, URLs in a model answer) forces the inset past its share, squeezing the
          sidebar/workspace and overflowing the page horizontally. */}
      <SidebarInset className="min-w-0">
        <AppShell>
          {/* Global provider-offline strip — shrink-0 so it stacks above the Header
              without eating the content area's height. */}
          <ProviderOfflineBanner />
          <Header
            chatTitle={chatTitle}
            activeConversation={convState.activeConversation}
          />
          <div className={`flex-1 min-h-0 ${isChat ? "overflow-hidden" : "overflow-y-auto"}`}>
            <WorkspacePanelProvider>
              <ChatSessionProvider>{children}</ChatSessionProvider>
            </WorkspacePanelProvider>
          </div>
        </AppShell>
      </SidebarInset>
    </ConversationContext.Provider>
    </SidebarProvider>
      <Toaster
        position="top-right"
        closeButton
        toastOptions={{ duration: 6000 }}
      />
    </ProviderHealthProvider>
  );
}
