"use client";

import { useState, useCallback, createContext, useContext, useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Toaster } from "sonner";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";
import { useProfile } from "@/hooks/useProfile";
import { useI18n } from "@/hooks/useI18n";
import { useConversations } from "@/hooks/useConversations";
import type { Conversation } from "@/lib/types";

// ── Context so any child can access conversation state ───────────────

interface ConversationContextValue {
  conversations: Conversation[];
  activeId: string | null;
  activeConversation: Conversation | null;
  showArchived: boolean;
  setActiveId: (id: string | null) => void;
  create: (title?: string, language?: string) => Promise<Conversation | null>;
  update: (id: string, updates: { title?: string; archived?: boolean; pinned?: boolean; language?: string }) => Promise<Conversation | null>;
  remove: (id: string) => Promise<boolean>;
  rename: (id: string, title: string) => Promise<Conversation | null>;
  archive: (id: string, archived: boolean) => Promise<void>;
  bulkDelete: (ids: string[]) => Promise<void>;
  bulkArchive: (ids: string[]) => Promise<void>;
  doExport: (id: string, format: "json" | "markdown") => Promise<void>;
  toggleArchived: () => void;
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

  if (pathname.startsWith("/login")) {
    return (
      <>
        {children}
        <Toaster
          position="top-right"
          richColors
          theme="light"
          toastOptions={{
            style: {
              fontFamily: '"Open Sans", sans-serif',
              borderRadius: "0.75rem",
            },
          }}
        />
      </>
    );
  }

  return <AuthenticatedLayoutShell>{children}</AuthenticatedLayoutShell>;
}

function AuthenticatedLayoutShell({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [workspaceSidebarOpen, setWorkspaceSidebarOpen] = useState(false);
  const pathname = usePathname();
  const router = useRouter();
  const profile = useProfile();
  const { t } = useI18n();
  const convState = useConversations();

  const openSidebar = useCallback(() => setSidebarOpen(true), []);
  const closeSidebar = useCallback(() => setSidebarOpen(false), []);

  const chatTitle = convState.activeConversation?.title ?? profile.appName ?? t("chat.aiAgent");
  const isChat = pathname === "/";

  useEffect(() => {
    if (!profile.loading && profile.appName) {
      document.title = profile.appName;
    }
  }, [profile.loading, profile.appName]);

  return (
    <>
    <ConversationContext.Provider value={{ ...convState, workspaceSidebarOpen, setWorkspaceSidebarOpen }}>
      <Sidebar
        isOpen={sidebarOpen}
        onClose={closeSidebar}
        conversations={convState.conversations}
        activeId={convState.activeId}
        onSelect={(id) => { convState.setActiveId(id); closeSidebar(); if (pathname !== "/") router.push("/"); }}
        onNewChat={async () => { convState.setActiveId(null); closeSidebar(); if (pathname !== "/") router.push("/"); }}
        onDelete={convState.remove}
        onPin={async (id, pinned) => { await convState.update(id, { pinned }); }}
        onRename={convState.rename}
        onArchive={async (id, archived) => { await convState.archive(id, archived); }}
        onExport={convState.doExport}
        onBulkDelete={convState.bulkDelete}
        onBulkArchive={convState.bulkArchive}
        showArchived={convState.showArchived}
        onToggleArchived={convState.toggleArchived}
      />
      <main className="flex-1 flex flex-col h-full min-h-0 bg-white relative isolate min-w-0 overflow-hidden">
        <Header
          chatTitle={isChat ? chatTitle : undefined}
          onOpenSidebar={openSidebar}
          activeConversation={isChat ? convState.activeConversation : null}
          workspaceSidebarOpen={isChat ? workspaceSidebarOpen : undefined}
          onToggleWorkspaceSidebar={isChat ? () => setWorkspaceSidebarOpen((prev) => !prev) : undefined}
        />
        <div className={`flex-1 min-h-0 ${isChat ? "overflow-hidden" : "overflow-y-auto"}`}>
          {children}
        </div>
      </main>
    </ConversationContext.Provider>
      <Toaster
        position="top-right"
        richColors
        theme="light"
        toastOptions={{
          style: {
            fontFamily: '"Open Sans", sans-serif',
            borderRadius: "0.75rem",
          },
        }}
      />
    </>
  );
}
