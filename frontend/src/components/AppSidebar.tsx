"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import {
  BarChart3,
  Brain,
  Compass,
  Download,
  LogOut,
  MoreVertical,
  Pencil,
  Pin,
  Plus,
  Settings,
  ShieldCheck,
  Trash2,
  Users,
} from "lucide-react";

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Avatar,
  AvatarFallback,
} from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ConfirmDialog } from "./ConfirmDialog";
import { ExportDialog } from "./ExportDialog";
import { useI18n } from "@/hooks/useI18n";
import { useProfile } from "@/hooks/useProfile";
import { useRole } from "@/context/RoleContext";
import { useConversationContext } from "./LayoutShell";
import { AUTH_ENABLED, SINGLE_USER_DISPLAY_NAME } from "@/lib/runtime";
import type { Conversation } from "@/lib/types";

const RECENT_LIMIT = 5;

function getInitials(name: string): string {
  return name
    .split(/\s+/)
    .map((w) => w[0])
    .filter(Boolean)
    .slice(0, 2)
    .join("")
    .toUpperCase();
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const pathname = usePathname();
  const router = useRouter();
  const { t } = useI18n();
  const { isAdmin, canAccessRag } = useRole();
  const {
    conversations,
    activeId,
    setActiveId,
    update,
    remove,
    rename,
  } = useConversationContext();

  const pinned = conversations.filter((c) => c.pinned);
  const recent = conversations.filter((c) => !c.pinned).slice(0, RECENT_LIMIT);

  const handleNewChat = useCallback(async () => {
    setActiveId(null);
    if (pathname !== "/") router.push("/");
  }, [setActiveId, pathname, router]);

  const handleSelect = useCallback(
    (id: string) => {
      setActiveId(id);
      if (pathname !== "/") router.push("/");
    },
    [setActiveId, pathname, router]
  );

  const handlePin = useCallback(
    async (id: string, pinned: boolean) => {
      await update(id, { pinned });
    },
    [update]
  );

  const profile = useProfile();
  const appTitle = profile.appName || SINGLE_USER_DISPLAY_NAME;
  const frameworkVersion = profile.frameworkVersion || "unknown";
  const initials = getInitials(SINGLE_USER_DISPLAY_NAME);

  const settingsLink = useCallback(() => {
    router.push("/settings");
  }, [router]);

  const memoryLink = useCallback(() => {
    router.push("/memories");
  }, [router]);

  const handleLogout = useCallback(async () => {
    await fetch("/api/auth/logout", { method: "POST" });
    window.location.assign("/login");
  }, []);

  return (
    <>
      <Sidebar collapsible="offcanvas" {...props}>
        <SidebarHeader className="min-h-16 md:min-h-20 justify-center px-4">
          <div className="flex flex-col w-full">
            <span className="text-sidebar-foreground text-base font-bold truncate">{appTitle}</span>
            {frameworkVersion !== "unknown" && (
              <span className="text-sidebar-foreground/50 text-xs font-mono">v{frameworkVersion}</span>
            )}
          </div>
        </SidebarHeader>

        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupContent>
              <div className="px-2 pt-1 pb-2">
                <Button
                  size="lg"
                  variant="default"
                  onClick={handleNewChat}
                  className="w-full justify-center gap-2"
                >
                  <Plus size={18} />
                  <span>{t("sidebar.newChat")}</span>
                </Button>
              </div>
            </SidebarGroupContent>
          </SidebarGroup>

          {pinned.length > 0 && (
            <SidebarGroup>
              <SidebarGroupLabel>{t("sidebar.pinned")}</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {pinned.map((c) => (
                    <ConversationRow
                      key={c.id}
                      conversation={c}
                      isActive={c.id === activeId}
                      onSelect={handleSelect}
                      onPin={handlePin}
                      onDelete={remove}
                      onRename={rename}
                    />
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          )}

          {recent.length > 0 && (
            <SidebarGroup>
              <SidebarGroupLabel>{t("sidebar.recentChats")}</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {recent.map((c) => (
                    <ConversationRow
                      key={c.id}
                      conversation={c}
                      isActive={c.id === activeId}
                      onSelect={handleSelect}
                      onPin={handlePin}
                      onDelete={remove}
                      onRename={rename}
                    />
                  ))}
                </SidebarMenu>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <button
                      onClick={() => router.push("/chats")}
                      className="inline-flex items-center w-full rounded-md py-1.5 text-xs text-muted-foreground hover:text-foreground hover:bg-sidebar-accent transition-colors"
                    >
                      {t("sidebar.seeAll")}
                    </button>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          )}

          {conversations.length === 0 && (
            <p className="text-sidebar-foreground/70 text-xs text-center mt-4 px-2">
              {t("sidebar.noConversations")}
            </p>
          )}

          {(isAdmin || canAccessRag) && (
            <SidebarGroup>
              <SidebarGroupLabel>{t("sidebar.administration")}</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {isAdmin && (
                    <SidebarMenuItem>
                      <SidebarMenuButton
                        tooltip={t("header.metrics")}
                        onClick={() => router.push("/metrics")}
                        className="justify-start"
                      >
                        <BarChart3 />
                        <span>{t("header.metrics")}</span>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  )}
                  {canAccessRag && (
                    <SidebarMenuItem>
                      <SidebarMenuButton
                        tooltip={t("header.ragExplorer")}
                        onClick={() => router.push("/admin/rag")}
                        className="justify-start"
                      >
                        <Compass />
                        <span>{t("header.ragExplorer")}</span>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  )}
                  {isAdmin && (
                    <SidebarMenuItem>
                      <SidebarMenuButton
                        tooltip="Users"
                        onClick={() => router.push("/admin/users")}
                        className="justify-start"
                      >
                        <Users />
                        <span>Users</span>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  )}
                  {isAdmin && (
                    <SidebarMenuItem>
                      <SidebarMenuButton
                        tooltip={t("header.admin")}
                        onClick={() => router.push("/admin")}
                        className="justify-start"
                      >
                        <ShieldCheck />
                        <span>{t("header.admin")}</span>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  )}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          )}
        </SidebarContent>

        <SidebarFooter>
          <SidebarMenu>
            <SidebarMenuItem>
              <DropdownMenu>
                <DropdownMenuTrigger
                  render={
                    <SidebarMenuButton
                      size="lg"
                      className="justify-start data-popup-open:bg-sidebar-accent data-popup-open:text-sidebar-accent-foreground"
                    >
                      <Avatar size="sm">
                        <AvatarFallback>{initials}</AvatarFallback>
                      </Avatar>
                      <div className="grid flex-1 text-left text-sm leading-tight">
                        <span className="truncate font-semibold">
                          {SINGLE_USER_DISPLAY_NAME}
                        </span>
                      </div>
                    </SidebarMenuButton>
                  }
                />
                <DropdownMenuContent
                  side="right"
                  align="end"
                  sideOffset={8}
                  className="w-48"
                >
                  <DropdownMenuItem onClick={memoryLink}>
                    <Brain />
                    <span>{t("header.memory")}</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={settingsLink}>
                    <Settings />
                    <span>{t("header.settings")}</span>
                  </DropdownMenuItem>
                  {AUTH_ENABLED && (
                    <>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        variant="destructive"
                        onClick={handleLogout}
                      >
                        <LogOut />
                        <span>{t("header.signOut")}</span>
                      </DropdownMenuItem>
                    </>
                  )}
                </DropdownMenuContent>
              </DropdownMenu>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarFooter>
      </Sidebar>
      <SidebarRail />
    </>
  );
}

// ── Conversation Row ──────────────────────────────────────────────

function ConversationRow({
  conversation: c,
  isActive,
  onSelect,
  onPin,
  onDelete,
  onRename,
}: {
  conversation: Conversation;
  isActive: boolean;
  onSelect: (id: string) => void;
  onPin: (id: string, pinned: boolean) => Promise<void>;
  onDelete: (id: string) => Promise<boolean>;
  onRename: (id: string, title: string) => Promise<Conversation | null>;
}) {
  const { t } = useI18n();
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState(c.title);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [showExport, setShowExport] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  const commitRename = useCallback(async () => {
    const trimmed = editValue.trim();
    if (trimmed && trimmed !== c.title) {
      await onRename(c.id, trimmed);
    } else {
      setEditValue(c.title);
    }
    setEditing(false);
  }, [editValue, c.id, c.title, onRename]);

  const startRename = useCallback(() => {
    setEditValue(c.title);
    setEditing(true);
  }, [c.title]);

  return (
    <SidebarMenuItem>
      {editing ? (
        <div className="flex items-center w-full px-1">
          <Input
            ref={inputRef}
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={commitRename}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitRename();
              if (e.key === "Escape") {
                setEditValue(c.title);
                setEditing(false);
              }
            }}
            onClick={(e) => e.stopPropagation()}
            className="flex-1 min-w-0 h-8 text-sm"
          />
        </div>
      ) : (
        <SidebarMenuButton
          isActive={isActive}
          tooltip={c.title}
          onClick={() => onSelect(c.id)}
          onDoubleClick={(e) => {
            e.preventDefault();
            startRename();
          }}
          className="justify-start"
        >
          <span className="truncate">{c.title}</span>
        </SidebarMenuButton>
      )}

      {!editing && (
        <DropdownMenu>
          <DropdownMenuTrigger
            render={
              <SidebarMenuAction showOnHover>
                <MoreVertical />
                <span className="sr-only">{t("sidebar.moreOptions")}</span>
              </SidebarMenuAction>
            }
          />
          <DropdownMenuContent side="right" align="start" sideOffset={4}>
            <DropdownMenuItem onClick={startRename}>
              <Pencil />
              <span>{t("sidebar.rename")}</span>
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => onPin(c.id, !c.pinned)}>
              <Pin />
              <span>
                {c.pinned ? t("sidebar.unpin") : t("sidebar.pin")}
              </span>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => setShowExport(true)}>
              <Download />
              <span>{t("common.export")}</span>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              variant="destructive"
              onClick={() => setConfirmDelete(true)}
            >
              <Trash2 />
              <span>{t("sidebar.delete")}</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )}

      <ConfirmDialog
        isOpen={confirmDelete}
        title={t("sidebar.delete")}
        message={t("sidebar.deleteConversationConfirm")}
        variant="danger"
        confirmLabel={t("common.delete")}
        onConfirm={() => {
          setConfirmDelete(false);
          onDelete(c.id);
        }}
        onCancel={() => setConfirmDelete(false)}
      />
      {showExport && (
        <ExportDialog
          conversationId={c.id}
          conversationTitle={c.title}
          onClose={() => setShowExport(false)}
        />
      )}
    </SidebarMenuItem>
  );
}
