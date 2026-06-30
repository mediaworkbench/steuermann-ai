"use client";

import { DissonanceQueueSection } from "@/components/product/DissonanceQueueSection";
import { ProceduralApprovalsSection } from "@/components/product/ProceduralApprovalsSection";
import { MemoryUndoSection } from "@/components/product/MemoryUndoSection";
import { useI18n } from "@/hooks/useI18n";

/**
 * User-scoped memory review (plan-memory.md Phase 6): resolve dissonance conflicts,
 * approve/reject learned procedural preferences, and undo recent engine actions.
 * Every section is self-scoped — the backend reads the user_id from the session only.
 */
export default function MemoryDreamingPage() {
  const { t } = useI18n();

  return (
    <div className="flex-1 overflow-y-auto bg-background">
      <div className="mx-auto w-full px-4 py-6 md:px-8 md:py-8 space-y-8 lg:px-12">
        <div>
          <h1 className="text-3xl font-bold text-foreground">{t("dreaming.userTitle")}</h1>
          <p className="mt-1 text-muted-foreground">{t("dreaming.userSubtitle")}</p>
        </div>

        <DissonanceQueueSection />
        <ProceduralApprovalsSection />
        <MemoryUndoSection />
      </div>
    </div>
  );
}
