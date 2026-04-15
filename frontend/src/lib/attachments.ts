import type { ConversationAttachment } from "@/lib/types";

export function selectActiveAttachmentIds(
  attachments: Array<Pick<ConversationAttachment, "id">>,
): string[] {
  const seen = new Set<string>();
  const orderedIds: string[] = [];
  for (const attachment of attachments) {
    if (!seen.has(attachment.id)) {
      seen.add(attachment.id);
      orderedIds.push(attachment.id);
    }
  }
  return orderedIds;
}