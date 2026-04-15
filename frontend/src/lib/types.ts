export interface ToolExecution {
  name: string;
  status: "success" | "error";
}

export interface Source {
  type: "web" | "rag";
  label: string;
  url: string | null;
  index?: number;
}

export interface MessageMetrics {
  response_time_ms?: number;
  input_tokens?: number;
  output_tokens?: number;
  finish_reason?: "stop" | "tool_use" | "max_tokens";
  model?: string;
  temperature?: number;
  tools_executed?: ToolExecution[];
  sources?: Source[];
  attachments_used?: Array<{ id: string; original_name: string }>;
  documents_used?: Array<{ id: string; filename: string; version: number }>;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
  metrics?: MessageMetrics;
  persistedId?: number;
  feedback?: "up" | "down";
}

export interface ChatResponse {
  response: string;
  metadata: {
    tokens_used: number;
    input_tokens?: number;
    output_tokens?: number;
    model_used?: string;
    tools_executed: string[];
    sources?: Source[];
    attachments_used?: Array<{
      id: string;
      original_name: string;
    }>;
    documents_used?: Array<{
      id: string;
      filename: string;
      version: number;
    }>;
    workspace_document_writeback?: {
      status: "saved";
      document_id: string;
      filename: string;
      version: number;
      size_bytes: number;
    } | null;
    workspace_action?: {
      operation: "copy_to_workspace" | "read_workspace_file" | "write_workspace_file" | "write_revised_copy";
      workspace?: {
        conversation_id: string;
        user_id: string;
        root_path: string;
        status: "active" | "expired" | "deleted";
        created_at?: string | null;
        updated_at?: string | null;
        last_activity_at?: string | null;
        expires_at?: string | null;
        files?: Array<{
          name: string;
          relative_path: string;
          size_bytes: number;
          modified_at?: string | null;
        }>;
      };
      file?: {
        name: string;
        relative_path: string;
        size_bytes: number;
        modified_at?: string | null;
      };
      path?: string;
      revised_path?: string | null;
    };
    model_warning?: string;
  };
}

export interface WorkspaceActionRequest {
  operation: "copy_to_workspace" | "read_workspace_file" | "write_workspace_file" | "write_revised_copy";
  attachment_id?: string;
  path?: string;
  target_name?: string;
  content?: string;
}

export interface ConversationAttachment {
  id: string;
  conversation_id: string;
  user_id: string;
  original_name: string;
  mime_type: string;
  size_bytes: number;
  status: "active" | "deleted" | "expired";
  created_at: string | null;
  expires_at: string | null;
}

// ── Conversations ────────────────────────────────────────────────────

export interface Conversation {
  id: string;
  user_id: string;
  title: string;
  language: string;
  fork_name?: string | null;
  archived: boolean;
  pinned: boolean;
  metadata: Record<string, unknown>;
  last_message?: string | null;
  message_count?: number | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface PersistedMessage {
  id: number;
  conversation_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  tokens_used?: number | null;
  model_name?: string | null;
  response_time_ms?: number | null;
  tools_used?: ToolExecution[] | null;
  feedback?: "up" | "down" | null;
  metadata: Record<string, unknown>;
  created_at: string | null;
}

export interface ConversationListResponse {
  conversations: Conversation[];
  total: number;
  limit: number;
  offset: number;
}

export interface ConversationDetailResponse {
  conversation: Conversation;
  messages: PersistedMessage[];
}

export interface SearchResult {
  message_id: number;
  conversation_id: string;
  conversation_title: string;
  role: string;
  content: string;
  created_at: string;
}
