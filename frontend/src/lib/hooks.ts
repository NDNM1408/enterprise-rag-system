"use client";

import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { api, litellmApi } from "./api";
import { useChatStore } from "./stores";
import type { CreateAgentInput } from "./schemas";
import type {
  Agent,
  CreateKbRequest,
  Document,
  KnowledgeBase,
} from "@/types";

// ---------------------------------------------------------------------------
//  Documents
// ---------------------------------------------------------------------------

function isDocumentActive(doc: Document): boolean {
  return (
    doc.status === "Created" ||
    doc.status === "Processing" ||
    doc.parsing_status === "Pending" ||
    doc.parsing_status === "Parsing" ||
    doc.ingesting_status === "Pending" ||
    doc.ingesting_status === "Processing"
  );
}

interface ListDocumentsResponse {
  kb_id: string;
  count: number;
  documents: Document[];
}

export function useDocuments(kbId: string) {
  return useQuery<Document[]>({
    queryKey: ["documents", kbId],
    queryFn: async () => {
      const { data } = await api.get<ListDocumentsResponse>(
        `/api/v1/${kbId}/documents`,
      );
      return data.documents ?? [];
    },
    // Poll while any document is mid-pipeline. As soon as everything is
    // terminal (Succeed/Failed) the polling stops.
    refetchInterval: (query) =>
      query.state.data?.some(isDocumentActive) ? 2500 : false,
  });
}

export function useUploadDocuments() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ kbId, files }: { kbId: string; files: File[] }) => {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));
      const { data } = await api.post(`/api/v1/${kbId}/documents`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      return data;
    },
    onSuccess: (_data, { kbId }) => {
      qc.invalidateQueries({ queryKey: ["documents", kbId] });
    },
  });
}

export function useDeleteDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ kbId, docId }: { kbId: string; docId: string }) => {
      const { data } = await api.delete(`/api/v1/${kbId}/documents/${docId}`);
      return data;
    },
    onSuccess: (_data, { kbId }) => {
      qc.invalidateQueries({ queryKey: ["documents", kbId] });
    },
  });
}

export interface DocumentPreviewChunk {
  id: string;
  content: string;
  parent_text: string;
  status: string;
}

export interface DocumentPreview {
  doc_id: string;
  kb_id: string;
  name: string;
  status: string;
  parsing_status?: string;
  ingesting_status?: string;
  parsed_markdown: string | null;
  /** "parsed" = MinerU output | "original" = native-text source | null */
  parsed_source?: "parsed" | "original" | null;
  parsed_markdown_s3_key?: string | null;
  chunks: DocumentPreviewChunk[];
  chunk_count: number;
}

export function useDocumentPreview(kbId: string, docId: string | null) {
  return useQuery<DocumentPreview>({
    queryKey: ["document-preview", kbId, docId],
    queryFn: async () => {
      const { data } = await api.get<DocumentPreview>(
        `/api/v1/${kbId}/documents/${docId}/preview`,
      );
      return data;
    },
    enabled: !!kbId && !!docId,
    staleTime: 60_000,
  });
}

// ---------------------------------------------------------------------------
//  Knowledge bases
// ---------------------------------------------------------------------------

interface PagedKbResponse {
  total: number;
  page: number;
  page_size: number;
  items: KnowledgeBase[];
}

export function useKnowledgeBases() {
  return useQuery<KnowledgeBase[]>({
    queryKey: ["knowledge-bases"],
    queryFn: async () => {
      const { data } = await api.get<PagedKbResponse>("/api/v1/knowledge_base/", {
        params: { page: 1, page_size: 100 },
      });
      return data.items ?? [];
    },
  });
}

export function useKnowledgeBase(kbId: string) {
  return useQuery<KnowledgeBase>({
    queryKey: ["knowledge-base", kbId],
    queryFn: async () => {
      const { data } = await api.get(`/api/v1/knowledge_base/${kbId}`);
      return data;
    },
    enabled: !!kbId,
  });
}

export function useCreateKnowledgeBase() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: CreateKbRequest) => {
      const { data } = await api.post("/api/v1/knowledge_base", body);
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["knowledge-bases"] });
    },
  });
}

// ---------------------------------------------------------------------------
//  KB query — retrieve top-K chunks for a free-form query
// ---------------------------------------------------------------------------

export interface RetrievedChunk {
  chunk_id: string;
  document_id: string;
  text: string;          // = content (with heading prefix; the embed string)
  parent_text?: string;  // full leaf section — what the LLM would see on hit
  metadata?: {
    tokens?: number;
    heading_path?: string | null;
    chunk_order_index?: number;
    [k: string]: unknown;
  };
  similarity?: number;   // 0..1 — classic vector similarity
  score?: number;        // hybrid / fuzzy fallback name
}

export interface QueryResult {
  kb_id: string;
  query_type: string;
  search_type: string;
  query_text: string;
  results: RetrievedChunk[];
  result_count: number;
}

export type SearchType = "semantic" | "hybrid" | "fuzzy";

export interface QueryKbVars {
  kbId: string;
  queryText: string;
  topK?: number;
  searchType?: SearchType;
  alpha?: number;
}

export function useQueryKnowledgeBase() {
  return useMutation<QueryResult, Error, QueryKbVars>({
    mutationFn: async ({ kbId, queryText, topK = 5, searchType = "semantic", alpha = 0.5 }) => {
      const { data } = await api.post<QueryResult>(`/api/v1/query/${kbId}`, {
        query_text: queryText,
        top_k: topK,
        search_type: searchType,
        alpha,
      });
      return data;
    },
  });
}

export function useDeleteKnowledgeBase() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (kbId: string) => {
      const { data } = await api.delete(`/api/v1/knowledge_base/${kbId}`);
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["knowledge-bases"] });
    },
  });
}

// ---------------------------------------------------------------------------
//  Agents + Chat (chatbot-service)
// ---------------------------------------------------------------------------

// Hardcoded for now — replace with auth/session user_id when auth lands.
const DEFAULT_USER_ID = "default-user";

export function useAgents() {
  return useQuery<Agent[]>({
    queryKey: ["agents"],
    queryFn: async () => {
      const { data } = await api.get<Agent[]>("/api/v1/agents");
      return data ?? [];
    },
  });
}

export function useAgent(agentId: string) {
  return useQuery<Agent>({
    queryKey: ["agent", agentId],
    queryFn: async () => {
      const { data } = await api.get<Agent>(`/api/v1/agents/${agentId}`);
      return data;
    },
    enabled: !!agentId,
  });
}

interface LiteLLMModel {
  id: string;
  object: string;
  owned_by?: string;
}

interface LiteLLMModelsResponse {
  data: LiteLLMModel[];
  object: string;
}

/** List models exposed by the LiteLLM proxy ``/v1/models`` endpoint.
 *
 * Filters out embedding/rerank/image models so only chat-capable IDs
 * surface in the agent-creation dropdown. Heuristic by name suffix —
 * LiteLLM doesn't expose model capability metadata in the OpenAI-compatible
 * response.
 */
export function useLiteLLMModels() {
  return useQuery<string[]>({
    queryKey: ["litellm-models"],
    queryFn: async () => {
      const { data } = await litellmApi.get<LiteLLMModelsResponse>("/v1/models");
      const ids = (data?.data ?? []).map((m) => m.id);
      return ids
        .filter((id) => {
          const lower = id.toLowerCase();
          return !(
            lower.includes("embed") ||
            lower.includes("rerank") ||
            lower.includes("image") ||
            lower.includes("vision-tts") ||
            lower.includes("whisper") ||
            lower.includes("tts")
          );
        })
        .sort();
    },
    staleTime: 60_000,
  });
}

export function useCreateAgent() {
  const qc = useQueryClient();
  return useMutation<Agent, Error, CreateAgentInput>({
    mutationFn: async (body) => {
      const { data } = await api.post<Agent>("/api/v1/agents", {
        ...body,
        created_by: DEFAULT_USER_ID,
      });
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agents"] });
    },
  });
}

export function useDeleteAgent() {
  const qc = useQueryClient();
  return useMutation<void, Error, string>({
    mutationFn: async (agentId) => {
      await api.delete(`/api/v1/agents/${agentId}`);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agents"] });
    },
  });
}

interface AgentKbVars {
  agentId: string;
  kbId: string;
}

export function useLinkKnowledgeBase() {
  const qc = useQueryClient();
  return useMutation<void, Error, AgentKbVars>({
    mutationFn: async ({ agentId, kbId }) => {
      await api.post(`/api/v1/agents/${agentId}/kb`, { kb_id: kbId });
    },
    onSuccess: (_data, { agentId }) => {
      qc.invalidateQueries({ queryKey: ["agent", agentId] });
      qc.invalidateQueries({ queryKey: ["agents"] });
    },
  });
}

export function useUnlinkKnowledgeBase() {
  const qc = useQueryClient();
  return useMutation<void, Error, AgentKbVars>({
    mutationFn: async ({ agentId, kbId }) => {
      await api.delete(`/api/v1/agents/${agentId}/kb/${kbId}`);
    },
    onSuccess: (_data, { agentId }) => {
      qc.invalidateQueries({ queryKey: ["agent", agentId] });
      qc.invalidateQueries({ queryKey: ["agents"] });
    },
  });
}

/**
 * Stream a chat completion from the agent — reads ``text/event-stream`` and
 * updates the chat store token-by-token.
 *
 * Flow:
 *   1. Optimistically push the user message.
 *   2. Set ``isStreaming = true`` so the UI can render a placeholder bubble
 *      with a typing indicator.
 *   3. POST to ``/chat/stream`` and consume the SSE body line-by-line.
 *   4. For each ``data: <chunk>`` line, append to ``currentStreamingContent``.
 *   5. On ``data: [DONE]``, commit the assembled content as a final AI
 *      message, clear the streaming buffer, and reset ``isStreaming``.
 */
export function useChat(agentId: string) {
  const qc = useQueryClient();
  const {
    conversationId,
    setConversationId,
    addMessage,
    setStreaming,
    setStreamingContent,
    setStreamingThinking,
  } = useChatStore();

  return useMutation<void, Error, string>({
    mutationFn: async (message: string) => {
      // 1. Show the user message instantly.
      addMessage({
        id: `local-${Date.now()}`,
        conversation_id: conversationId ?? "",
        role: "human",
        content: message,
        create_time: new Date().toISOString(),
      });

      // 2. Switch the UI into streaming state — empty placeholder bubble
      //    will render a typing indicator until the first token arrives.
      setStreamingContent("");
      setStreamingThinking("");
      setStreaming(true);

      const url = `${api.defaults.baseURL ?? ""}/api/v1/agents/${agentId}/chat/stream`;
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          user_id: DEFAULT_USER_ID,
          conversation_id: conversationId ?? undefined,
        }),
      });

      if (!response.ok || !response.body) {
        setStreaming(false);
        throw new Error(`Stream failed: HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let assembled = "";
      let assembledThinking = "";

      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          // SSE frames are separated by a blank line.
          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";

          for (const frame of frames) {
            // Each frame is one or more `data: ...` lines.
            const lines = frame.split("\n");
            for (const line of lines) {
              if (!line.startsWith("data: ")) continue;
              const payload = line.slice("data: ".length);
              if (payload === "[DONE]") {
                continue;
              }
              // Backend emits JSON-encoded events:
              //   {type: "meta",     conversation_id}
              //   {type: "delta",    content}    ← answer tokens
              //   {type: "thinking", delta}      ← reasoning tokens
              try {
                const event = JSON.parse(payload) as {
                  type: string;
                  content?: string;
                  delta?: string;
                  conversation_id?: string;
                };
                if (event.type === "meta" && event.conversation_id) {
                  setConversationId(event.conversation_id);
                } else if (event.type === "delta" && event.content) {
                  assembled += event.content;
                  setStreamingContent(assembled);
                } else if (event.type === "thinking" && event.delta) {
                  assembledThinking += event.delta;
                  setStreamingThinking(assembledThinking);
                }
              } catch {
                // Tolerate any non-JSON frame by appending it raw — keeps
                // backwards-compat if the server ever sends plain chunks.
                assembled += payload;
                setStreamingContent(assembled);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }

      // 5. Commit the final response as a real AI message.
      addMessage({
        id: `local-${Date.now()}-ai`,
        conversation_id: conversationId ?? "",
        role: "ai",
        content: assembled,
        thinking: assembledThinking || undefined,
        create_time: new Date().toISOString(),
      });
      setStreamingContent("");
      setStreamingThinking("");
      setStreaming(false);

      // The backend assigns / reuses a conversation id; refetch the history
      // to capture it for follow-up turns.
      qc.invalidateQueries({ queryKey: ["conversations"] });
    },
    onError: () => {
      setStreaming(false);
      setStreamingContent("");
    },
  });
}
