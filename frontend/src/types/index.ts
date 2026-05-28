// Knowledge Base types
export interface KnowledgeBase {
  id: string;
  name: string;
  description?: string;
  embed_id?: string;
  parser_config?: {
    rag_mode?: "classic" | "llm-wiki";
    agentic_search?: boolean;
    top_n?: number;
    agentic_max_iter?: number;
    agentic_top_k_per_iter?: number;
  };
  create_time: string;
  update_time: string;
  document_count?: number;
}

export interface CreateKbRequest {
  name: string;
  description?: string;
  parser_config?: {
    rag_mode?: "classic" | "llm-wiki";
    agentic_search?: boolean;
    top_n?: number;
    agentic_max_iter?: number;
    agentic_top_k_per_iter?: number;
  };
}

export interface UpdateKbRequest {
  name?: string;
  description?: string;
  parser_config?: {
    rag_mode?: "classic" | "llm-wiki";
    agentic_search?: boolean;
    top_n?: number;
    agentic_max_iter?: number;
    agentic_top_k_per_iter?: number;
  };
}

// Document types
export type DocumentStatus = "Created" | "Processing" | "Succeed" | "Failed";
export type ParsingStatus = "Pending" | "Parsing" | "Parsed" | "Skipped" | "Failed";
export type IngestingStatus = "Pending" | "Processing" | "Succeed" | "Failed";

export interface Document {
  id: string;
  kb_id: string;
  name: string;
  s3_url?: string;
  status: DocumentStatus;
  parsing_status?: ParsingStatus;
  parsing_progress?: number;       // 0-100
  parsing_error?: string | null;
  ingesting_status?: IngestingStatus;
  ingesting_progress?: number;     // 0-100
  create_time: string;
  update_time: string;
  cmetadata?: Record<string, unknown>;
}

export interface UploadDocumentResponse {
  kb_id: string;
  uploaded_count: number;
  filenames: string[];
  status: string;
}

// Agent types
export interface Agent {
  id: string;
  name: string;
  description?: string;
  llm_model: string;
  llm_temperature: number;
  is_active: boolean;
  system_prompt?: string;
  kb_ids?: string[];
  knowledge_bases?: { id: string; name: string }[];
  create_time: string;
  update_time: string;
}

export interface CreateAgentRequest {
  name: string;
  description?: string;
  llm_model: string;
  llm_temperature?: number;
  system_prompt?: string;
}

export interface UpdateAgentRequest {
  name?: string;
  description?: string;
  llm_model?: string;
  llm_temperature?: number;
  system_prompt?: string;
}

// Chat types
export interface ChatRequest {
  message: string;
  user_id: string;
  conversation_id?: string;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  context?: string | null;
}

export interface Source {
  chunk_id: string;
  document_id: string;
  text: string;
  score: number;
  metadata?: Record<string, unknown>;
}

export interface AgenticIter {
  iter: number;
  phase:
    | "iter_start"
    | "iter_done"
    | "stop"
    | "selecting"
    | "done";
  sub_queries?: string[];
  axes?: string[];
  thought?: string;
  new_count?: number;
  total_accumulated?: number;
  top_preview?: Array<{
    doc_name?: string;
    heading_path?: string;
    chunk_type?: string;
    similarity?: number;
  }>;
  reason?: string;
  candidates?: number;
  stop_reason?: string;
  selected?: number;
  raw_accumulated?: number;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: "human" | "ai" | "system";
  content: string;
  thinking?: string;
  agentic?: AgenticIter[];
  create_time: string;
}

export interface Conversation {
  id: string;
  agent_id: string;
  user_id: string;
  title?: string;
  cmetadata?: Record<string, unknown>;
  create_time: string;
  update_time: string;
}

// Query types
export type SearchType = "semantic" | "hybrid" | "fuzzy";
export type GraphRagMode = "local" | "global" | "hybrid" | "naive" | "mix";

export interface QueryRequest {
  query_text: string;
  top_k?: number;
  search_type?: SearchType;
  alpha?: number;
  mode?: GraphRagMode;
  chunk_top_k?: number;
  max_entity_tokens?: number;
  max_relation_tokens?: number;
  max_total_tokens?: number;
}

export interface QueryResult {
  chunk_id: string;
  document_id: string;
  text: string;
  metadata?: Record<string, unknown>;
  score: number;
}

export interface ClassicQueryResponse {
  kb_id: string;
  query_type: "classic";
  search_type: SearchType;
  query_text: string;
  results: QueryResult[];
  result_count: number;
}

export interface GraphQueryResponse {
  kb_id: string;
  query_type: "graph";
  mode: GraphRagMode;
  query_text: string;
  context: string;
  keywords: Record<string, unknown>;
  entity_count: number;
  relation_count: number;
  chunk_count: number;
}

export type QueryResponse = ClassicQueryResponse | GraphQueryResponse;

// Pagination
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// API Error
export interface ApiError {
  type: string;
  title: string;
  status: number;
  detail: string;
  instance: string;
  request_id: string;
  timestamp: string;
}
