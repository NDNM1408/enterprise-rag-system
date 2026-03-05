# API Documentation

This document describes all API endpoints, request/response formats, and data structures.

## Response Envelope

All API responses use a standardized envelope:

### Success Response
```json
{
  "success": true,
  "data": <payload>,
  "meta": {
    "request_id": "uuid-string",
    "timestamp": "2026-03-03T10:30:00.000000"
  }
}
```

### Error Response (RFC 9457)
```json
{
  "type": "https://api.datahub.com/errors/not-found",
  "title": "Resource Not Found",
  "status": 404,
  "detail": "Human-readable message",
  "instance": "/api/v1/knowledge_base/123",
  "request_id": "uuid",
  "timestamp": "2026-03-03T10:30:00.000000"
}
```

**Frontend Note:** The axios interceptor in `client.ts` automatically unwraps `response.data.data` to `response.data`.

---

## Data-API (Port 8000)

### Knowledge Base Endpoints

#### List Knowledge Bases
```
GET /api/v1/knowledge_base/
```

**Query Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| page | int | 1 | Page number (1-indexed) |
| page_size | int | 10 | Items per page (max 100) |
| filter | string | null | JSON filter string |
| sort | string | null | JSON sort string |

**Response:**
```json
{
  "total": 100,
  "page": 1,
  "page_size": 10,
  "items": [
    {
      "id": "uuid",
      "name": "string",
      "description": "string | null",
      "tenant_id": "string | null",
      "created_by": "string | null",
      "embed_id": "string",
      "parser_id": "string | null",
      "parser_config": { "rag_mode": "classic" | "graphrag" },
      "create_time": "2026-03-03T10:30:00",
      "update_time": "2026-03-03T10:30:00",
      "document_count": 5
    }
  ]
}
```

#### Create Knowledge Base
```
POST /api/v1/knowledge_base
```

**Request Body:**
```json
{
  "name": "string (required)",
  "description": "string (optional)",
  "parser_config": {
    "rag_mode": "classic" | "graphrag"
  }
}
```

**Response:** Single KB object

#### Get Knowledge Base
```
GET /api/v1/knowledge_base/{kb_id}
```

**Response:** Single KB object (same structure as items above, without document_count)

#### Delete Knowledge Base
```
DELETE /api/v1/knowledge_base/{kb_id}
```

**Response:**
```json
{
  "kb_id": "uuid",
  "deleted": true
}
```

---

### Document Endpoints

#### List Documents
```
GET /api/v1/{kb_id}/documents
```

**Response:**
```json
{
  "kb_id": "uuid",
  "count": 5,
  "documents": [
    {
      "id": "uuid",
      "name": "string",
      "kb_id": "uuid",
      "status": "Created" | "Processing" | "Succeed" | "Failed",
      "etag": "string | null",
      "cmetadata": {},
      "create_time": "2026-03-03T10:30:00",
      "update_time": "2026-03-03T10:30:00"
    }
  ]
}
```

**Frontend Note:** Access via `response.data.documents` (not `items`).

#### Upload Documents
```
POST /api/v1/{kb_id}/documents
```

**Request:** `multipart/form-data`
- `files`: List of files (required)
- `cmetadata`: JSON string (optional)

**Response:**
```json
{
  "kb_id": "uuid",
  "uploaded_count": 2,
  "filenames": ["file1.html", "file2.html"],
  "status": "processing"
}
```

#### Delete Document
```
DELETE /api/v1/{kb_id}/documents/{doc_id}
```

**Response:**
```json
{
  "kb_id": "uuid",
  "doc_id": "uuid",
  "deleted": true
}
```

---

### Query Endpoints

#### Query Knowledge Base
```
POST /api/v1/query/{kb_id}
```

**Request Body:**
```json
{
  "query_text": "string (required)",
  "top_k": 10,
  "search_type": "semantic" | "hybrid" | "fuzzy",
  "alpha": 0.5,
  "mode": "local" | "global" | "hybrid" | "naive" | "mix",
  "chunk_top_k": 10,
  "max_entity_tokens": 4000,
  "max_relation_tokens": 4000,
  "max_total_tokens": 16000
}
```

**Classic RAG Response:**
```json
{
  "kb_id": "uuid",
  "query_type": "classic",
  "search_type": "semantic",
  "query_text": "string",
  "results": [
    {
      "chunk_id": "uuid",
      "document_id": "uuid",
      "text": "string",
      "metadata": {},
      "score": 0.85
    }
  ],
  "result_count": 10
}
```

**Graph RAG Response:**
```json
{
  "kb_id": "uuid",
  "query_type": "graph",
  "mode": "hybrid",
  "query_text": "string",
  "context": "string",
  "keywords": {},
  "entity_count": 5,
  "relation_count": 3,
  "chunk_count": 2
}
```

---

## Chatbot-Service (Port 8001)

### Agent Endpoints

#### List Agents
```
GET /api/v1/agents
```

**Query Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| skip | int | 0 | Offset |
| limit | int | 10 | Max items |
| tenant_id | string | null | Filter by tenant |
| is_active | bool | null | Filter by active status |

**Response:**
```json
[
  {
    "id": "uuid",
    "name": "string",
    "description": "string | null",
    "system_prompt": "string | null",
    "llm_model": "string",
    "llm_temperature": 0.7,
    "is_active": true,
    "tenant_id": "string | null",
    "created_by": "string | null",
    "create_time": "2026-03-03T10:30:00",
    "update_time": "2026-03-03T10:30:00",
    "kb_ids": ["uuid1", "uuid2"]
  }
]
```

**Frontend Note:** Returns array directly (no pagination wrapper).

#### Create Agent
```
POST /api/v1/agents
```

**Request Body:**
```json
{
  "name": "string (required)",
  "description": "string (optional)",
  "system_prompt": "string (optional)",
  "llm_model": "string (required)",
  "llm_temperature": 0.7
}
```

**Response:** Single agent object

#### Get Agent
```
GET /api/v1/agents/{agent_id}
```

**Response:** Single agent object

#### Update Agent
```
PUT /api/v1/agents/{agent_id}
```

**Request Body:** Same as create (all fields optional)

**Response:** Single agent object

#### Delete Agent
```
DELETE /api/v1/agents/{agent_id}
```

**Response:**
```json
{
  "deleted": true
}
```

#### Link Knowledge Base
```
POST /api/v1/agents/{agent_id}/kb
```

**Request Body:**
```json
{
  "kb_id": "string (required)"
}
```

**Response:**
```json
{
  "agent_id": "uuid",
  "kb_id": "uuid",
  "link_id": "uuid"
}
```

#### Unlink Knowledge Base
```
DELETE /api/v1/agents/{agent_id}/kb/{kb_id}
```

**Response:**
```json
{
  "unlinked": true
}
```

---

### Chat Endpoints

#### Send Message
```
POST /api/v1/agents/{agent_id}/chat
```

**Request Body:**
```json
{
  "message": "string (required)",
  "user_id": "string (required)",
  "conversation_id": "string (optional)"
}
```

**Response:**
```json
{
  "response": "string",
  "conversation_id": "uuid",
  "context": "string | null"
}
```

#### Stream Message
```
POST /api/v1/agents/{agent_id}/chat/stream
```

**Request Body:** Same as regular chat

**Response:** `text/event-stream`
```
data: chunk1
data: chunk2
data: [DONE]
```

---

### Conversation Endpoints

#### List Conversations
```
GET /api/v1/conversations
```

**Query Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| user_id | string | required | Filter by user |
| agent_id | string | null | Filter by agent |
| skip | int | 0 | Offset |
| limit | int | 20 | Max items |

**Response:**
```json
[
  {
    "id": "uuid",
    "agent_id": "uuid",
    "user_id": "uuid",
    "title": "string | null",
    "cmetadata": {},
    "create_time": "2026-03-03T10:30:00",
    "update_time": "2026-03-03T10:30:00"
  }
]
```

#### Get Conversation History
```
GET /api/v1/conversations/{conversation_id}/messages
```

**Query Parameters:**
| Name | Type | Description |
|------|------|-------------|
| user_id | string | Required |

**Response:**
```json
[
  {
    "id": "uuid",
    "conversation_id": "uuid",
    "role": "human" | "ai" | "system",
    "content": "string",
    "create_time": "2026-03-03T10:30:00"
  }
]
```

---

## Field Naming Conventions

| Backend Field | Frontend Should Use | Notes |
|--------------|---------------------|-------|
| `create_time` | `create_time` | Not `created_at` |
| `update_time` | `update_time` | Not `updated_at` |
| `kb_id` | `kb_id` | Knowledge base ID |
| `doc_id` | `doc_id` | Document ID |

## Common Issues & Solutions

### 1. Response Unwrapping
The axios interceptor unwraps `{success, data, meta}` to just `data`. Access response directly:
```typescript
const response = await apiClient.get("/endpoint");
// response.data is already unwrapped
```

### 2. Array Field Names
Different endpoints use different array field names:
- KB list: `response.data.items`
- Document list: `response.data.documents`
- Query results: `response.data.results`
- Agent list: `response.data` (direct array)

### 3. Polling for Status
Document processing is async. Poll every 5 seconds when documents have `status: "Processing"`:
```typescript
refetchInterval: (query) => {
  const hasProcessing = query.state.data?.some((doc) => doc.status === "Processing");
  return hasProcessing ? 5000 : false;
}
```
