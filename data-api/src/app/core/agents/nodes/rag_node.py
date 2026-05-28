"""
RAG node for retrieving context and generating responses.

Pipeline (extract-then-answer, v4):
  1. Retrieve top-K child chunks; dedupe by parent_id → list of parents.
  2. For each parent, load cached facts (``parent_facts`` table). Parents
     never seen before are fact-extracted in parallel and the results are
     persisted, so each parent is extracted exactly once across all queries.
  3. Answer from the aggregated facts (thinking on) instead of raw parent
     text — exhaustive per-row extraction surfaces details (e.g. BYOK) that
     a single-pass read over long sections tends to drop.
"""

import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional, AsyncIterator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.core.agents.state import ChatbotState
from app.configurations.configurations import settings
from app.infrastructure.clients.litellm_client import LiteLLMClient
from app.infrastructure.clients.data_api_client import DataApiClient
from app.infrastructure.clients.embedding_client import EmbeddingClient
from app.infrastructure.connectors.postgres.database import db_session
from app.infrastructure.connectors.postgres.repositories.knowledge_base_repository import (
    KnowledgeBaseRepository,
)
from app.infrastructure.connectors.postgres.repositories.parent_facts_repository import (
    ParentFactsRepository,
)
from app.exceptions import ExternalServiceError


logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Use the provided context to answer the user's question accurately and concisely.

If the context doesn't contain relevant information, say so and provide a general response based on your knowledge.

Always be helpful, accurate, and professional."""

# Query-INDEPENDENT extraction so the result is cacheable per parent and
# reused across every future query that retrieves the same parent.
EXTRACT_SYSTEM_PROMPT = """You are an expert fact extractor for a document knowledge base.

Given a SECTION of a document, extract every atomic fact it states.

Rules:
1. ONLY extract facts present in the section. Do NOT add outside knowledge.
2. Each fact is one concise, self-contained atomic statement (one sentence).
3. Be exhaustive — include ALL facts, even small details.
4. Preserve specific entities, numbers, names, dates, identifiers, and terminology exactly as written in the section.
5. If the section contains a table, extract one fact per row, carrying the row's key/label into the fact so it stays self-contained.
6. Split bullet lists and enumerations into separate facts.
7. Do NOT answer any question — just extract facts from this section.

Output STRICT JSON only: {"facts": ["fact 1", "fact 2", ...]}"""

ANSWER_FROM_FACTS_PROMPT = """{base}

You are given a QUESTION and a list of FACTS extracted from the retrieved documents.

Rules:
1. Answer the question using ONLY the provided facts. Do NOT add outside knowledge.
2. Synthesize the facts into a coherent, complete answer.
3. If the facts are insufficient, say so honestly — but only after using whatever facts ARE provided.
4. Match the question's expected breadth — if asked "what specifically did they build", list the specific architectural components / technologies / requirement IDs by name.
5. Be on-point. Bullet points are fine for lists."""


class RAGNode:
    """Node for RAG-based response generation."""

    def __init__(
        self,
        llm_client: LiteLLMClient,
        data_api_client: DataApiClient,
        model: str = "gemini/gemini-2.0-flash",
        temperature: float = 0.7,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.name = "rag"
        self.llm_client = llm_client
        self.data_api_client = data_api_client
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.facts_repo = ParentFactsRepository()
        self._kb_repo = KnowledgeBaseRepository()
        # Built lazily on the first chat against an agentic KB.
        self._agentic_svc = None

    async def __call__(self, state: ChatbotState) -> Dict[str, Any]:
        """Retrieve context from knowledge bases and generate response."""
        messages = state.get("messages", [])
        kb_ids = state.get("kb_ids", [])

        user_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        if not user_query:
            return {"messages": [AIMessage(content="I didn't receive a question. How can I help you?")]}

        context = await self._build_context(kb_ids, user_query)
        llm_messages = self._build_llm_messages(messages, context)

        try:
            response = await self.llm_client.chat(
                messages=llm_messages,
                model=self.model,
                temperature=self.temperature,
            )
            return {
                "messages": [AIMessage(content=response)],
                "context": context,
            }
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise ExternalServiceError("LLM", str(e))

    async def stream(self, state: ChatbotState) -> AsyncIterator[Dict[str, Any]]:
        """Stream response from LLM as tagged events.

        Yields:
          ``{"type": "agentic", "phase": ...}`` for each planner hop and
                                                  selector phase (only when
                                                  the KB has agentic_search
                                                  enabled).
          ``{"type": "thinking", "delta": ...}`` reasoning tokens.
          ``{"type": "content",  "delta": ...}`` answer tokens.

        Chat service forwards events to SSE verbatim — frontend renders
        agentic events as a collapsible per-iter progress block."""
        messages = state.get("messages", [])
        kb_ids = state.get("kb_ids", [])

        user_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        if not user_query:
            yield {"type": "content", "delta": "I didn't receive a question. How can I help you?"}
            return

        agentic_kb_id, kb_cfg = await self._resolve_agentic_kb(kb_ids)

        parents: List[Dict[str, Any]] = []
        graph_contexts: List[str] = []

        if agentic_kb_id:
            # In-process agentic loop — every planner hop streams out to UI.
            # Honour the KB's tunable knobs; fall back to the node default
            # ``TOP_K_CHILD`` and to AgenticSearchService's settings defaults.
            top_n = (kb_cfg or {}).get("top_n") or self.TOP_K_CHILD
            async for ev in self._agentic_service().astream(
                kb_id=agentic_kb_id,
                query_text=user_query,
                top_n=top_n,
                max_iter=(kb_cfg or {}).get("agentic_max_iter"),
                per_iter_k=(kb_cfg or {}).get("agentic_top_k_per_iter"),
            ):
                phase = ev.get("phase")
                if phase == "result":
                    parents = ev["payload"].get("results", [])
                    yield {
                        "type": "agentic",
                        "phase": "done",
                        "stop_reason": ev["payload"].get("agentic", {}).get("stop_reason"),
                        "raw_accumulated": ev["payload"].get("agentic", {}).get("raw_accumulated"),
                        "selected": len(parents),
                    }
                else:
                    # Forward iter_start / iter_done / stop / selecting as-is.
                    yield {"type": "agentic", **ev}
        else:
            parents, graph_contexts = await self._retrieve_parents(kb_ids, user_query)

        # ── Build LLM context (fact-extract cache reused across paths) ────
        if not parents and not graph_contexts:
            context = ""
        elif not settings.ENABLE_FACT_EXTRACTION:
            blocks = [p.get("parent_text") or p.get("content") or p.get("text", "") for p in parents]
            context = "\n\n---\n\n".join([b for b in blocks if b] + graph_contexts)
        else:
            facts = await self._get_or_extract_facts(parents)
            context = self._render_facts_context(parents, facts, graph_contexts)

        llm_messages = self._build_llm_messages(messages, context)

        try:
            async for event in self.llm_client.stream_chat(
                messages=llm_messages,
                model=self.model,
                temperature=self.temperature,
            ):
                yield event
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            raise ExternalServiceError("LLM", str(e))

    # ------------------------------------------------------------------
    # Agentic helpers
    # ------------------------------------------------------------------

    async def _resolve_agentic_kb(
        self, kb_ids: List[str],
    ) -> tuple[Optional[str], Optional[dict]]:
        """Return ``(kb_id, parser_config)`` for the first agentic-enabled KB.

        Multi-KB agents fall back to single-shot if no KB is agentic; if
        any IS agentic, only that one runs the loop (cross-KB agentic
        merge is a future improvement). Returns the config alongside the
        id so the caller can read per-KB tunable knobs (top_n,
        agentic_max_iter, agentic_top_k_per_iter)."""
        if not kb_ids:
            return None, None
        for kb_id in kb_ids:
            try:
                kb = await self._kb_repo.get(id=kb_id)
            except Exception:
                continue
            cfg = getattr(kb, "parser_config", None) or {}
            if isinstance(cfg, dict) and cfg.get("agentic_search"):
                return kb_id, cfg
        return None, None

    def _agentic_service(self):
        if self._agentic_svc is not None:
            return self._agentic_svc
        # Lazy import to avoid an import cycle (query_service imports
        # AgenticSearchService too).
        from app.application.services.agentic_search_service import AgenticSearchService
        from app.application.services.query_service import _llm_select
        self._agentic_svc = AgenticSearchService(
            embedding_client=EmbeddingClient(),
            llm_client=self.llm_client,
            repository_factory=db_session.get_session(),
            selector_fn=_llm_select,
        )
        return self._agentic_svc

    # Number of child chunks to retrieve per knowledge base. The query
    # endpoint already over-fetches and dedupes by parent_id, so this is
    # the post-dedupe target count per KB.
    TOP_K_CHILD = 10

    async def _build_context(self, kb_ids: List[str], query: str) -> str:
        """Build the LLM context.

        With fact extraction on (default): retrieve parents → load/extract
        their facts (cached per parent_id) → render facts grouped by source.
        With it off: fall back to concatenated raw parent text.
        """
        parents, graph_contexts = await self._retrieve_parents(kb_ids, query)
        if not parents and not graph_contexts:
            return ""

        if not settings.ENABLE_FACT_EXTRACTION:
            blocks = [p["parent_text"] for p in parents] + graph_contexts
            return "\n\n---\n\n".join(blocks)

        facts_per_parent = await self._get_or_extract_facts(parents)
        return self._render_facts_context(parents, facts_per_parent, graph_contexts)

    async def _retrieve_parents(self, kb_ids: List[str], query: str):
        """Return ``(parents, graph_contexts)``.

        ``parents`` is a score-ordered, parent_id-deduped list of dicts:
        ``{parent_id, parent_text, document_id, kb_id, doc_name, heading_path}``.
        ``graph_contexts`` holds pre-built context strings from graph/wiki KBs
        that have no chunk list (no fact extraction applied to those)."""
        if not kb_ids:
            return [], []

        per_kb_results = await self.data_api_client.batch_query_knowledge_bases(
            kb_ids=kb_ids,
            query_text=query,
            top_k=self.TOP_K_CHILD,
        )

        all_chunks: List[Dict[str, Any]] = []
        graph_contexts: List[str] = []

        for kb_result in per_kb_results:
            if not kb_result.get("success"):
                continue
            data = kb_result.get("data", {}) or {}
            chunks = data.get("results") or data.get("chunks")
            if isinstance(chunks, list):
                for c in chunks:
                    if isinstance(c, dict):
                        all_chunks.append(c)
            elif isinstance(data.get("context"), str) and data["context"].strip():
                graph_contexts.append(data["context"])

        all_chunks.sort(
            key=lambda c: float(c.get("similarity") or c.get("score") or 0.0),
            reverse=True,
        )

        seen: set = set()
        parents: List[Dict[str, Any]] = []
        for chunk in all_chunks:
            parent_text = chunk.get("parent_text") or chunk.get("content") or chunk.get("text", "")
            if not parent_text:
                continue
            pid = chunk.get("parent_id")
            key = ("pid", pid) if pid else ("ptxt", hash(parent_text))
            if key in seen:
                continue
            seen.add(key)
            parents.append({
                "parent_id": pid,
                "parent_text": parent_text,
                "document_id": chunk.get("document_id"),
                "kb_id": chunk.get("kb_id"),
                "doc_name": chunk.get("doc_name"),
                "heading_path": chunk.get("heading_path"),
            })

        return parents, graph_contexts

    async def _get_or_extract_facts(self, parents: List[Dict[str, Any]]) -> List[List[str]]:
        """Return facts per parent (aligned to ``parents`` order).

        Cached parents are loaded from ``parent_facts``; misses are extracted
        in parallel on the cheap model (no thinking) and persisted so each
        parent is only ever extracted once. On extraction failure the slot is
        left empty and the renderer falls back to raw parent text."""
        pids = [p["parent_id"] for p in parents if p.get("parent_id")]
        cached = await self.facts_repo.get_many(pids) if pids else {}

        results: List[List[str]] = [[] for _ in parents]
        to_extract: List[int] = []
        for i, p in enumerate(parents):
            pid = p.get("parent_id")
            if pid and pid in cached:
                results[i] = cached[pid]
            else:
                to_extract.append(i)

        if to_extract:
            extracted = await asyncio.gather(
                *(self._extract_one(parents[i]["parent_text"]) for i in to_extract),
                return_exceptions=True,
            )
            save_records = []
            for i, facts in zip(to_extract, extracted):
                if isinstance(facts, Exception) or not facts:
                    if isinstance(facts, Exception):
                        logger.warning("fact extraction failed for a parent: %s", facts)
                    continue
                results[i] = facts
                pid = parents[i].get("parent_id")
                if pid:
                    save_records.append({
                        "parent_id": pid,
                        "kb_id": parents[i].get("kb_id"),
                        "document_id": parents[i].get("document_id"),
                        "facts_json": json.dumps(facts, ensure_ascii=False),
                    })
            if save_records:
                await self.facts_repo.save_many(save_records)

        logger.info(
            "fact cache: %d hit, %d extracted (of %d parents)",
            len(parents) - len(to_extract), len(to_extract), len(parents),
        )
        return results

    async def _extract_one(self, parent_text: str) -> List[str]:
        """Extract atomic facts from one parent on the cheap model, no thinking."""
        resp = await self.llm_client.chat(
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": f"SECTION:\n{parent_text}\n\nExtract all facts. Output JSON only."},
            ],
            model=settings.FACT_EXTRACTION_MODEL,
            temperature=0.0,
            enable_thinking=False,
        )
        return self._parse_facts(resp)

    @staticmethod
    def _parse_facts(text: str) -> List[str]:
        if not text:
            return []
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not m:
                return []
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return []
        facts = data.get("facts") if isinstance(data, dict) else None
        if not isinstance(facts, list):
            return []
        return [str(f).strip() for f in facts if str(f).strip()]

    def _render_facts_context(
        self,
        parents: List[Dict[str, Any]],
        facts_per_parent: List[List[str]],
        graph_contexts: List[str],
    ) -> str:
        """Group facts under a per-source header. Parents whose extraction
        produced nothing fall back to their raw text so no context is lost."""
        parts: List[str] = []
        for p, facts in zip(parents, facts_per_parent):
            doc = p.get("doc_name") or "document"
            section = p.get("heading_path") or ""
            header = f"[Source: {doc}" + (f" | Section: {section}]" if section else "]")
            if facts:
                body = "\n".join(f"- {f}" for f in facts)
            else:
                body = p["parent_text"]
            parts.append(f"{header}\n{body}")
        parts.extend(graph_contexts)
        return "\n\n".join(parts)

    def _build_llm_messages(
        self,
        messages: List,
        context: str,
    ) -> List[Dict[str, str]]:
        """Build messages list for LLM API call."""
        llm_messages = []

        if context and settings.ENABLE_FACT_EXTRACTION:
            system_content = (
                ANSWER_FROM_FACTS_PROMPT.format(base=self.system_prompt)
                + "\n\nFACTS:\n" + context
            )
        elif context:
            system_content = f"{self.system_prompt}\n\n--- Retrieved Context ---\n{context}"
        else:
            system_content = self.system_prompt

        llm_messages.append({"role": "system", "content": system_content})

        for msg in messages[-10:]:
            if isinstance(msg, HumanMessage):
                llm_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                llm_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                pass

        return llm_messages
