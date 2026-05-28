"""Agentic basin-pivot search (hier_v2 Phase 2) — sync + streaming variants.

Each iteration the planner LLM fans out 1-3 sub-queries on DIFFERENT pivot
axes (KIND of fact / target section-type / current basin / uncovered axes
— see the prompt below). Sub-queries are embedded and vector-searched in
parallel; results are merged with two-pass dedup (chunk_id + parent/table
id), accumulated across iters in first-seen order. The loop stops when
the planner returns ``stop`` (a chunk holds a sentence-level answer, or
exhaustion: ≥3 lexical surfaces explored and the last 2 iters added 0 new
chunks). The accumulated chunks then pass through the same LLM selector
the single-shot path uses, so /query's response shape stays identical.

Iter-1 special: the original query MUST be ``sub_queries[0]``; the planner
may add up to two alternative axes.

Dense retrieval clusters by surface topic — exploring multiple axes per
turn covers more semantic basins than serial rephrasing within one basin.
This is the whole reason agentic exists; without basin-pivots the loop
just rephrases inside one cluster and gains nothing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Tuple

from app.configurations.configurations import settings
from app.infrastructure.clients.embedding_client import EmbeddingClient
from app.infrastructure.clients.litellm_client import LiteLLMClient
from app.infrastructure.repositories.document_embeddings_repository import (
    DocumentEmbeddingsRepository,
)

logger = logging.getLogger(__name__)


# ── Planner prompts (basin-pivot — verbatim shape from research_rag) ────────

PLANNER_SYSTEM = (
    "You are a retrieval planning agent. Each turn, fan out 1-3 sub-queries "
    "on DIFFERENT pivot axes, then later decide stop or search again. Dense "
    "retrieval clusters by surface topic, so exploring multiple axes per "
    "turn covers more semantic basins than serial rephrasing within one "
    "basin. You do NOT generate answers. Output STRICT JSON only."
)

PLANNER_USER_TMPL = """Original query: {query}
Iteration: {it}/{max_iter}

Previous sub-queries (DO NOT repeat axes already covered):
{prev_block}

Retrieved so far (top 3 per search, deduplicated):
{obs_block}

REASON BRIEFLY in `thought` (one line each):
A. KIND of fact the query needs (entity-attribute, ownership/assignment,
   definition, process/flow, comparison, count, policy/rule, history,
   constraint, source-of-truth, ...).
B. Section-type hypothesis — which section-type most likely CONTAINS this
   fact? (feature-spec, requirement, glossary, roster/directory, changelog,
   policy, API-ref, governance, onboarding, architecture, decision-record,
   runbook, ...).
C. Current basin — which section-type are retrieved chunks dominated by?
   (write "none" at iter 1).
D. Uncovered axes you will pivot to THIS turn (1-3).

DECIDE: stop or fan-out search.

STOP — return action="stop" ONLY when EITHER:
(a) A retrieved chunk contains a SENTENCE-LEVEL STATEMENT directly answering
    the original query — you could extract the answer near-verbatim from it.
    Mere mention of entities/keywords != answer. Many chunks != answer.
(b) You have explored >=3 distinct lexical surfaces AND the last 2 iterations
    added 0 new unique chunks (exhaustion across axes).

SEARCH — output 1-3 sub_queries on DISTINCT axes. EACH sub_query MUST:
- Use vocabulary characteristic of its target section-type (B) that is
  UNLIKELY to appear in the basin you're stuck in (C).
- AVOID generic words ("team", "owner", "responsible", "development",
  "platform", "feature") that score uniformly across section-types.
- Differ from previous sub-queries on the lexical-surface axis.

ITER-1 SPECIAL: at iteration 1, the original query MUST be sub_query #1.
You may add up to 2 more on alternative axes.

Output STRICT JSON only:
{{
  "thought": "A: ...; B: target=<types>; C: stuck-in=<type or none>; D: axes=[...]",
  "action": "search" | "stop",
  "sub_queries": ["<sq1>", "<sq2>", "<sq3>"],
  "axes": ["<axis name 1>", "<axis name 2>", "<axis name 3>"],
  "reason": "<which stop condition - only if action=stop>"
}}"""


# ── Helpers ────────────────────────────────────────────────────────────────

def _strip_fence(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    return s


def _chunk_dedup_keys(row: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Same dedupe shape the single-shot path uses (parent_id ∪ table_id ∪
    parent_text-hash fallback)."""
    keys: List[Tuple[str, Any]] = []
    pid = row.get("parent_id")
    tid = row.get("table_id")
    if pid:
        keys.append(("pid", pid))
    if tid:
        keys.append(("tid", tid))
    if not keys:
        pt = row.get("parent_text") or row.get("content") or row.get("text") or ""
        if pt:
            keys.append(("ptxt", hash(pt)))
    return keys


def _format_observations(accumulated: Dict[str, Dict[str, Any]], limit: int = 10) -> str:
    """Top-N preview block for the planner — shows what's been retrieved so
    far so the planner can name the current basin (C) and choose pivots."""
    if not accumulated:
        return "(none yet)"
    rows = sorted(accumulated.values(), key=lambda r: -float(r.get("similarity") or 0))[:limit]
    lines: List[str] = []
    for r in rows:
        doc = r.get("doc_name") or "?"
        sec = (r.get("heading_path") or "")[:50]
        preview = (r.get("text") or r.get("parent_text") or "").strip()
        preview = " ".join(preview.split())[:180]
        lines.append(f"- ({doc} · {sec}) {preview}")
    return "\n".join(lines)


def _format_prev_sub_queries(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "(none — first turn)"
    lines: List[str] = []
    for h in history:
        sqs = ", ".join(f'"{s}"' for s in h.get("sub_queries", []))
        axes = h.get("axes") or []
        axes_str = f"  axes={axes}" if axes else ""
        lines.append(f"iter {h['iter']}: {sqs}{axes_str}")
    return "\n".join(lines)


# ── Service ────────────────────────────────────────────────────────────────

class AgenticSearchService:
    """Basin-pivot agentic loop over the pgvector ``chunk`` table.

    Reuses the existing single-shot vector repository + LLM selector — the
    only new piece is the iterative planner + per-iter merge."""

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        llm_client: LiteLLMClient,
        repository_factory,                # callable() → DocumentEmbeddingsRepository (sessioned)
        selector_fn,                       # async (llm, query, candidates, top_n) → (selected, fallback)
    ):
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.repo_factory = repository_factory
        self.selector_fn = selector_fn

    async def run(
        self,
        *,
        kb_id: str,
        query_text: str,
        top_n: int,
    ) -> Dict[str, Any]:
        """Non-streaming run — drain ``astream`` and return the final result."""
        final: Dict[str, Any] = {}
        async for ev in self.astream(kb_id=kb_id, query_text=query_text, top_n=top_n):
            if ev.get("phase") == "result":
                final = ev["payload"]
        return final

    async def astream(
        self,
        *,
        kb_id: str,
        query_text: str,
        top_n: int,
    ):
        """Yield per-iter events so the chat SSE can show progress live.

        Event shape (all keyed on ``phase``):

          {phase: "iter_start", iter, sub_queries, axes, thought}
          {phase: "iter_done",  iter, new_count, total_accumulated, top_preview}
          {phase: "stop",       iter, reason}
          {phase: "selecting",  candidates}
          {phase: "result",     payload: {results, agentic: {...}}}

        ``top_preview`` is a tiny [{doc_name, heading_path, chunk_type, similarity}]
        slice (≤3) the UI uses to show what surfaced this iter without dumping
        whole chunks.
        """
        max_iter = settings.AGENTIC_MAX_ITER
        per_iter_k = settings.AGENTIC_TOP_K_PER_ITER

        history: List[Dict[str, Any]] = []
        accumulated: Dict[str, Dict[str, Any]] = {}
        zero_new_streak = 0
        stop_reason: str | None = None

        for it in range(1, max_iter + 1):
            decision = await self._plan(query_text, history, accumulated, it, max_iter)

            if it >= 2 and decision.get("action") == "stop":
                stop_reason = decision.get("reason") or "planner_stop"
                history.append({
                    "iter": it, "thought": decision.get("thought", ""),
                    "action": "stop", "sub_queries": [], "axes": [],
                    "reason": stop_reason,
                })
                yield {"phase": "stop", "iter": it, "reason": stop_reason,
                       "thought": decision.get("thought", "")}
                break

            sub_queries = self._normalize_sub_queries(decision, query_text, it)
            axes = decision.get("axes") or []
            if not sub_queries:
                sub_queries = [query_text]

            yield {
                "phase": "iter_start",
                "iter": it,
                "sub_queries": sub_queries,
                "axes": axes,
                "thought": decision.get("thought", ""),
            }

            per_sq = await asyncio.gather(
                *(self._search_one(kb_id, sq, per_iter_k) for sq in sub_queries),
                return_exceptions=True,
            )
            merged = self._merge_iter([r for r in per_sq if not isinstance(r, Exception)])

            new_count = 0
            for h in merged:
                cid = h.get("chunk_id")
                if cid and cid not in accumulated:
                    accumulated[cid] = h
                    new_count += 1

            top_preview = [
                {
                    "doc_name": h.get("doc_name"),
                    "heading_path": h.get("heading_path"),
                    "chunk_type": h.get("chunk_type"),
                    "similarity": round(float(h.get("similarity") or 0), 3),
                }
                for h in merged[:3]
            ]

            history.append({
                "iter": it,
                "thought": decision.get("thought", ""),
                "action": "search",
                "sub_queries": sub_queries,
                "axes": axes,
                "merged_count": len(merged),
                "new_count": new_count,
            })

            yield {
                "phase": "iter_done",
                "iter": it,
                "new_count": new_count,
                "total_accumulated": len(accumulated),
                "top_preview": top_preview,
            }

            if new_count == 0:
                zero_new_streak += 1
            else:
                zero_new_streak = 0
            distinct_surfaces = sum(len(h.get("sub_queries", [])) for h in history)
            if zero_new_streak >= 2 and distinct_surfaces >= 3:
                stop_reason = "exhaustion_no_new"
                yield {"phase": "stop", "iter": it, "reason": stop_reason,
                       "thought": "code-level exhaustion guard"}
                break

        if not stop_reason:
            stop_reason = "max_iters" if len(history) >= max_iter else "complete"

        candidates = list(accumulated.values())
        candidates.sort(key=lambda r: -float(r.get("similarity") or 0))

        yield {"phase": "selecting", "candidates": len(candidates)}

        selected, fallback = await self.selector_fn(
            self.llm_client, query_text, candidates, top_n,
        )

        yield {
            "phase": "result",
            "payload": {
                "results": selected,
                "agentic": {
                    "iterations": history,
                    "stop_reason": stop_reason,
                    "raw_accumulated": len(accumulated),
                    "selector_fallback": fallback,
                    "selector_model": settings.SELECTOR_MODEL,
                },
            },
        }

    # ------------------------------------------------------------------
    # Planner LLM call.
    # ------------------------------------------------------------------

    async def _plan(
        self,
        query: str,
        history: List[Dict[str, Any]],
        accumulated: Dict[str, Dict[str, Any]],
        it: int,
        max_iter: int,
    ) -> Dict[str, Any]:
        user = PLANNER_USER_TMPL.format(
            query=query,
            it=it,
            max_iter=max_iter,
            prev_block=_format_prev_sub_queries(history),
            obs_block=_format_observations(accumulated),
        )
        try:
            raw = await self.llm_client.chat(
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM},
                    {"role": "user",   "content": user},
                ],
                model=settings.AGENTIC_PLANNER_MODEL,
                temperature=0.0,
                enable_thinking=False,
            )
            return json.loads(_strip_fence(raw))
        except Exception as exc:
            logger.warning("planner failed at iter %d (%s); falling back to single-query search", it, exc)
            return {
                "action": "search",
                "sub_queries": [query],
                "axes": [],
                "thought": f"fallback: planner error ({exc})",
            }

    # ------------------------------------------------------------------
    # Normalise sub-queries (iter-1 must include the original; cap 3).
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_sub_queries(decision: Dict[str, Any], query: str, it: int) -> List[str]:
        sqs = [str(s).strip() for s in (decision.get("sub_queries") or []) if str(s).strip()]
        # Dedupe (case-insensitive) preserving order.
        seen, out = set(), []
        for s in sqs:
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
        if it == 1:
            if not out or out[0].lower() != query.strip().lower():
                out = [query] + [s for s in out if s.lower() != query.strip().lower()]
        return out[:3]

    # ------------------------------------------------------------------
    # Embed + search one sub-query.
    # ------------------------------------------------------------------

    async def _search_one(self, kb_id: str, sub_query: str, top_k: int) -> List[Dict[str, Any]]:
        try:
            vec = await self.embedding_client.get_embedding(sub_query)
        except Exception as exc:
            logger.warning("embedding failed for sub_query=%r: %s", sub_query, exc)
            return []
        async with self.repo_factory() as repo_session:
            repo = DocumentEmbeddingsRepository(repo_session)
            try:
                return await repo.query_by_vector(
                    kb_id=kb_id, query_embedding=vec, top_k=top_k,
                )
            except Exception as exc:
                logger.warning("vector search failed for sub_query=%r: %s", sub_query, exc)
                return []

    # ------------------------------------------------------------------
    # Per-iter merge: dedup chunk_id (max score), then collapse on
    # parent_id / table_id so sibling segments don't crowd out other basins.
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_iter(per_sq_hits: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        by_chunk: Dict[str, Dict[str, Any]] = {}
        for hits in per_sq_hits:
            for h in hits:
                cid = h.get("chunk_id")
                if not cid:
                    continue
                if cid not in by_chunk or float(h.get("similarity") or 0) > float(by_chunk[cid].get("similarity") or 0):
                    by_chunk[cid] = h
        # Parent/table collapse.
        by_parent: Dict[Any, Dict[str, Any]] = {}
        for h in by_chunk.values():
            keys = _chunk_dedup_keys(h) or [("chunk_id", h["chunk_id"])]
            primary = keys[0]
            if primary not in by_parent or float(h.get("similarity") or 0) > float(by_parent[primary].get("similarity") or 0):
                by_parent[primary] = h
        return sorted(by_parent.values(), key=lambda x: -float(x.get("similarity") or 0))
