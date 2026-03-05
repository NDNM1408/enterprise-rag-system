"""
LLM-based entity/relation extraction and parsing.

Ported from lightrag/operate.py — extraction + parsing logic only.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass

from app.infrastructure.graph.llm_client import LLMClient
from app.infrastructure.graph.prompts import (
    COMPLETION_DELIMITER,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_LANGUAGE,
    ENTITY_CONTINUE_EXTRACTION_PROMPT,
    ENTITY_EXTRACTION_EXAMPLES,
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_USER_PROMPT,
    TUPLE_DELIMITER,
)

logger = logging.getLogger(__name__)

# Maximum length for entity names (from LightRAG DEFAULT_ENTITY_NAME_MAX_LENGTH)
_ENTITY_NAME_MAX_LENGTH = 256


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExtractedEntity:
    entity_name: str
    entity_type: str
    description: str
    source_id: str
    file_path: str


@dataclass
class ExtractedRelation:
    src_id: str
    tgt_id: str
    weight: float
    description: str
    keywords: str
    source_id: str
    file_path: str


# ---------------------------------------------------------------------------
# Text sanitization (ported from lightrag/utils.py)
# ---------------------------------------------------------------------------

def _sanitize_text(text: str, remove_inner_quotes: bool = False) -> str:
    """Sanitize and normalize extracted text.

    Ported from lightrag.utils.sanitize_and_normalize_extracted_text.
    """
    if not text:
        return ""

    name = text

    # Clean HTML paragraph/line-break tags
    name = re.sub(r"</p\s*>|<p\s*>|<p/>", "", name, flags=re.IGNORECASE)
    name = re.sub(r"</br\s*>|<br\s*>|<br/>", "", name, flags=re.IGNORECASE)

    # Chinese full-width letters → half-width
    name = name.translate(
        str.maketrans(
            "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        )
    )
    # Chinese full-width numbers → half-width
    name = name.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

    # Chinese full-width symbols → half-width
    name = name.replace("－", "-").replace("＋", "+").replace("／", "/").replace("＊", "*")
    name = name.replace("（", "(").replace("）", ")")
    name = name.replace("—", "-")
    name = name.replace("　", " ")  # full-width space

    # Remove spaces between Chinese characters
    name = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", name)
    # Remove spaces between Chinese and English/numbers
    name = re.sub(
        r"(?<=[\u4e00-\u9fa5])\s+(?=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])", "", name
    )
    name = re.sub(
        r"(?<=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])\s+(?=[\u4e00-\u9fa5])", "", name
    )

    # Remove outer quotes
    if len(name) >= 2:
        for open_q, close_q in [('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019"), ("\u300a", "\u300b")]:
            if name.startswith(open_q) and name.endswith(close_q):
                inner = name[1:-1]
                if open_q not in inner and close_q not in inner:
                    name = inner
                    break

    if remove_inner_quotes:
        name = name.replace("\u201c", "").replace("\u201d", "").replace("\u2018", "").replace("\u2019", "")
        name = re.sub(r"['\"]+(?=[\u4e00-\u9fa5])", "", name)
        name = re.sub(r"(?<=[\u4e00-\u9fa5])['\"]+", "", name)
        name = name.replace("\u00a0", " ")
        name = re.sub(r"(?<=[^\d])\u202F", " ", name)

    name = name.strip()

    # Filter pure numeric < 3 chars
    if len(name) < 3 and re.match(r"^[0-9]+$", name):
        return ""
    # Filter short dot-digit strings
    if len(name) < 6 and all(c.isdigit() or c == "." for c in name) and "." in name:
        return ""

    return name


# ---------------------------------------------------------------------------
# Delimiter corruption fixer (ported from lightrag/utils.py)
# ---------------------------------------------------------------------------

def _fix_delimiter_corruption(record: str, delimiter_core: str, tuple_delimiter: str) -> str:
    """Fix various forms of tuple_delimiter corruption from LLM output."""
    if not record or not delimiter_core or not tuple_delimiter:
        return record

    esc = re.escape(delimiter_core)

    # <|##|> → <|#|>
    record = re.sub(rf"<\|{esc}\|*?{esc}\|>", tuple_delimiter, record)
    # <|\#|> → <|#|>
    record = re.sub(rf"<\|\\{esc}\|>", tuple_delimiter, record)
    # <|> or <||> → <|#|>
    record = re.sub(r"<\|+>", tuple_delimiter, record)
    # <X|#|Y> → <|#|>
    record = re.sub(rf"<.?\|{esc}\|.?>", tuple_delimiter, record)
    # <#> or <#|> or <|#> → <|#|>
    record = re.sub(rf"<\|?{esc}\|?>", tuple_delimiter, record)
    # <X#|> or <|#X> → <|#|>
    record = re.sub(rf"<[^|]{esc}\|>|<\|{esc}[^|]>", tuple_delimiter, record)
    # <|#| or <|#|| → <|#|>
    record = re.sub(rf"<\|{esc}\|+(?!>)", tuple_delimiter, record)
    # <|#: → <|#|>
    record = re.sub(rf"<\|{esc}:(?!>)", tuple_delimiter, record)
    # <||#> → <|#|>
    record = re.sub(rf"<\|+{esc}>", tuple_delimiter, record)
    # <||> → <|#|>
    record = re.sub(r"<\|\|(?!>)", tuple_delimiter, record)
    # |#|> → <|#|>
    record = re.sub(rf"(?<!<)\|{esc}\|>", tuple_delimiter, record)

    return record


def _split_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers."""
    if not markers:
        return [content]
    content = content if content is not None else ""
    results = re.split("|".join(re.escape(m) for m in markers), content)
    return [r.strip() for r in results if r.strip()]


def _is_float(s: str) -> bool:
    """Check if a string is a valid float."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Single-record parsers
# ---------------------------------------------------------------------------

def _parse_entity(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str,
) -> ExtractedEntity | None:
    """Parse and validate a single entity tuple (4 fields)."""
    if len(record_attributes) != 4 or "entity" not in record_attributes[0]:
        return None

    entity_name = _sanitize_text(record_attributes[1], remove_inner_quotes=True)
    if not entity_name or not entity_name.strip():
        return None

    entity_type = _sanitize_text(record_attributes[2], remove_inner_quotes=True)
    if not entity_type.strip() or any(
        c in entity_type for c in ["'", "(", ")", "<", ">", "|", "/", "\\"]
    ):
        return None
    entity_type = entity_type.replace(" ", "").lower()

    description = _sanitize_text(record_attributes[3])
    if not description.strip():
        return None

    # Truncate long names
    if len(entity_name) > _ENTITY_NAME_MAX_LENGTH:
        logger.warning(
            "%s: entity name len %d > %d, truncating '%s...'",
            chunk_key, len(entity_name), _ENTITY_NAME_MAX_LENGTH, entity_name[:20],
        )
        entity_name = entity_name[:_ENTITY_NAME_MAX_LENGTH]

    return ExtractedEntity(
        entity_name=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=chunk_key,
        file_path=file_path,
    )


def _parse_relation(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str,
) -> ExtractedRelation | None:
    """Parse and validate a single relation tuple (5 fields)."""
    if len(record_attributes) != 5 or "relation" not in record_attributes[0]:
        return None

    source = _sanitize_text(record_attributes[1], remove_inner_quotes=True)
    target = _sanitize_text(record_attributes[2], remove_inner_quotes=True)

    if not source or not target or source == target:
        return None

    keywords = _sanitize_text(record_attributes[3], remove_inner_quotes=True)
    keywords = keywords.replace("\uff0c", ",")  # Chinese comma → English

    description = _sanitize_text(record_attributes[4])

    raw_weight = record_attributes[-1].strip('"').strip("'")
    weight = float(raw_weight) if _is_float(raw_weight) else 1.0

    # Truncate long names
    for name_val in (source, target):
        if len(name_val) > _ENTITY_NAME_MAX_LENGTH:
            logger.warning(
                "%s: relation entity name len %d > %d, truncating",
                chunk_key, len(name_val), _ENTITY_NAME_MAX_LENGTH,
            )
    source = source[:_ENTITY_NAME_MAX_LENGTH]
    target = target[:_ENTITY_NAME_MAX_LENGTH]

    return ExtractedRelation(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=description,
        keywords=keywords,
        source_id=chunk_key,
        file_path=file_path,
    )


# ---------------------------------------------------------------------------
# Full extraction result parser
# ---------------------------------------------------------------------------

def _parse_extraction_result(
    result: str,
    chunk_key: str,
    file_path: str,
    tuple_delimiter: str = TUPLE_DELIMITER,
    completion_delimiter: str = COMPLETION_DELIMITER,
) -> tuple[dict[str, list[ExtractedEntity]], dict[tuple[str, str], list[ExtractedRelation]]]:
    """Parse LLM extraction output into entity and relation dicts."""
    entities: dict[str, list[ExtractedEntity]] = defaultdict(list)
    relations: dict[tuple[str, str], list[ExtractedRelation]] = defaultdict(list)

    if completion_delimiter not in result:
        logger.warning("%s: completion delimiter not found in extraction result", chunk_key)

    # Split by newlines and completion delimiter
    records = _split_by_multi_markers(
        result, ["\n", completion_delimiter, completion_delimiter.lower()]
    )

    # Fix records where LLM used tuple_delimiter as record separator
    fixed_records: list[str] = []
    for record in records:
        record = record.strip()
        if not record:
            continue
        # Split if tuple_delimiter was used as record separator
        entity_records = _split_by_multi_markers(
            record, [f"{tuple_delimiter}entity{tuple_delimiter}"]
        )
        for er in entity_records:
            if not er.startswith("entity") and not er.startswith("relation"):
                er = f"entity<|{er}"
            relation_records = _split_by_multi_markers(
                er,
                [
                    f"{tuple_delimiter}relationship{tuple_delimiter}",
                    f"{tuple_delimiter}relation{tuple_delimiter}",
                ],
            )
            for rr in relation_records:
                if not rr.startswith("entity") and not rr.startswith("relation"):
                    rr = f"relation{tuple_delimiter}{rr}"
                fixed_records.append(rr)

    # Parse each record
    delimiter_core = tuple_delimiter[2:-2]  # "#" from "<|#|>"
    for record in fixed_records:
        record = record.strip()
        if not record:
            continue

        # Fix delimiter corruption
        record = _fix_delimiter_corruption(record, delimiter_core, tuple_delimiter)
        if delimiter_core != delimiter_core.lower():
            record = _fix_delimiter_corruption(record, delimiter_core.lower(), tuple_delimiter)

        attrs = _split_by_multi_markers(record, [tuple_delimiter])

        entity = _parse_entity(attrs, chunk_key, file_path)
        if entity is not None:
            entities[entity.entity_name].append(entity)
            continue

        relation = _parse_relation(attrs, chunk_key, file_path)
        if relation is not None:
            relations[(relation.src_id, relation.tgt_id)].append(relation)

    return dict(entities), dict(relations)


# ---------------------------------------------------------------------------
# EntityExtractor
# ---------------------------------------------------------------------------

class EntityExtractor:
    """Extract entities and relations from text using an LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        entity_types: list[str] | None = None,
        language: str = DEFAULT_LANGUAGE,
        max_gleaning: int = 1,
    ) -> None:
        self.llm_client = llm_client
        self.entity_types = entity_types or DEFAULT_ENTITY_TYPES
        self.language = language
        self.max_gleaning = max_gleaning

    async def extract(
        self,
        text: str,
        chunk_key: str,
        file_path: str,
    ) -> tuple[dict[str, list[ExtractedEntity]], dict[tuple[str, str], list[ExtractedRelation]]]:
        """Extract entities and relations from text.

        Returns:
            (entities_by_name, relations_by_pair) where each value is a dict
            mapping to lists of extracted items.
        """
        entity_types_str = ", ".join(self.entity_types)
        examples_str = "\n".join(ENTITY_EXTRACTION_EXAMPLES)

        system_prompt = ENTITY_EXTRACTION_SYSTEM_PROMPT.format(
            entity_types=entity_types_str,
            tuple_delimiter=TUPLE_DELIMITER,
            completion_delimiter=COMPLETION_DELIMITER,
            language=self.language,
            examples=examples_str,
        )

        user_prompt = ENTITY_EXTRACTION_USER_PROMPT.format(
            entity_types=entity_types_str,
            input_text=text,
            tuple_delimiter=TUPLE_DELIMITER,
            completion_delimiter=COMPLETION_DELIMITER,
            language=self.language,
        )

        # Initial extraction
        result = await self.llm_client.complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        all_entities: dict[str, list[ExtractedEntity]] = defaultdict(list)
        all_relations: dict[tuple[str, str], list[ExtractedRelation]] = defaultdict(list)

        entities, relations = _parse_extraction_result(result, chunk_key, file_path)
        for k, v in entities.items():
            all_entities[k].extend(v)
        for k, v in relations.items():
            all_relations[k].extend(v)

        # Gleaning: ask for missed entities/relations
        history: list[dict] = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": result},
        ]

        for i in range(self.max_gleaning):
            continue_prompt = ENTITY_CONTINUE_EXTRACTION_PROMPT.format(
                tuple_delimiter=TUPLE_DELIMITER,
                completion_delimiter=COMPLETION_DELIMITER,
                language=self.language,
            )

            gleaning_result = await self.llm_client.complete(
                prompt=continue_prompt,
                system_prompt=system_prompt,
                history_messages=history,
            )

            glean_entities, glean_relations = _parse_extraction_result(
                gleaning_result, chunk_key, file_path
            )
            for k, v in glean_entities.items():
                all_entities[k].extend(v)
            for k, v in glean_relations.items():
                all_relations[k].extend(v)

            history.extend([
                {"role": "user", "content": continue_prompt},
                {"role": "assistant", "content": gleaning_result},
            ])

        logger.info(
            "%s: extracted %d entities, %d relations",
            chunk_key, len(all_entities), len(all_relations),
        )

        return dict(all_entities), dict(all_relations)
