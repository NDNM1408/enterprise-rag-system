"""Replacements for ragflow's ``rag.nlp`` helpers used inside vendored parsers.

We only need ``find_codec``; ``rag_tokenizer`` was solely used for chunk-size
estimation and is dropped because this service does not chunk.
"""
from __future__ import annotations

import chardet

# Mirror of the codec list in ragflow/rag/nlp/__init__.py — covers most legacy
# Asian + European encodings the original code attempts before falling back.
_FALLBACK_CODECS = (
    "utf-8", "gb18030", "gbk", "gb2312", "big5",
    "shift_jis", "euc-jp", "euc-kr",
    "iso-8859-1", "iso-8859-2", "iso-8859-15",
    "windows-1250", "windows-1251", "windows-1252", "windows-1253",
    "windows-1254", "windows-1255", "windows-1256", "windows-1257", "windows-1258",
    "latin-2",
)


def find_codec(blob: bytes) -> str:
    """Best-effort encoding detection for arbitrary text payloads."""
    detected = chardet.detect(blob[:1024])
    if detected.get("confidence", 0) > 0.5:
        enc = detected.get("encoding") or "utf-8"
        return "utf-8" if enc.lower() == "ascii" else enc
    for c in _FALLBACK_CODECS:
        try:
            blob[:1024].decode(c)
            return c
        except Exception:
            pass
        try:
            blob.decode(c)
            return c
        except Exception:
            pass
    return "utf-8"
