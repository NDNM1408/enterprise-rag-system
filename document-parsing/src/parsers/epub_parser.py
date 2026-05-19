"""EPUB → markdown.

Walks spine reading order and concatenates per-document markdown rendered by
``HtmlParser``. Same OPF/container logic as ragflow's ``RAGFlowEpubParser``.
"""
from __future__ import annotations

import logging
import warnings
import zipfile
from io import BytesIO
from xml.etree import ElementTree

from core.base import BaseParser, ParseResult
from parsers.html_parser import HtmlParser

log = logging.getLogger(__name__)

_OPF_NS = "http://www.idpf.org/2007/opf"
_CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
_XHTML_MEDIA_TYPES = {"application/xhtml+xml", "text/html", "text/xml"}


class EpubParser(BaseParser):
    name = "epub"
    extensions = ("epub",)

    def __init__(self) -> None:
        self._html = HtmlParser()

    def parse(self, payload: bytes, filename: str) -> ParseResult:
        if not payload:
            raise ValueError("Empty EPUB payload")

        out: list[str] = []
        sections = 0
        with zipfile.ZipFile(BytesIO(payload)) as zf:
            items = self._spine_items(zf)
            for item_path in items:
                try:
                    html_bytes = zf.read(item_path)
                except KeyError:
                    continue
                if not html_bytes:
                    continue
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    try:
                        result = self._html.parse(html_bytes, item_path)
                    except Exception as e:
                        log.warning("Skip EPUB section %s: %s", item_path, e)
                        continue
                if result.markdown.strip():
                    out.append(result.markdown.strip())
                    sections += 1

        markdown = "\n\n---\n\n".join(out)
        return ParseResult(
            markdown=markdown,
            parser=self.name,
            page_count=sections,
            metadata={"extension": "epub", "section_count": sections},
        )

    @staticmethod
    def _spine_items(zf: zipfile.ZipFile) -> list[str]:
        try:
            container_xml = zf.read("META-INF/container.xml")
        except KeyError:
            return EpubParser._fallback_xhtml_order(zf)
        try:
            container_root = ElementTree.fromstring(container_xml)
        except ElementTree.ParseError:
            log.warning("Bad container.xml; falling back to xhtml order.")
            return EpubParser._fallback_xhtml_order(zf)

        rootfile_el = container_root.find(f".//{{{_CONTAINER_NS}}}rootfile")
        if rootfile_el is None:
            return EpubParser._fallback_xhtml_order(zf)
        opf_path = rootfile_el.get("full-path", "")
        if not opf_path:
            return EpubParser._fallback_xhtml_order(zf)
        opf_dir = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""

        try:
            opf_xml = zf.read(opf_path)
        except KeyError:
            return EpubParser._fallback_xhtml_order(zf)
        try:
            opf_root = ElementTree.fromstring(opf_xml)
        except ElementTree.ParseError:
            return EpubParser._fallback_xhtml_order(zf)

        manifest: dict[str, tuple[str, str]] = {}
        for item in opf_root.findall(f".//{{{_OPF_NS}}}item"):
            item_id = item.get("id", "")
            href = item.get("href", "")
            media_type = item.get("media-type", "")
            if item_id and href:
                manifest[item_id] = (href, media_type)

        spine: list[str] = []
        for itemref in opf_root.findall(f".//{{{_OPF_NS}}}itemref"):
            idref = itemref.get("idref", "")
            if idref not in manifest:
                continue
            href, media_type = manifest[idref]
            if media_type not in _XHTML_MEDIA_TYPES:
                continue
            spine.append(opf_dir + href)
        return spine if spine else EpubParser._fallback_xhtml_order(zf)

    @staticmethod
    def _fallback_xhtml_order(zf: zipfile.ZipFile) -> list[str]:
        return sorted(
            n for n in zf.namelist()
            if n.lower().endswith((".xhtml", ".html", ".htm")) and not n.startswith("META-INF/")
        )
