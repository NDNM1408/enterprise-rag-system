import io
import logging

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {"txt", "html", "htm", "md", "docx", "pdf"}


class UnsupportedFileTypeError(Exception):
    """Raised when the document format is not supported."""


class DocumentParser:
    """Document parser supporting txt, html, htm, md, docx, and pdf files.

    - TXT / MD   : decoded as plain UTF-8 text
    - HTML / HTM : plain text extracted with BeautifulSoup
    - DOCX       : plain text extracted with python-docx
    - PDF        : plain text extracted with pypdf
    - Other      : raises UnsupportedFileTypeError
    """

    def parse(self, content: bytes, file_name: str) -> str:
        """Parse raw document bytes into plain text.

        Args:
            content: Raw bytes of the document.
            file_name: Original filename (used to detect format).

        Returns:
            Extracted plain-text string.

        Raises:
            UnsupportedFileTypeError: If the file extension is not supported.
        """
        ext = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""

        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"File type '.{ext}' is not supported. "
                f"Currently supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
            )

        if ext in ("html", "htm"):
            return self._parse_html(content)
        elif ext == "docx":
            return self._parse_docx(content)
        elif ext == "pdf":
            return self._parse_pdf(content)
        else:  # txt, md
            return content.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Format-specific parsers
    # ------------------------------------------------------------------

    def _parse_html(self, content: bytes) -> str:
        soup = BeautifulSoup(content, "lxml")
        return soup.get_text(separator="\n", strip=True)

    def _parse_docx(self, content: bytes) -> str:
        from docx import Document  # python-docx

        doc = Document(io.BytesIO(content))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(paragraphs)

    def _parse_pdf(self, content: bytes) -> str:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)
