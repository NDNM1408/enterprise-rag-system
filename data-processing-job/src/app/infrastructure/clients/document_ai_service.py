import io
import httpx
from typing import Any
from app.configurations.configurations import settings

class DocumentAIService:
    def __init__(self):
        self.url = settings.DOCUMENT_AI_URL

    async def parse_document(self, content: bytes) -> dict[str, Any]:
        file_stream = io.BytesIO(content)

        files = {
            "file": ("document.pdf", file_stream, "application/pdf")
        }

        timeout = httpx.Timeout(300.0)  # Tăng timeout

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(self.url, files=files)
                response.raise_for_status()
                return {
                    "parsed_data": response.json()['content']
                }
            except httpx.HTTPStatusError as e:
                return {
                    "error": f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
                }
            except Exception as e:
                return {
                    "error": str(e)
                }
    # async def parse_document(sekf, document):
    #     html_str = open("/home/minh/My Project/data-hub/data-processing-job/src/app/infrastructure/clients/057-QĐ đăng ký điện ngoài giờ (H).pdf-1.html", "r").read()
    #     return {
    #         "parsed_data": html_str
    #     }
