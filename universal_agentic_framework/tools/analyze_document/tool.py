"""Document analysis tool — extract structured data from document images."""

import json
import re
import httpx
import structlog
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from universal_agentic_framework.tools.vision_utils import (
    _build_data_url,
    _build_request_payload,
    _load_vision_api_config,
    _resolve_local_image,
)

logger = structlog.get_logger()

_SYSTEM_PROMPT = (
    "You are a document data extractor. Analyze this document image and return a JSON object "
    "with the following fields: document_type (string), vendor (string), date (string), "
    "total (string), currency (string), "
    "line_items (array of objects with fields: description, quantity, unit_price, amount), "
    "notes (string). "
    "Use null for fields not present in the document. "
    "Return ONLY the JSON object, no prose, no markdown fences."
)

_DOCUMENT_TYPE_HINT_MAP = {
    "invoice": "This document appears to be an invoice.",
    "receipt": "This document appears to be a receipt.",
    "form": "This document appears to be a form.",
    "contract": "This document appears to be a contract.",
}


class AnalyzeDocumentInput(BaseModel):
    """Input for document data extraction."""

    image_source: str = Field(
        description="Image URL (http/https) or absolute local file path from an uploaded attachment.",
    )
    document_type: str = Field(
        default="auto",
        description="Optional hint about document type: invoice, receipt, form, contract, or auto.",
    )


class AnalyzeDocumentTool(BaseTool):
    """Extract structured data from document images using the configured vision model."""

    name: str = "analyze_document_tool"
    description: str = (
        "Extract structured data from a document image — invoices, receipts, forms, or contracts. "
        "Accepts an image URL (http/https) or a local file path from an uploaded attachment. "
        "Returns JSON with vendor, date, total, currency, and line items. "
        "Use when the user asks to scan, digitize, or extract data from a document image. "
        "Trigger phrases: scan invoice, extract from receipt, digitize document, "
        "read the bill, what does the invoice say, Rechnung scannen, Beleg auslesen, "
        "Formular digitalisieren, Quittung."
    )
    args_schema: type[BaseModel] = AnalyzeDocumentInput

    attachments_base_dir: str = "/tmp/steuermann-ai/chat-workspaces"
    max_image_bytes: int = 10 * 1024 * 1024

    def _build_user_prompt(self, document_type: str) -> str:
        hint = _DOCUMENT_TYPE_HINT_MAP.get(document_type.lower(), "")
        base = "Extract all structured data from this document image."
        return f"{hint} {base}".strip() if hint else base

    def _run(self, image_source: str, document_type: str = "auto", **kwargs) -> str:
        """Extract document data synchronously."""
        try:
            api_base, bare_model, temperature, api_key = _load_vision_api_config()
            if not api_base:
                return "Error: Vision model api_base is not configured."

            if image_source.startswith(("http://", "https://")):
                with httpx.Client(timeout=30.0) as client:
                    fetch_resp = client.get(image_source, follow_redirects=True)
                    fetch_resp.raise_for_status()
                    image_bytes = fetch_resp.content
                    mime_type = fetch_resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
            else:
                image_bytes, mime_type = _resolve_local_image(image_source, self.attachments_base_dir)

            if len(image_bytes) > self.max_image_bytes:
                return f"Error: Image is too large ({len(image_bytes):,} bytes; limit is {self.max_image_bytes:,} bytes)."

            data_url = _build_data_url(image_bytes, mime_type)
            payload = _build_request_payload(
                bare_model, temperature, self._build_user_prompt(document_type), data_url,
                max_tokens=2048, system_prompt=_SYSTEM_PROMPT,
            )

            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"{api_base}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key or 'no-key'}"},
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
                return self._clean_json_output(raw)

        except Exception as exc:
            logger.error("analyze_document_tool failed", error=str(exc), image_source=image_source)
            return f"Error analyzing document: {exc}"

    async def _arun(self, image_source: str, document_type: str = "auto", **kwargs) -> str:
        """Extract document data asynchronously."""
        try:
            api_base, bare_model, temperature, api_key = _load_vision_api_config()
            if not api_base:
                return "Error: Vision model api_base is not configured."

            if image_source.startswith(("http://", "https://")):
                async with httpx.AsyncClient(timeout=30.0) as client:
                    fetch_resp = await client.get(image_source, follow_redirects=True)
                    fetch_resp.raise_for_status()
                    image_bytes = fetch_resp.content
                    mime_type = fetch_resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
            else:
                image_bytes, mime_type = _resolve_local_image(image_source, self.attachments_base_dir)

            if len(image_bytes) > self.max_image_bytes:
                return f"Error: Image is too large ({len(image_bytes):,} bytes; limit is {self.max_image_bytes:,} bytes)."

            data_url = _build_data_url(image_bytes, mime_type)
            payload = _build_request_payload(
                bare_model, temperature, self._build_user_prompt(document_type), data_url,
                max_tokens=2048, system_prompt=_SYSTEM_PROMPT,
            )

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{api_base}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key or 'no-key'}"},
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
                return self._clean_json_output(raw)

        except Exception as exc:
            logger.error("analyze_document_tool failed", error=str(exc), image_source=image_source)
            return f"Error analyzing document: {exc}"

    @staticmethod
    def _clean_json_output(raw: str) -> str:
        """Strip markdown fences if the model wrapped the JSON anyway."""
        stripped = raw.strip()
        # Remove ```json ... ``` or ``` ... ``` fences
        fenced = re.match(r"```(?:json)?\s*([\s\S]+?)\s*```", stripped)
        if fenced:
            stripped = fenced.group(1).strip()
        # Validate it parses; return as-is (string) — let caller parse if needed
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            pass  # Return raw response; model may have added prose despite instructions
        return stripped
