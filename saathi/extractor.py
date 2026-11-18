"""Turn a notification into structured JSON.

LLMs sometimes return broken JSON or a wrong type ("vacancies": "12 (UR-5)").
Instead of failing, we send the validation error back and let the model fix
its own output, up to max_retries times.
"""

import json
import logging

from pydantic import ValidationError

from .llm import LLM
from .pdf_loader import Page
from .prompts import EXTRACT_RETRY_TEMPLATE, EXTRACT_SYSTEM, EXTRACT_TEMPLATE
from .schemas import NotificationDetails

log = logging.getLogger(__name__)

# Gemini Flash handles long input fine, but most notifications put everything
# important in the first pages and the rest is annexures / forms.
MAX_CHARS = 60_000


class ExtractionError(Exception):
    pass


def pages_to_text(pages: list[Page]) -> str:
    text = "\n\n".join(f"[p.{p.number}]\n{p.text}" for p in pages)
    return text[:MAX_CHARS]


def _strip_code_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def extract_details(pages: list[Page], llm: LLM, max_retries: int = 2) -> NotificationDetails:
    schema = json.dumps(NotificationDetails.model_json_schema(), indent=1)
    prompt = EXTRACT_TEMPLATE.format(schema=schema, text=pages_to_text(pages))
    raw = llm.generate(prompt, system=EXTRACT_SYSTEM, json_mode=True)

    for attempt in range(max_retries + 1):
        try:
            return NotificationDetails.model_validate_json(_strip_code_fence(raw))
        except (ValidationError, ValueError) as err:
            if attempt == max_retries:
                raise ExtractionError(f"Could not get valid JSON after {max_retries + 1} tries: {err}") from err
            log.info("extraction attempt %d failed, asking model to fix it", attempt + 1)
            retry_prompt = EXTRACT_RETRY_TEMPLATE.format(error=str(err)[:1500], previous=raw[:8000])
            raw = llm.generate(retry_prompt, system=EXTRACT_SYSTEM, json_mode=True)
