import time
from typing import Protocol


class LLM(Protocol):
    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str: ...


class GeminiLLM:
    def __init__(self, api_key: str, model: str = "gemini-3.8-flash", temperature: float = 0.1):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing. Copy .env.example to .env and add your key.")
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        config = self._types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        response = with_retry(
            lambda: self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        )
        return (response.text or "").strip()


RETRYABLE = {429, 500, 502, 503, 504}


def with_retry(call, attempts: int = 4, base_delay: float = 2.0, sleep=time.sleep):
    """Retry rate-limit / overload errors with exponential backoff (2s, 4s, 8s).

    Gemini returns 503 "high demand" now and then on the free tier; one
    retry usually fixes it. Anything else (bad key, bad request) fails fast.
    """
    for attempt in range(attempts):
        try:
            return call()
        except Exception as err:  # google.genai.errors.APIError has .code
            code = getattr(err, "code", None)
            if code not in RETRYABLE or attempt == attempts - 1:
                raise
            sleep(base_delay * 2 ** attempt)


class FakeLLM:
    """Returns pre-set replies in order. Used in tests so they run offline."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise RuntimeError("FakeLLM ran out of replies")
        return self.replies.pop(0)
