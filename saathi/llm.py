from typing import Protocol


class LLM(Protocol):
    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str: ...


class GeminiLLM:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.1):
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
        response = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        return (response.text or "").strip()


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
