from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class BaseSummarizer:
    def summarize(self, text: str, *, max_words: int) -> str:
        raise NotImplementedError


class NoOpSummarizer(BaseSummarizer):
    def summarize(self, text: str, *, max_words: int) -> str:  # noqa: D401
        """Return the text unchanged when no LLM is configured."""

        return text


class LLMSummarizer(BaseSummarizer):
    """Uses an OpenAI-compatible model to condense text."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.2) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def summarize(self, text: str, *, max_words: int) -> str:
        system_prompt = (
            "You are a technical editor for emergency field manuals. "
            "Condense the provided passages into precise, factually grounded instructions. "
            "Do not invent new information and always keep the response under the requested length."
        )
        user_prompt = (
            "Summarize the following text so it fits within {max_words} words. "
            "Use short sentences or bullet points and preserve critical cautions.\n\n{text}"
        ).format(max_words=max_words, text=text[:6000])

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=800,
            )
            content: Optional[str] = response.choices[0].message.content if response.choices else None
            if not content:
                logger.warning("LLM response was empty; falling back to source text")
                return text
            return content.strip()
        except Exception as exc:  # pragma: no cover - network failure fallback
            logger.error("LLM summarization failed: %s", exc)
            return text
