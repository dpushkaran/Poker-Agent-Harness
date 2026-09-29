"""Minimal Ollama HTTP client with schema-constrained JSON output."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


class LLMError(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str
    latency_s: float
    prompt_tokens: int
    output_tokens: int
    thinking: str | None = None


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:30b-a3b",
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def chat(
        self,
        messages: list[dict],
        schema: dict | None = None,
        temperature: float = 0.2,
        think: bool = False,
        seed: int | None = None,
    ) -> ChatResult:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": {"temperature": temperature},
        }
        if schema is not None:
            body["format"] = schema
        if seed is not None:
            body["options"]["seed"] = seed
        start = time.perf_counter()
        try:
            r = self._http.post("/api/chat", json=body)
        except httpx.TimeoutException as e:
            raise LLMError(f"Ollama timed out after {self._http.timeout.read}s") from e
        except httpx.HTTPError as e:
            raise LLMError(f"cannot reach Ollama: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"Ollama returned {r.status_code}: {r.text[:200]}")
        data = r.json()
        msg = data.get("message") or {}
        return ChatResult(
            content=msg.get("content", ""),
            latency_s=time.perf_counter() - start,
            prompt_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            thinking=msg.get("thinking"),
        )

    def available_models(self) -> list[str]:
        try:
            r = self._http.get("/api/tags")
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"cannot reach Ollama: {e}") from e
        return [m["name"] for m in r.json().get("models", [])]

    def health(self) -> tuple[bool, str]:
        """(ok, message) describing whether the configured model is ready."""
        try:
            models = self.available_models()
        except LLMError as e:
            return False, f"{e}. Start it with `ollama serve`."
        if self.model not in models and f"{self.model}:latest" not in models:
            return False, f"model {self.model} not pulled; run `ollama pull {self.model}`"
        return True, f"{self.model} ready"
