# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import random
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any


class TransportError(RuntimeError):
    def __init__(self, category: str, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.category = category
        self.retry_after = retry_after


class FatalTransportError(TransportError):
    pass


@dataclass(frozen=True)
class TransportResult:
    response: dict[str, Any]
    metadata: dict[str, Any]


class SlidingWindowLimiter:
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int) -> None:
        if self.requests_per_minute <= 0 or self.tokens_per_minute <= 0:
            raise ValueError("RPM and TPM limits must be positive")
        if estimated_tokens > self.tokens_per_minute:
            raise ValueError(
                f"Single request estimate {estimated_tokens} exceeds node TPM limit {self.tokens_per_minute}"
            )
        while True:
            wait_seconds = 0.0
            async with self._lock:
                now = time.monotonic()
                cutoff = now - 60.0
                while self._requests and self._requests[0] <= cutoff:
                    self._requests.popleft()
                while self._tokens and self._tokens[0][0] <= cutoff:
                    self._tokens.popleft()
                token_total = sum(value for _, value in self._tokens)
                if len(self._requests) < self.requests_per_minute and token_total + estimated_tokens <= self.tokens_per_minute:
                    self._requests.append(now)
                    self._tokens.append((now, estimated_tokens))
                    return
                deadlines: list[float] = []
                if len(self._requests) >= self.requests_per_minute:
                    deadlines.append(self._requests[0] + 60.0)
                if token_total + estimated_tokens > self.tokens_per_minute and self._tokens:
                    running = token_total
                    for timestamp, value in self._tokens:
                        running -= value
                        if running + estimated_tokens <= self.tokens_per_minute:
                            deadlines.append(timestamp + 60.0)
                            break
                wait_seconds = max(0.001, min(deadlines) - now) if deadlines else 0.001
            await asyncio.sleep(wait_seconds)


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _strip_json_fence(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()
    return text


def _retry_after(headers: Any) -> float | None:
    value = headers.get("Retry-After") if headers else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


class QwenTransport:
    def __init__(self, config: dict[str, Any]):
        key_name = str(config.get("api_key_env", "DASHSCOPE_API_KEY"))
        api_key = os.environ.get(key_name)
        if not api_key:
            raise FatalTransportError("auth", f"Missing API key environment variable: {key_name}")
        self._api_key = api_key
        self.model = str(config["model"])
        self.model_revision = str(config.get("model_revision") or self.model)
        self.base_url = str(config["base_url"]).rstrip("/")
        self.timeout = float(config.get("timeout_seconds", 180))
        self.temperature = float(config.get("temperature", 0.1))
        self.max_tokens = int(config.get("max_tokens", 4096))
        self.enable_thinking = bool(config.get("enable_thinking", False))

    def _complete_sync(self, prompt: str, image_paths: list[Path]) -> TransportResult:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(path)}})
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Estimate uncertain simulation priors, not measured truth. Return one JSON object. "
                        "Follow the user's output_contract exactly, use only its field names, and never move "
                        "required fields into a nested or alternate schema. Never claim calibration or "
                        "verification from images."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "enable_thinking": self.enable_thinking,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise FatalTransportError("auth", f"HTTP {exc.code}") from exc
            if exc.code in {400, 404}:
                raise FatalTransportError("config", f"HTTP {exc.code}") from exc
            category = "rate_limited" if exc.code == 429 else "server_error" if exc.code >= 500 else "http_error"
            raise TransportError(category, f"HTTP {exc.code}", retry_after=_retry_after(exc.headers)) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise TransportError("timeout", type(exc).__name__) from exc
        try:
            text = response_payload["choices"][0]["message"]["content"]
            parsed = json.loads(_strip_json_fence(text))
            if not isinstance(parsed, dict):
                raise ValueError("response is not an object")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parsed = {
                "assets": [],
                "_semantic_rejection": "invalid_json_or_response_shape",
                "_raw_text": str(response_payload)[:20000],
            }
        return TransportResult(
            response=parsed,
            metadata={
                "model": response_payload.get("model", self.model),
                "model_revision": self.model_revision,
                "request_id": response_payload.get("id"),
                "usage": response_payload.get("usage"),
                "latency_seconds": time.monotonic() - started,
                "transport": "qwen",
            },
        )

    async def complete(self, prompt: str, image_paths: list[Path]) -> TransportResult:
        return await asyncio.to_thread(self._complete_sync, prompt, image_paths)


class MockTransport:
    def __init__(self, config: dict[str, Any]):
        self.model = str(config.get("model", "mock-physical-prior-v1"))
        self.model_revision = str(config.get("model_revision", "fixture"))
        self.invalid_first_modulo = int(config.get("invalid_first_modulo", 0))
        self.latency_seconds = float(config.get("latency_seconds", 0.0))
        self.calls = 0

    async def complete(self, prompt: str, image_paths: list[Path]) -> TransportResult:
        # Keep large local mock gates cooperative so configured concurrency is
        # exercised instead of being serialized by synchronous fixture work.
        await asyncio.sleep(0)
        del image_paths
        if self.latency_seconds:
            await asyncio.sleep(self.latency_seconds)
        request = json.loads(prompt)
        self.calls += 1
        assets: list[dict[str, Any]] = []
        for item in request["assets"]:
            ordinal = int(item.get("fixture_ordinal", 0))
            if self.invalid_first_modulo and request.get("semantic_attempt", 1) == 1 and ordinal % self.invalid_first_modulo == 0:
                continue
            if request["stage"] == "routing":
                route = item["allowed_routes"][0]
                assets.append(
                    {
                        "sample_id": item["sample_id"],
                        "simulation_domain": route["simulation_domain"],
                        "representation": route["representations"][0],
                        "model_family": route["model_family"],
                        "confidence": 0.5,
                        "limitations": ["synthetic_fixture_non_publishable"],
                    }
                )
            else:
                parameters = []
                for spec in item["required_parameters"]:
                    lower = float(spec["lower"])
                    upper = float(spec["upper"])
                    value = (lower + upper) / 2.0
                    parameters.append(
                        {
                            "name": spec["name"],
                            "value": value,
                            "confidence": 0.4,
                            "rationale": "deterministic synthetic fixture value",
                        }
                    )
                assets.append({"sample_id": item["sample_id"], "parameters": parameters})
        return TransportResult(
            response={"assets": assets},
            metadata={
                "model": self.model,
                "model_revision": self.model_revision,
                "request_id": f"mock-{self.calls:08d}",
                "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(assets) * 64},
                "latency_seconds": self.latency_seconds,
                "transport": "mock",
            },
        )


async def complete_with_retries(
    transport: QwenTransport | MockTransport,
    prompt: str,
    image_paths: list[Path],
    *,
    max_retries: int,
    backoff_seconds: float,
    jitter_ratio: float,
) -> tuple[TransportResult, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_retries + 1):
        try:
            result = await transport.complete(prompt, image_paths)
            attempts.append({"attempt": attempt, "status": "success", "category": None})
            return result, attempts
        except FatalTransportError:
            raise
        except TransportError as exc:
            attempts.append({"attempt": attempt, "status": "retry", "category": exc.category})
            if attempt >= max_retries or exc.category == "invalid_json":
                raise
            base = exc.retry_after if exc.retry_after is not None else backoff_seconds * (2 ** (attempt - 1))
            jitter = base * jitter_ratio * random.random()
            await asyncio.sleep(base + jitter)
    raise AssertionError("retry loop exhausted")
