# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import statistics
import time
from pathlib import Path
from typing import Any

from .io import atomic_write_json
from .transport import MockTransport, SlidingWindowLimiter


async def _level(samples: int, batch_size: int, workers: int, latency_seconds: float) -> dict[str, Any]:
    transport = MockTransport(
        {
            "model": "mock-concurrency-sweep",
            "model_revision": "fixture",
            "latency_seconds": latency_seconds,
        }
    )
    limiter = SlidingWindowLimiter(1_000_000, 1_000_000_000)
    semaphore = asyncio.Semaphore(workers)
    batches = [
        list(range(start, min(samples, start + batch_size)))
        for start in range(0, samples, batch_size)
    ]
    in_flight = 0
    peak = 0
    latencies: list[float] = []

    async def one(batch: list[int]) -> None:
        nonlocal in_flight, peak
        prompt = {
            "stage": "routing",
            "semantic_attempt": 1,
            "assets": [
                {
                    "sample_id": f"s{ordinal:06d}",
                    "fixture_ordinal": ordinal,
                    "allowed_routes": [
                        {
                            "simulation_domain": "volumetric_soft_body",
                            "representations": ["watertight_volume"],
                            "model_family": "fixture",
                        }
                    ],
                }
                for ordinal in batch
            ],
        }
        import json

        encoded = json.dumps(prompt, separators=(",", ":"))
        await limiter.acquire(max(1, len(encoded) // 4))
        async with semaphore:
            in_flight += 1
            peak = max(peak, in_flight)
            started = time.monotonic()
            try:
                await transport.complete(encoded, [])
            finally:
                latencies.append(time.monotonic() - started)
                in_flight -= 1

    started = time.monotonic()
    await asyncio.gather(*(one(batch) for batch in batches))
    wall = time.monotonic() - started
    ordered = sorted(latencies)

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]

    return {
        "samples": samples,
        "batch_size": batch_size,
        "requested_workers": workers,
        "observed_peak_in_flight": peak,
        "requests": len(batches),
        "wall_time_seconds": wall,
        "requests_per_second": len(batches) / wall,
        "assets_per_second": samples / wall,
        "p50_latency_seconds": percentile(0.50),
        "p95_latency_seconds": percentile(0.95),
        "mean_latency_seconds": statistics.fmean(latencies),
        "terminal_failures": 0,
    }


def run_mock_concurrency_sweep(
    *,
    samples: int,
    batch_sizes: list[int],
    worker_levels: list[int],
    latency_seconds: float,
    output: Path,
) -> dict[str, Any]:
    if samples <= 0 or any(value <= 0 for value in [*batch_sizes, *worker_levels]):
        raise ValueError("samples, batch sizes, and worker levels must be positive")

    async def run() -> list[dict[str, Any]]:
        levels: list[dict[str, Any]] = []
        for batch_size in batch_sizes:
            for workers in worker_levels:
                levels.append(await _level(samples, batch_size, workers, latency_seconds))
        return levels

    levels = asyncio.run(run())
    report = {
        "schema_name": "softsim.mock_concurrency_sweep",
        "schema_version": 1,
        "synthetic_fixture": True,
        "publishable": False,
        "latency_seconds": latency_seconds,
        "levels": levels,
    }
    atomic_write_json(output, report)
    return report
