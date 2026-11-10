# SPDX-License-Identifier: Apache-2.0
"""Stable server-side shard assignment for the 2,000-instance matrix."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping


def shard_for(instance_id: str, shard_count: int) -> int:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    prefix = hashlib.sha256(str(instance_id).encode("utf-8")).hexdigest()[:8]
    return int(prefix, 16) % shard_count


def partition_instances(instances: Iterable[Mapping[str, Any]], shard_count: int) -> dict[int, list[dict[str, Any]]]:
    result = {index: [] for index in range(shard_count)}
    for instance in instances:
        instance_id = instance.get("benchmark_instance_id")
        if not instance_id:
            raise ValueError("instance has no benchmark_instance_id")
        result[shard_for(str(instance_id), shard_count)].append(dict(instance))
    for rows in result.values():
        rows.sort(key=lambda row: str(row["benchmark_instance_id"]))
    return result
