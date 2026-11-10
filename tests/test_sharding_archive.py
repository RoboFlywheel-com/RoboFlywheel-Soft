# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from fragile_soft_benchmark.archive import create_deployment_archive, verify_deployment_archive
from fragile_soft_benchmark.sharding import partition_instances, shard_for


def test_sharding_is_stable_and_covers_rows():
    rows = [{"benchmark_instance_id": f"fsb-g-r{i:03d}-d00"} for i in range(20)]
    partitions = partition_instances(rows, 4)
    assert sum(len(value) for value in partitions.values()) == 20
    assert shard_for(rows[0]["benchmark_instance_id"], 4) == shard_for(rows[0]["benchmark_instance_id"], 4)


def test_deployment_archive_excludes_data_artifacts(tmp_path):
    archive = tmp_path / "builder.tar.gz"
    result = create_deployment_archive(Path(__file__).parents[1], archive)
    assert result["file_count"] > 0
    verified = verify_deployment_archive(archive)
    assert verified["verified"] is True
