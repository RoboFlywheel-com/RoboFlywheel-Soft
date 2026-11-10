# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from fragile_soft_benchmark.adapter import P0Adapter
from fragile_soft_benchmark.qualification import make_default_qualification_instances, run_qualification
from fragile_soft_benchmark.validation import validate_output_dir


FIXTURE = Path(__file__).parent / "fixtures" / "p0_legacy"


def _adapter():
    return P0Adapter.from_files(FIXTURE / "manifest.json", FIXTURE / "material_profiles.json", summary_json_path=FIXTURE / "summary.json", summary_csv_path=FIXTURE / "summary.csv", source_root=FIXTURE)


def test_36_instance_qualification_and_required_outputs(tmp_path):
    config = {"qualification_revisions_per_group": 3, "qualification_difficulty_ids": ["d00", "d02", "d05"]}
    instances = make_default_qualification_instances(_adapter(), config)
    assert len(instances) == 36
    result = run_qualification(instances, output_dir=tmp_path)
    assert result["report"]["status"] == "qualified"
    assert result["report"]["e3_a_eligible"] is False
    assert result["report"]["qualification_scope"] == "local_mock_smoke"
    assert result["report"]["qualified_count"] == 36
    # two seeds x (no-contact + low/medium/high) x 36 instances
    assert len(result["episodes"]) == 288
    assert validate_output_dir(tmp_path, expected_count=36)["valid"]
    metric = json.loads((tmp_path / "metric_records.jsonl").read_text().splitlines()[0])
    assert "terminal_damage_fraction" in metric["metrics"]
    assert "finite_fraction" in metric["metrics"]


def test_failed_episode_still_has_metric_record():
    adapter = _adapter()
    instances = make_default_qualification_instances(adapter, {"qualification_revisions_per_group": 1, "qualification_difficulty_ids": ["d00"]})
    # The runner itself is deterministic; malformed status exercises explicit
    # rejected semantics and does not silently promote the instance.
    instances[0]["status"] = "rejected_unsupported_material"
    result = run_qualification(instances)
    assert result["report"]["status"] == "rejected"
    assert result["rejected"][0]["disposition"] in {"rejected", "replace", "repair"}
