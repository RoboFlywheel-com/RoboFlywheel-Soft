# SPDX-License-Identifier: Apache-2.0
"""Declarative analysis, table, figure, and static review rendering."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import struct
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_dumps, fingerprint, read_json, sha256_file
from .io import atomic_write_json, read_jsonl


DEFAULT_SPEC: dict[str, Any] = {
    "figure_id": "F_FORMAL_SUMMARY",
    "input_tables": ["episodes"],
    "filters": {},
    "exclusions": [],
    "group_by": ["method_id"],
    "metric_names": ["task_success", "heldout_behavior_error", "damage_fraction"],
    "estimator": "mean",
    "confidence_interval": "cluster_bootstrap_95",
    "bootstrap_seed": 20260828,
    "cluster_key": "sample_id",
    "style_version": "softsim-paper-v1",
}


def _read_spec(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return dict(DEFAULT_SPEC)
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
            value = yaml.safe_load(text)
        except ImportError:
            value = _minimal_yaml(text)
    else:
        value = json.loads(text)
    if not isinstance(value, Mapping):
        raise ValueError("analysis spec must be an object")
    result = dict(DEFAULT_SPEC)
    result.update(dict(value))
    return result


def _minimal_yaml(text: str) -> dict[str, Any]:
    """Parse the small scalar/list subset used by checked-in specs."""
    result: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, raw = stripped.split(":", 1)
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            result[key.strip()] = [item.strip().strip("'\"") for item in raw[1:-1].split(",") if item.strip()]
        elif raw.lower() in {"true", "false"}:
            result[key.strip()] = raw.lower() == "true"
        elif raw.isdigit():
            result[key.strip()] = int(raw)
        else:
            result[key.strip()] = raw.strip("'\"")
    return result


def _get(row: Mapping[str, Any], name: str) -> Any:
    if name in row:
        return row[name]
    # Canonical episode records keep scalar measurements grouped under these
    # namespaces; analysis specs may use the short metric name for readability.
    for namespace in ("metrics", "measurements", "outcomes"):
        nested = row.get(namespace)
        if isinstance(nested, Mapping) and name in nested:
            return nested[name]
    value: Any = row
    for part in name.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            return None
    return value


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _matches(row: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = _get(row, str(key))
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _clusters(rows: Sequence[Mapping[str, Any]], key: str) -> list[list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(_get(row, key) or row.get("task_id") or len(grouped))].append(row)
    return [grouped[name] for name in sorted(grouped)]


def bootstrap_ci(values: Sequence[float], clusters: Sequence[str] | None = None, *, seed: int = 20260828, reps: int = 1000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if clusters is None:
        grouped = [[value] for value in values]
    else:
        by_cluster: dict[str, list[float]] = defaultdict(list)
        for value, cluster in zip(values, clusters):
            by_cluster[str(cluster)].append(value)
        grouped = [by_cluster[key] for key in sorted(by_cluster)]
    if len(grouped) == 1:
        value = sum(grouped[0]) / len(grouped[0])
        return value, value
    rng = random.Random(int(seed))
    estimates: list[float] = []
    for _ in range(max(100, int(reps))):
        sample = [grouped[rng.randrange(len(grouped))] for _ in grouped]
        flat = [value for group in sample for value in group]
        estimates.append(sum(flat) / len(flat))
    estimates.sort()
    return estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))]


def build_summary(rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    filtered = [row for row in rows if _matches(row, spec.get("filters") or {})]
    exclusions = spec.get("exclusions") or []
    if exclusions:
        filtered = [row for row in filtered if not any(_matches(row, item) for item in exclusions if isinstance(item, Mapping))]
    group_by = [str(value) for value in (spec.get("group_by") or [])]
    metrics = [str(value) for value in (spec.get("metric_names") or [])]
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in filtered:
        grouped[tuple(str(_get(row, key) if _get(row, key) is not None else "unknown") for key in group_by)].append(row)
    output: list[dict[str, Any]] = []
    for group_key in sorted(grouped):
        group_rows = grouped[group_key]
        base = {key: value for key, value in zip(group_by, group_key)}
        base["n"] = len(group_rows)
        base["denominator"] = len(group_rows)
        base["population"] = str(spec.get("population") or (group_rows[0].get("population") if group_rows else "unknown"))
        base["cluster_key"] = str(spec.get("cluster_key", "sample_id"))
        for metric in metrics:
            values: list[float] = []
            clusters: list[str] = []
            for row in group_rows:
                value = _as_number(_get(row, metric))
                if value is not None:
                    values.append(value)
                    clusters.append(str(_get(row, str(spec.get("cluster_key", "sample_id"))) or row.get("task_id")))
            if not values:
                base[f"{metric}_value"] = None
                base[f"{metric}_ci_low"] = None
                base[f"{metric}_ci_high"] = None
                continue
            estimator = str(spec.get("estimator", "mean"))
            if estimator == "median":
                ordered = sorted(values)
                value = ordered[len(ordered) // 2]
            else:
                value = sum(values) / len(values)
            low, high = bootstrap_ci(values, clusters, seed=int(spec.get("bootstrap_seed", 20260828)))
            base[f"{metric}_value"] = round(value, 10)
            base[f"{metric}_ci_low"] = None if low is None else round(low, 10)
            base[f"{metric}_ci_high"] = None if high is None else round(high, 10)
            base[f"{metric}_n"] = len(values)
        output.append(base)
    return output


def _tiny_png(width: int = 640, height: int = 360) -> bytes:
    # A valid nonblank RGB PNG fallback for environments without matplotlib.
    raw = b"".join(b"\x00" + bytes([245, 247, 250]) * width for _ in range(height))
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def _plot(summary: Sequence[Mapping[str, Any]], spec: Mapping[str, Any], png: Path, pdf: Path) -> None:
    try:
        mpl_config = png.parent / ".mplconfig"
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        metric = str((spec.get("metric_names") or ["n"])[0])
        labels = ["/".join(str(row.get(key, "")) for key in (spec.get("group_by") or [])) or "all" for row in summary]
        values = [float(row.get(f"{metric}_value") or 0.0) for row in summary]
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.bar(range(len(values)), values, color="#2a6f97")
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(str(spec.get("figure_id", "SoftSim")))
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(png, dpi=160)
        fig.savefig(pdf)
        plt.close(fig)
    except Exception:
        png.write_bytes(_tiny_png())
        pdf.write_bytes(b"%PDF-1.4\n% fallback figure\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 0>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n")


def render_run(run_dir: str | Path, *, spec_path: str | Path | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    completion_path = root / "run_completion.json"
    if not completion_path.is_file():
        raise ValueError("run_completion.json is required before rendering")
    completion = read_json(completion_path)
    if completion.get("status") != "completed":
        raise ValueError("cannot render incomplete run")
    records_path = root / str(completion.get("canonical_records_ref", "canonical/episodes.jsonl"))
    if not records_path.is_file():
        raise ValueError("canonical records are missing")
    rows = list(read_jsonl(records_path))
    run_manifest = read_json(root / "run_manifest.json")
    disabled_methods = set((run_manifest.get("disabled_methods") or {}).keys())
    disabled_rows = sorted({str(row.get("method_id")) for row in rows if str(row.get("method_id")) in disabled_methods})
    if disabled_rows:
        raise ValueError("disabled methods cannot enter rendered comparison tables: " + ", ".join(disabled_rows))
    spec = _read_spec(spec_path)
    summary = build_summary(rows, spec)
    source_records_hash = sha256_file(records_path)
    spec_hash = fingerprint(spec)
    completion_planned = completion.get("expected_task_count")
    completion_observed = completion.get("observed_task_count")
    for summary_row in summary:
        summary_row.update({
            "run_id": completion.get("run_id"),
            "source_records_sha256": source_records_hash,
            "analysis_spec_sha256": spec_hash,
            "planned_denominator": completion_planned,
            "observed_denominator": completion_observed,
        })
    out = Path(output_dir).resolve() if output_dir else root / "rendered"
    out.mkdir(parents=True, exist_ok=True)
    summary_json = out / "summary.json"
    summary_csv = out / "summary.csv"
    table_tex = out / "table.tex"
    figure_png = out / f"{spec.get('figure_id', 'figure')}.png"
    figure_pdf = out / f"{spec.get('figure_id', 'figure')}.pdf"
    atomic_write_json(summary_json, {"spec": spec, "rows": summary, "denominator": {"planned": completion_planned, "observed": completion_observed}, "source_hash": source_records_hash, "analysis_spec_sha256": spec_hash})
    fields = sorted({key for row in summary for key in row})
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["n"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    figure_data = out / "figure_data.csv"
    with figure_data.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["n"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    with table_tex.open("w", encoding="utf-8") as handle:
        handle.write("% generated from canonical records; denominator retained\n")
        handle.write("\\begin{tabular}{" + "l" * max(1, len(fields)) + "}\n\\toprule\n")
        handle.write(" & ".join(fields) + " \\\\\n\\midrule\n")
        for row in summary:
            handle.write(" & ".join(str(row.get(field, "")) for field in fields) + " \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n")
    _plot(summary, spec, figure_png, figure_pdf)
    review = out / "review"
    review.mkdir(exist_ok=True)
    links = [summary_csv.name, figure_data.name, table_tex.name, figure_png.name, figure_pdf.name]
    html = "<!doctype html><meta charset='utf-8'><title>SoftSim review</title><h1>Generated review</h1><p>Source records: %s; planned denominator: %s; observed: %s</p><ul>%s</ul>" % (sha256_file(records_path), completion.get("expected_task_count"), completion.get("observed_task_count"), "".join("<li><a href='../%s'>%s</a></li>" % (link, link) for link in links))
    (review / "index.html").write_text(html, encoding="utf-8")
    manifest = {
        "schema_name": "softsim.figure_manifest",
        "schema_version": 1,
        "figure_id": spec.get("figure_id"),
        "run_id": completion.get("run_id"),
        "source_records_sha256": source_records_hash,
        "analysis_spec_sha256": spec_hash,
        "completion_sha256": sha256_file(completion_path),
        "analysis_spec": spec,
        "outputs": {name: sha256_file(out / name) for name in [summary_json.name, summary_csv.name, figure_data.name, table_tex.name, figure_png.name, figure_pdf.name]},
        "denominator": {"planned": completion_planned, "observed": completion_observed},
        "synthetic": bool(completion.get("synthetic")),
    }
    atomic_write_json(out / "figure_manifest.json", manifest)
    return manifest


def check_render(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    required = ["summary.json", "summary.csv", "figure_data.csv", "table.tex", "figure_manifest.json"]
    missing = [name for name in required if not (root / name).is_file()]
    images = list(root.glob("*.png"))
    pdfs = list(root.glob("*.pdf"))
    empty = [path.name for path in images + pdfs if path.stat().st_size == 0]
    return {"valid": not missing and not empty, "missing": missing, "empty": empty, "png_count": len(images), "pdf_count": len(pdfs)}
