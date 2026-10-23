import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from feature_helpers import observation_rows, observations

from market_microstructure.processing.build_dataset import file_hash
from market_microstructure.processing.feature_config import FeatureConfig
from market_microstructure.processing.features import IDENTIFIERS, feature_table
from market_microstructure.processing.labels import eligible_target, label_table


def refresh_manifest(source):
    path = Path(source) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name in manifest["outputs"]:
        manifest["outputs"][name] = {
            "sha256": file_hash(Path(source) / name),
            "bytes": (Path(source) / name).stat().st_size,
        }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def research_fixture(path, *, n=100, periods=None, horizons=(100, 500, 1000, 5000)):
    path.mkdir()
    config = FeatureConfig(horizons_ms=horizons)
    features, targets, groups = [], {h: [] for h in horizons}, None
    for number, (count, offset, session) in enumerate(periods or [(n, 0, "s")]):
        rows = observation_rows(
            count,
            offset_ms=offset,
            segment=f"r/{session}/{number}",
            mids=[100 + i / 100 for i in range(count)],
        )
        for row in rows:
            row["session_id"] = session
        obs = observations(rows)
        f, groups = feature_table(obs, config, 100)
        features.append(f)
        label = label_table(obs, config, 100, ".01")
        joined = pa.table(
            {
                **{name: obs[name] for name in IDENTIFIERS},
                **{name: label[name] for name in label.column_names},
            }
        )
        for h in horizons:
            targets[h].append(
                eligible_target(joined, h).select(
                    list(IDENTIFIERS) + [f"label_end_ns_{h}ms", f"y_return_{h}ms", f"y_class_{h}ms"]
                )
            )
    pq.write_table(pa.concat_tables(features), path / "features.parquet", row_group_size=13)
    for h in horizons:
        pq.write_table(
            pa.concat_tables(targets[h]), path / f"labels_{h}ms.parquet", row_group_size=11
        )
    catalog = {"feature_columns": [n for names in groups.values() for n in names], "groups": groups}
    (path / "feature_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "kind": "features_and_labels",
        "interval_ms": 100,
        "configuration": config.to_dict(),
        "threshold": {"mode": "price_ticks", "tick_size": ".01", "epsilon_ticks": "1"},
        "outputs": {
            name: {}
            for name in ["features.parquet", "feature_catalog.json"]
            + [f"labels_{h}ms.parquet" for h in horizons]
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    refresh_manifest(path)
    return path
