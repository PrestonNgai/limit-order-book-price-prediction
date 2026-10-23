from decimal import Decimal

import numpy as np
import pyarrow as pa
import pytest
from feature_helpers import MS, observation_rows, observations

from market_microstructure.processing.feature_config import FeatureConfig
from market_microstructure.processing.labels import (
    TrainingDiagnostics,
    eligible_target,
    label_table,
)


def test_exact_tick_ties_and_future_returns():
    table = observations(n=6, mids=[100, 101, 100, 98, 100, 100])
    output = label_table(table, FeatureConfig(), 100, "1")
    assert output["y_class_100ms"].to_pylist() == [0, 0, -1, 1, 0, None]
    assert output["y_return_100ms"][0].as_py() == 0.01
    assert output["label_end_ns_100ms"][0].as_py() == 100 * MS
    assert eligible_target(output, 100).num_rows == 5
    assert eligible_target(output, 5000).num_rows == 0


def test_decimal_ties_not_binary_float_accidents():
    table = observations(n=4, mids=["84620.15", "84620.16", "84620.15", "84620.165"])
    result = label_table(table, FeatureConfig(), 100, ".01")
    assert result["y_class_100ms"].to_pylist() == [0, 0, 1, None]


def test_fixed_relative_epsilon():
    table = observations(n=3, mids=[100, 101, 103])
    result = label_table(table, FeatureConfig(relative_epsilon=".01"), 100, ".01")
    assert result["y_class_100ms"].to_pylist() == [0, 1, None]


@pytest.mark.parametrize("tolerance,valid", [(0, 0), (49, 0), (50, 4), (99, 4)])
def test_forward_alignment_never_uses_a_price_before_the_horizon(tolerance, valid):
    config = FeatureConfig(horizons_ms=(150,), alignment_tolerance_ms=tolerance)
    result = label_table(observations(n=6), config, 100, ".01")
    target = eligible_target(result, 150)
    assert target.num_rows == valid
    if valid:
        assert target["label_end_ns_150ms"].to_pylist() == [200 * MS, 300 * MS, 400 * MS, 500 * MS]


def test_changing_future_affects_only_labels_whose_endpoints_change():
    a = observation_rows(12)
    b = observation_rows(12)
    b[6]["midprice"] = Decimal(101)
    config = FeatureConfig(horizons_ms=(100, 500))
    x = label_table(observations(a), config, 100, ".01")
    y = label_table(observations(b), config, 100, ".01")
    assert x["y_return_100ms"][4] == y["y_return_100ms"][4]
    assert x["y_return_100ms"][5] != y["y_return_100ms"][5]
    assert x["y_return_500ms"][0] == y["y_return_500ms"][0]
    assert x["y_return_500ms"][1] != y["y_return_500ms"][1]


def test_training_diagnostics_purge_boundary_and_ignore_future_values():
    config = FeatureConfig(horizons_ms=(100, 500))
    original = observation_rows(20, mids=list(range(100, 120)))
    changed = observation_rows(20, mids=list(range(100, 120)))
    for row in changed[10:]:
        row["midprice"] = Decimal(9999)
    reports = []
    for rows in (original, changed):
        obs = observations(rows)
        labels = label_table(obs, config, 100, ".01")
        joined = pa.table(
            {
                "timestamp_ns": obs["timestamp_ns"],
                **{name: labels[name] for name in labels.column_names},
            }
        )
        diag = TrainingDiagnostics(config, 200 * MS, 1000 * MS)
        diag.consume(joined)
        reports.append(diag.report())
    assert reports[0] == reports[1]
    assert reports[0]["horizons_ms"]["100"]["eligible_rows"] == 7
    assert reports[0]["horizons_ms"]["500"]["eligible_rows"] == 3


def test_all_endpoints_strictly_future_and_within_tolerance():
    config = FeatureConfig(horizons_ms=(100, 150, 500), alignment_tolerance_ms=50)
    obs = observations(n=30)
    result = label_table(obs, config, 100, ".01")
    times = obs["timestamp_ns"].to_numpy()
    for horizon in config.horizons_ms:
        valid = result[f"label_valid_{horizon}ms"].to_numpy(zero_copy_only=False)
        ends = np.array(result[f"label_end_ns_{horizon}ms"].fill_null(0))[valid]
        delta = ends - times[valid]
        assert np.all(delta >= horizon * MS)
        assert np.all(delta <= (horizon + 50) * MS)
