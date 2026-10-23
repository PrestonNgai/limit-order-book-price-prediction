from decimal import Decimal

import numpy as np
import pytest
from feature_helpers import MS, no_trades, observation_rows, observations

from market_microstructure.processing.feature_config import FeatureConfig
from market_microstructure.processing.features import feature_arrays, feature_table, safe_divide


def test_hand_calculated_book_depth_imbalance_and_microprice():
    table, groups = feature_table(observations(), FeatureConfig(), 100)
    row = table.to_pylist()[0]
    assert row["f_midprice"] == 100
    assert row["f_spread"] == 2
    assert row["f_relative_spread"] == 0.02
    for k in (1, 3, 5, 10):
        assert row[f"f_bid_depth_{k}"] == 2 * k
        assert row[f"f_ask_depth_{k}"] == k
        assert row[f"f_obi_{k}"] == pytest.approx(1 / 3)
    assert row["f_microprice"] == pytest.approx(301 / 3)
    assert row["f_microprice_minus_mid"] == pytest.approx(1 / 3)
    assert set(groups) == {"book", "imbalance", "depth", "microprice", "flow", "trades", "dynamics"}
    assert sum(map(len, groups.values())) == 99
    assert all(not name.startswith(("y_", "label_")) for name in table.column_names)


def test_hand_calculated_full_trade_window():
    row = feature_table(observations(), FeatureConfig(), 100)[0].to_pylist()[5]
    assert row["f_trade_count_500ms"] == 5
    assert row["f_trade_volume_500ms"] == 15
    assert row["f_trade_buy_volume_500ms"] == 10
    assert row["f_trade_sell_volume_500ms"] == 5
    assert row["f_trade_imbalance_500ms"] == pytest.approx(1 / 3)
    assert row["f_trade_mean_size_500ms"] == 3
    assert row["f_trade_vwap_500ms"] == 100
    assert row["f_trade_vwap_minus_mid_500ms"] == 0


def test_displayed_depth_flow_counts_differences_not_initial_depth():
    rows = observation_rows(10)
    for i, row in enumerate(rows):
        for level in range(1, 11):
            row[f"bid_sz_{level}"] += Decimal(i) / 10
            row[f"ask_sz_{level}"] -= Decimal(i) / 100
    output = feature_table(observations(rows), FeatureConfig(), 100)[0].to_pylist()
    assert output[0]["f_bid_added_100ms"] is None
    assert output[4]["f_bid_added_500ms"] is None
    assert output[5]["f_bid_added_500ms"] == pytest.approx(5)
    assert output[5]["f_ask_removed_500ms"] == pytest.approx(0.5)
    assert output[5]["f_depth_flow_imbalance_500ms"] == pytest.approx(5.5)


def test_window_lower_boundary_is_excluded_and_zero_trade_policy():
    rows = observation_rows(8)
    for row in rows:
        no_trades(row)
    rows[1].update(
        trade_count=1,
        trade_volume=Decimal(2),
        trade_buy_volume=Decimal(2),
        trade_notional=Decimal(198),
    )
    output = feature_table(observations(rows), FeatureConfig(), 100)[0].to_pylist()
    assert output[5]["f_trade_volume_500ms"] == 2
    assert output[5]["f_trade_vwap_500ms"] == 99
    assert output[6]["f_trade_volume_500ms"] == 0
    assert output[6]["f_trade_imbalance_500ms"] == 0
    assert output[6]["f_trade_vwap_500ms"] is None
    assert output[6]["f_trade_mean_size_500ms"] is None


def test_unknown_coverage_and_baseline_warmup_are_not_zero_activity():
    rows = observation_rows(8)
    for row in rows:
        row["trade_baseline_receive_time_ns"] = 50 * MS
    rows[4]["trade_coverage_uncertain"] = True
    output = feature_table(observations(rows), FeatureConfig(), 100)[0].to_pylist()
    assert output[1]["f_trade_count_100ms"] is None
    assert output[2]["f_trade_count_100ms"] == 1
    assert output[4]["f_trade_count_100ms"] is None
    assert output[5]["f_trade_count_500ms"] is None
    assert output[6]["f_trade_count_500ms"] is None
    assert output[4]["f_obi_1"] is not None


def test_missing_depth_propagates_without_disabling_shallower_features():
    rows = observation_rows(8)
    rows[2]["ask_px_10"] = rows[2]["ask_sz_10"] = None
    output = feature_table(observations(rows), FeatureConfig(), 100)[0].to_pylist()
    assert output[2]["f_ask_depth_10"] is None
    assert output[2]["f_obi_10"] is None
    assert output[2]["f_obi_5"] == pytest.approx(1 / 3)
    assert output[3]["f_ask_added_100ms"] is None


def test_lags_are_past_only():
    table = observations(n=12, mids=list(range(100, 112)))
    rows = feature_table(table, FeatureConfig(), 100)[0].to_pylist()
    assert rows[5]["f_mid_return_500ms"] == 0.05
    assert rows[5]["f_microprice_change_500ms"] == 5
    assert rows[4]["f_mid_return_500ms"] is None
    assert rows[1]["f_trade_imbalance_100ms_lag_100ms"] is None
    assert rows[2]["f_trade_imbalance_100ms_lag_100ms"] == pytest.approx(1 / 3)


def test_future_perturbation_and_truncation_leave_past_features_unchanged():
    original = observation_rows(90)
    changed = observation_rows(90)
    for row in changed[31:]:
        row["midprice"] = Decimal(999)
        row["bid_sz_1"] = Decimal(9000)
        row["trade_notional"] = Decimal(100000)
        row["trade_volume"] = Decimal(500)
    a, _ = feature_arrays(observations(original), FeatureConfig(), 100)
    b, _ = feature_arrays(observations(changed), FeatureConfig(), 100)
    c, _ = feature_arrays(observations(original[:31]), FeatureConfig(), 100)
    for name in a:
        np.testing.assert_array_equal(a[name][:31], b[name][:31])
        np.testing.assert_array_equal(a[name][:31], c[name])


@pytest.mark.parametrize("change", ["gap", "duplicate", "reorder", "segment"])
def test_reject_unsafe_observation_block(change):
    rows = observation_rows(5)
    if change == "segment":
        rows[3]["segment_id"] = "different"
    else:
        rows[3]["timestamp_ns"] += {"gap": MS, "duplicate": -100 * MS, "reorder": -200 * MS}[change]
    with pytest.raises(ValueError):
        feature_table(observations(rows), FeatureConfig(), 100)


def test_safe_divide_and_zero_depth():
    np.testing.assert_array_equal(safe_divide([1, 0, 4], [0, 0, 2]), [np.nan, np.nan, 2])
    rows = observation_rows(2)
    rows[1]["bid_sz_1"] = rows[1]["ask_sz_1"] = Decimal(0)
    result = feature_table(observations(rows), FeatureConfig(), 100)[0].to_pylist()
    assert result[1]["f_obi_1"] is None


def test_tiny_trade_after_large_trade_keeps_vwap_precision():
    rows = observation_rows(8)
    for row in rows:
        no_trades(row)
    rows[1].update(
        trade_count=1,
        trade_volume=Decimal("10000"),
        trade_buy_volume=Decimal("10000"),
        trade_notional=Decimal("846201500"),
    )
    rows[2].update(
        trade_count=1,
        trade_volume=Decimal(".000001"),
        trade_buy_volume=Decimal(".000001"),
        trade_notional=Decimal(".08462015"),
    )
    result = feature_table(observations(rows), FeatureConfig(), 100)[0].to_pylist()
    assert result[2]["f_trade_volume_100ms"] == 0.000001
    assert result[2]["f_trade_vwap_100ms"] == pytest.approx(84620.15, rel=0, abs=1e-8)
    assert result[6]["f_trade_vwap_500ms"] == pytest.approx(84620.15, rel=0, abs=1e-8)
    assert result[7]["f_trade_volume_500ms"] == 0
