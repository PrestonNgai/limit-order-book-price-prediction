"""Causality, availability, freshness and continuity are financial data invariants."""

import json
from decimal import Decimal

import pytest

from market_microstructure.data.coinbase import CoinbaseAdapter
from market_microstructure.processing.config import ProcessingConfig
from market_microstructure.processing.replay import JournalReplay, ReplayError
from market_microstructure.processing.sampling import SnapshotSampler

MS = 1_000_000


def raw(payload, ms, index, *, session="s", run="r", mono=None):
    return {
        "schema_version": 1,
        "record_type": "raw_message",
        "source": "coinbase",
        "symbol": "BTC-USD",
        "run_id": run,
        "session_id": session,
        "message_index": index,
        "receive_time_ns": ms * MS,
        "receive_monotonic_ns": ms * MS if mono is None else mono,
        "encoding": "utf-8",
        "payload": payload if isinstance(payload, str) else json.dumps(payload),
    }


def marker(kind, ms=0, *, session="s", run="r"):
    return {
        "schema_version": 1,
        "record_type": kind,
        "source": "coinbase",
        "symbol": "BTC-USD",
        "run_id": run,
        "session_id": session,
        "receive_time_ns": ms * MS,
    }


def sample(records, *, output_date=None, **settings):
    rows, segments, quality = [], [], []
    sampler = SnapshotSampler(
        ProcessingConfig(**settings),
        rows.append,
        segments.append,
        quality.append,
        source="coinbase",
        symbol="BTC-USD",
        output_date=output_date,
    )
    replay = JournalReplay(CoinbaseAdapter(), sampler)
    for record in records:
        replay.consume(record)
    replay.finish()
    return rows, segments, quality, sampler.counts


def test_known_snapshots_are_asof_receive_time(messages):
    rows, _, _, _ = sample(
        [
            raw(messages["snapshot"], 0, 0),
            raw(messages["update"], 50, 1),
            raw(messages["heartbeat"], 250, 2),
        ]
    )
    assert [r["timestamp_ns"] // MS for r in rows] == [0, 100, 200]
    assert [r["midprice"] for r in rows] == [
        Decimal("100.01"),
        Decimal("100.015"),
        Decimal("100.015"),
    ]
    assert rows[0]["last_book_exchange_time_ns"] is None
    assert rows[1]["last_book_receive_time_ns"] == 50 * MS
    assert rows[1]["book_age_ns"] == 50 * MS
    assert all(r["last_book_receive_time_ns"] <= r["timestamp_ns"] for r in rows)


def test_equal_timestamps_include_all_messages_in_recorded_order(messages):
    a = {**messages["update"], "changes": [["buy", "100.01", "1"]]}
    b = {**messages["update"], "changes": [["buy", "100.01", "9"], ["sell", "100.02", "7"]]}
    rows, _, _, _ = sample(
        [
            raw(messages["snapshot"], 0, 0),
            raw(a, 100, 1),
            raw(b, 100, 2),
            raw(messages["heartbeat"], 150, 3),
        ]
    )
    assert rows[-1]["timestamp_ns"] == 100 * MS
    assert rows[-1]["bid_sz_1"] == Decimal("9")
    assert rows[-1]["ask_sz_1"] == Decimal("7")
    assert rows[-1]["last_message_index"] == 2


def test_future_perturbation_cannot_change_earlier_snapshots(messages):
    prefix = [raw(messages["snapshot"], 0, 0), raw(messages["heartbeat"], 450, 1)]
    a = raw(messages["update"], 500, 2)
    b = raw({**messages["update"], "changes": [["buy", "100.005", "99"]]}, 500, 2)
    end = raw(messages["heartbeat"], 650, 3)
    left = sample([*prefix, a, end])[0]
    right = sample([*prefix, b, end])[0]
    assert [r for r in left if r["timestamp_ns"] < 500 * MS] == [
        r for r in right if r["timestamp_ns"] < 500 * MS
    ]
    assert left[-1]["midprice"] != right[-1]["midprice"]


def test_late_exchange_timestamp_is_not_backdated(messages):
    later = {**messages["update"], "time": "2026-09-24T12:00:02Z"}
    earlier = {
        **messages["update"],
        "time": "2026-09-24T12:00:01Z",
        "changes": [["buy", "100.01", "8"]],
    }
    rows, _, _, counts = sample(
        [
            raw(messages["snapshot"], 0, 0),
            raw(later, 100, 1),
            raw(earlier, 250, 2),
            raw(messages["heartbeat"], 350, 3),
        ]
    )
    assert rows[2]["bid_sz_1"] == Decimal("1.25")
    assert rows[3]["bid_sz_1"] == Decimal("8")
    assert counts["exchange_time_regression"] == 1


def test_stale_hold_stops_then_update_starts_new_segment(messages):
    rows, segments, _, counts = sample(
        [
            raw(messages["snapshot"], 10, 0),
            raw(messages["heartbeat"], 1000, 1),
            raw(messages["update"], 1010, 2),
            raw(messages["heartbeat"], 1150, 3),
        ],
        max_staleness_ms=150,
    )
    assert [r["timestamp_ns"] // MS for r in rows] == [100, 1100]
    assert rows[0]["segment_id"] != rows[1]["segment_id"]
    assert segments[0]["end_reason"] == "staleness"
    assert counts["staleness_break"] == 1


def test_stale_interval_splits_even_between_grid_points(messages):
    rows, _, _, _ = sample(
        [
            raw(messages["snapshot"], 0, 0),
            raw(messages["update"], 200, 1),
        ],
        max_staleness_ms=150,
    )
    assert [r["timestamp_ns"] // MS for r in rows] == [0, 100, 200]
    assert rows[-1]["segment_id"] != rows[-2]["segment_id"]


def test_maximum_age_is_inclusive(messages):
    rows, _, _, _ = sample(
        [raw(messages["snapshot"], 0, 0), raw(messages["heartbeat"], 300, 1)], max_staleness_ms=200
    )
    assert [r["timestamp_ns"] // MS for r in rows] == [0, 100, 200]


def test_missing_raw_index_requires_snapshot_and_does_not_fill_gap(messages):
    rows, _, _, counts = sample(
        [
            raw(messages["snapshot"], 0, 0),
            raw(messages["heartbeat"], 50, 1),
            raw(messages["update"], 200, 3),
            raw(messages["snapshot"], 350, 4),
            raw(messages["heartbeat"], 450, 5),
        ]
    )
    assert [r["timestamp_ns"] // MS for r in rows] == [0, 400]
    assert counts["missing_raw_messages"] == 1
    assert counts["updates_without_snapshot"] == 1
    assert rows[0]["segment_id"] != rows[1]["segment_id"]
    assert rows[-1]["trade_coverage_uncertain"]


def test_reconnect_and_lifecycle_do_not_extend_to_detection_time(messages):
    rows, _, _, _ = sample(
        [
            marker("session_start"),
            raw(messages["snapshot"], 0, 0),
            raw(messages["heartbeat"], 50, 1),
            marker("session_end", 200),
            marker("session_start", 300, session="next"),
            raw(messages["update"], 300, 2, session="next"),
            raw(messages["snapshot"], 350, 3, session="next"),
            raw(messages["heartbeat"], 450, 4, session="next"),
            marker("recorder_stop", 10000, session="next"),
        ]
    )
    assert [r["timestamp_ns"] // MS for r in rows] == [0, 400]
    assert rows[-1]["session_id"] == "next"


def test_malformed_payload_invalidates_until_fresh_snapshot(messages):
    rows, _, _, counts = sample(
        [
            raw(messages["snapshot"], 0, 0),
            raw("{broken", 120, 1),
            raw(messages["update"], 200, 2),
            raw(messages["snapshot"], 350, 3),
            raw(messages["heartbeat"], 450, 4),
        ]
    )
    assert [r["timestamp_ns"] // MS for r in rows] == [0, 100, 400]
    assert counts["malformed_payload"] == 1


def test_crossed_book_never_produces_snapshot(messages):
    crossed = {**messages["update"], "changes": [["buy", "101", "1"]]}
    rows, _, _, counts = sample(
        [
            raw(messages["snapshot"], 0, 0),
            raw(crossed, 100, 1),
            raw(messages["snapshot"], 250, 2),
            raw(messages["heartbeat"], 350, 3),
        ]
    )
    assert [r["timestamp_ns"] // MS for r in rows] == [0, 300]
    assert counts["invalid_book"] == 1


def test_clock_regression_drops_pending_grid_and_requires_snapshot(messages):
    rows, _, _, counts = sample(
        [
            raw(messages["snapshot"], 200, 0),
            raw(messages["heartbeat"], 300, 1),
            raw(messages["update"], 250, 2),
            raw(messages["update"], 260, 3),
            raw(messages["snapshot"], 400, 4),
            raw(messages["heartbeat"], 500, 5),
        ]
    )
    assert [r["timestamp_ns"] // MS for r in rows] == [200, 400, 500]
    assert counts["clock_regression"] == 1


def test_monotonic_clock_regression_also_invalidates(messages):
    rows, _, _, counts = sample(
        [
            raw(messages["snapshot"], 0, 0, mono=100),
            raw(messages["update"], 100, 1, mono=50),
        ]
    )
    assert rows == []
    assert counts["clock_regression"] == 1


def test_no_snapshot_means_no_output_and_prefix_gap_is_explicit(messages):
    rows, _, _, counts = sample([raw(messages["update"], 0, 100)])
    assert rows == []
    assert counts["missing_input_prefix"] == 1
    assert counts["updates_without_snapshot"] == 1


@pytest.mark.parametrize("indexes", [(0, 0), (2, 1)])
def test_duplicate_or_reordered_indexes_are_fatal(messages, indexes):
    with pytest.raises(ReplayError, match="duplicate or reordered"):
        sample([raw(messages["snapshot"], 0, indexes[0]), raw(messages["update"], 100, indexes[1])])


def test_wrong_product_is_a_structural_error(messages):
    value = {**raw(messages["snapshot"], 0, 0), "symbol": "ETH-USD"}
    with pytest.raises(ReplayError, match="source/product"):
        sample([value])


def test_trade_gap_flag_preserved_without_inventing_trade_completeness(messages):
    gap = {**messages["match"], "trade_id": 14}
    rows, _, _, _ = sample(
        [
            raw(messages["snapshot"], 0, 0),
            raw(messages["reference"], 10, 1),
            raw(gap, 50, 2),
            raw(messages["heartbeat"], 150, 3),
        ]
    )
    assert rows[0]["trade_baseline_receive_time_ns"] is None
    assert rows[1]["trade_baseline_receive_time_ns"] == 10 * MS
    assert rows[1]["trade_coverage_uncertain"]


def test_missing_depth_is_null_not_zero_liquidity(messages):
    rows, _, _, _ = sample([raw(messages["snapshot"], 0, 0)])
    assert rows[0]["bid_levels_available"] == rows[0]["ask_levels_available"] == 2
    assert rows[0]["bid_px_3"] is None
    assert rows[0]["bid_sz_3"] is None
    assert rows[0]["ask_px_10"] is None


def test_date_filter_keeps_previous_day_bootstrap(messages):
    midnight_ms = 86400 * 1000
    rows, _, _, _ = sample(
        [
            raw(messages["snapshot"], midnight_ms - 150, 0),
            raw(messages["heartbeat"], midnight_ms + 250, 1),
        ],
        output_date="1970-01-02",
    )
    assert [r["timestamp_ns"] // MS for r in rows] == [
        midnight_ms,
        midnight_ms + 100,
        midnight_ms + 200,
    ]


def test_replacement_snapshot_always_starts_new_segment(messages):
    rows, _, _, _ = sample(
        [
            raw(messages["snapshot"], 0, 0),
            raw(messages["snapshot"], 150, 1),
            raw(messages["heartbeat"], 250, 2),
        ]
    )
    assert rows[0]["segment_id"] == rows[1]["segment_id"]
    assert rows[2]["segment_id"] != rows[1]["segment_id"]


def test_interval_is_configurable(messages):
    rows, _, _, _ = sample(
        [raw(messages["snapshot"], 0, 0), raw(messages["heartbeat"], 150, 1)],
        interval_ms=50,
        depth=1,
    )
    assert [r["timestamp_ns"] // MS for r in rows] == [0, 50, 100, 150]
    assert "bid_px_2" not in rows[0]


def test_annotations_do_not_advance_clock_or_double_apply(messages):
    rows, _, _, _ = sample(
        [
            raw(messages["snapshot"], 0, 0),
            {**marker("normalized_event", 10000), "event": {"kind": "book_update"}},
            raw(messages["heartbeat"], 150, 1),
        ]
    )
    assert [r["timestamp_ns"] // MS for r in rows] == [0, 100]
    assert all(r["best_bid"] == Decimal("100.00") for r in rows)


@pytest.mark.parametrize("schema", [True, 2, None, "1"])
def test_bad_schema_rejected(messages, schema):
    with pytest.raises(ReplayError):
        sample([{**raw(messages["snapshot"], 0, 0), "schema_version": schema}])


def test_raw_after_closed_session_is_fatal(messages):
    with pytest.raises(ReplayError, match="closed session"):
        sample(
            [
                raw(messages["snapshot"], 0, 0),
                marker("session_end"),
                raw(messages["update"], 100, 1),
            ]
        )
