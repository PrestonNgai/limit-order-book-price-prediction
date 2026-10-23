"""Financial state invariants, atomicity and explicit recovery behavior."""

from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext

import pytest

from market_microstructure.book import (
    BookError,
    BookNotReady,
    BookState,
    InvalidBook,
    InvalidLevel,
    Level2OrderBook,
    LevelUpdate,
    Side,
)

D = Decimal


@pytest.fixture
def book():
    result = Level2OrderBook()
    result.load_snapshot(
        bids=[("99", "4"), ("100", "2"), ("98", "6")],
        asks=[("103", "7"), ("101", "3"), ("102", "5")],
    )
    return result


def test_bootstrap_required():
    book = Level2OrderBook()
    assert book.state == BookState.WAITING_FOR_SNAPSHOT
    assert book.invalid_reason is None
    with pytest.raises(BookNotReady):
        book.update_level("bid", "100", "2")
    with pytest.raises(BookNotReady):
        book.snapshot()


def test_snapshot_sorting_and_quotes(book):
    assert book.best_bid == D("100")
    assert book.best_ask == D("101")
    assert book.midprice == D("100.5")
    assert book.spread == D("1")
    assert [level.price for level in book.top_bids(2)] == [D("100"), D("99")]
    assert [level.price for level in book.top_asks(2)] == [D("101"), D("102")]
    assert len(book.top_bids(20)) == 3
    assert len(book.top_asks(20)) == 3
    assert book.validate() is None


def test_absolute_quantity_insert_replace_delete(book):
    book.update_level(Side.BID, "100", "8")
    assert book.bids[D("100")] == D("8")  # absolute, not 2 + 8
    book.update_level("bid", "100.25", "1")
    assert book.best_bid == D("100.25")
    book.update_level("bid", "100.25", "0")
    assert D("100.25") not in book.bids
    assert book.best_bid == D("100")
    book.update_level("ask", "100.75", "9")
    assert book.best_ask == D("100.75")
    book.update_level("ask", "100.75", "0")
    assert book.best_ask == D("101")


def test_absent_delete_and_empty_batch_are_noops(book):
    expected = book.snapshot()
    book.update_level("bid", "95", "0")
    book.update_level("ask", "107", "0")
    book.apply_updates([])
    assert book.snapshot() == expected


def test_repeated_price_last_quantity_wins(book):
    book.apply_updates(
        [
            LevelUpdate("bid", "100", "7"),
            LevelUpdate("bid", "100", "0"),
            LevelUpdate("bid", "100", "11"),
        ]
    )
    assert book.bids[D("100")] == D("11")


def test_batch_allows_temporary_cross_but_requires_valid_final_state(book):
    # Applying the first item separately would cross; the complete message is valid.
    book.apply_updates(
        [
            LevelUpdate("bid", "101.5", "3"),
            LevelUpdate("ask", "101", "0"),
        ]
    )
    assert book.best_bid == D("101.5")
    assert book.best_ask == D("102")
    book.validate()


@pytest.mark.parametrize("price", ["101", "102"])
def test_locked_and_crossed_updates_invalidate_and_rollback(book, price):
    before_bids, before_asks = dict(book.bids), dict(book.asks)
    with pytest.raises(InvalidBook):
        book.apply_updates(
            [
                LevelUpdate("bid", "100", "20"),
                LevelUpdate("ask", "105", "4"),
                LevelUpdate("bid", price, "1"),
            ]
        )
    assert book.state == BookState.INVALID
    assert book.invalid_reason
    # White-box assertions verify transaction rollback; public reads are disabled.
    assert book._bids == before_bids
    assert book._asks == before_asks
    with pytest.raises(InvalidBook):
        _ = book.midprice


def test_repeated_key_journal_restores_original_quantity(book):
    original = dict(book.bids)
    with pytest.raises(InvalidBook):
        book.apply_updates(
            [
                LevelUpdate("bid", "100", "0"),
                LevelUpdate("bid", "100", "17"),
                LevelUpdate("bid", "102", "1"),
            ]
        )
    assert book._bids == original


@pytest.mark.parametrize("side", ["bid", "ask"])
def test_cannot_delete_entire_side(book, side):
    levels = book.bids if side == "bid" else book.asks
    with pytest.raises(InvalidBook):
        book.apply_updates([LevelUpdate(side, price, "0") for price in levels])
    assert book.state == BookState.INVALID


@pytest.mark.parametrize("field", ["price", "quantity"])
@pytest.mark.parametrize(
    "value", ["NaN", "sNaN", "Infinity", "-Infinity", "-1", "", "abc", 1.5, True, None]
)
def test_reject_invalid_numeric_inputs(book, field, value):
    kwargs = {"side": "bid", "price": "100", "quantity": "2"}
    kwargs[field] = value
    with pytest.raises(InvalidLevel):
        book.apply_updates([LevelUpdate(**kwargs)])
    assert book.state == BookState.INVALID


def test_zero_price_is_invalid(book):
    with pytest.raises(InvalidLevel):
        book.update_level("bid", "0", "2")


@pytest.mark.parametrize("side", ["buy", "sell", "", None, 1])
def test_invalid_side_is_rejected_without_venue_guessing(book, side):
    with pytest.raises(InvalidLevel):
        book.update_level(side, "100", "2")


def test_decimal_string_integer_and_decimal_inputs(book):
    book.update_level("bid", 100, D("0.125"))
    assert book.bids[D("100.0")] == D("0.125")


def test_bad_late_item_never_partially_mutates(book):
    original = dict(book.bids)
    with pytest.raises(InvalidLevel):
        book.apply_updates([LevelUpdate("bid", "100", "99"), ("ask", "101", "2")])
    assert book._bids == original


def test_failed_update_iterator_invalidates_without_mutation(book):
    original = dict(book.bids)

    def broken_stream():
        yield LevelUpdate("bid", "100", "99")
        raise RuntimeError("source decoder failed")

    with pytest.raises(RuntimeError, match="decoder failed"):
        book.apply_updates(broken_stream())
    assert book._bids == original
    assert book.state == BookState.INVALID


@pytest.mark.parametrize(
    ("bids", "asks"),
    [
        ([], [("101", "2")]),
        ([("100", "2")], []),
        ([("101", "2")], [("101", "2")]),
        ([("102", "2")], [("101", "2")]),
        ([("100", "0")], [("101", "2")]),
        ([("100", "1"), ("100.0", "2")], [("101", "2")]),
        ([("100", "2")], [("101", "1"), ("101.0", "2")]),
        ([("100", "-1")], [("101", "2")]),
        ([("100", "2", "extra")], [("101", "2")]),
        (None, [("101", "2")]),
    ],
)
def test_invalid_snapshots_fail_closed_preserve_old_maps(book, bids, asks):
    previous = dict(book.bids), dict(book.asks)
    with pytest.raises(BookError):
        book.load_snapshot(bids, asks)
    assert book.state == BookState.INVALID
    assert (book._bids, book._asks) == previous


def test_snapshot_replacement_clears_stale_levels_and_recovers(book):
    book.invalidate("source reported a gap")
    with pytest.raises(InvalidBook):
        book.update_level("bid", "90", "1")
    book.load_snapshot([("90", "7")], [("91", "8")])
    assert book.state == BookState.LIVE
    assert book.invalid_reason is None
    assert dict(book.bids) == {D("90"): D("7")}
    assert dict(book.asks) == {D("91"): D("8")}


@pytest.mark.parametrize(
    "reader",
    [
        lambda book: book.bids,
        lambda book: book.asks,
        lambda book: book.best_bid,
        lambda book: book.best_ask,
        lambda book: book.midprice,
        lambda book: book.spread,
        lambda book: book.top_bids(),
        lambda book: book.top_asks(),
        lambda book: book.snapshot(),
        lambda book: book.validate(),
    ],
)
def test_every_reader_refuses_invalid_state(book, reader):
    book.invalidate("disconnect")
    with pytest.raises(InvalidBook, match="disconnect"):
        reader(book)


def test_reset_requires_new_snapshot(book):
    book.reset()
    assert book.state == BookState.WAITING_FOR_SNAPSHOT
    assert book._bids == {}
    assert book._asks == {}
    with pytest.raises(BookNotReady):
        book.validate()


@pytest.mark.parametrize("reason", ["", "   ", None])
def test_invalidation_needs_reason(book, reason):
    with pytest.raises(ValueError):
        book.invalidate(reason)
    assert book.state == BookState.LIVE


@pytest.mark.parametrize("depth", [0, -1, 1.5, True, "2"])
def test_bad_depth_is_query_error_not_book_corruption(book, depth):
    with pytest.raises(ValueError):
        book.snapshot(depth)
    assert book.state == BookState.LIVE


def test_views_are_immutable_and_detached(book):
    bids = book.bids
    view = book.snapshot(2)
    with pytest.raises(TypeError):
        bids[D("100")] = D("19")
    with pytest.raises(FrozenInstanceError):
        view.bids[0].quantity = D("19")
    with pytest.raises(FrozenInstanceError):
        view.bids = ()
    book.update_level("bid", "100", "19")
    assert bids[D("100")] == D("2")
    assert view.bids[0].quantity == D("2")


def test_half_tick_mid_and_spread_are_exact_in_book_and_view():
    book = Level2OrderBook()
    book.load_snapshot([("60000.00", "1")], [("60000.01", "1")])
    with localcontext() as context:
        context.prec = 4
        assert book.midprice == D("60000.005")
        assert book.spread == D("0.01")
        assert book.snapshot().midprice == book.midprice
        assert book.snapshot().spread == book.spread


def test_validate_detects_internal_corruption(book):
    book._bids[D("100")] = D("-1")  # Deliberate corruption to exercise public audit.
    with pytest.raises(InvalidLevel):
        book.validate()
    assert book.state == BookState.INVALID
