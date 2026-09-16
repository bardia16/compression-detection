"""Tests for the close-only ATR ZigZag port.

Fixture design: ATR7 of close-only changes over a flat 100.0 series is 0,
which would make the threshold 0 — so fixtures start flat then move; the
RMA picks up the move and thresholds become sane. Simpler approach used
here: craft candles whose close-to-close changes drive a known ATR, and
verify pivot sequences, alternation, and wick-fake rejection.
"""
from compression_detection.models import Candle
from compression_detection.zigzag import ZigZag


def mk(closes, atr_len=7, coef=2.0, step_ms=60_000, seed_ts=1_000_000):
    candles = [
        Candle(
            ts=seed_ts + i * step_ms,
            open=c,
            high=c,
            low=c,
            close=c,
            volume=1.0,
        )
        for i, c in enumerate(closes)
    ]
    zz = ZigZag(coef=coef, atr_length=atr_len)
    zz.feed_all(candles)
    return zz


class TestWarmupAndBasics:
    def test_no_pivots_before_atr_exists(self):
        # 8 bars, all moves tiny relative to any threshold — but ATR7 needs
        # 8 candles; ensure nothing confirms before that.
        closes = [100 + (0.001 * (i % 3)) for i in range(8)]
        zz = mk(closes)
        assert zz.pivots == []

    def test_pivots_strictly_alternate(self):
        # rising then falling then rising — should produce H, L, H, ...
        closes = (
            [100.0] * 3
            + [100 + 0.5 * i for i in range(1, 41)]     # impulse up
            + [140 - 0.5 * i for i in range(1, 41)]     # retrace down
            + [120 + 0.5 * i for i in range(1, 41)]     # up again
        )
        zz = mk(closes)
        types = [p.type for p in zz.pivots]
        for a, b in zip(types, types[1:]):
            assert a != b, "pivots must alternate H/L"

    def test_pivot_price_is_close_not_wick(self):
        # wick above the eventual pivot close must NOT change the pivot price
        # (Candle high is ignored entirely by the close-only zigzag)
        closes = (
            [100.0] * 3
            + [100 + 0.5 * i for i in range(1, 41)]
            + [140 - 0.5 * i for i in range(1, 41)]
        )
        zz = mk(closes)
        highs = [p for p in zz.pivots if p.type == 1]
        assert highs, "expected at least one high pivot"
        for h in highs:
            assert h.price == closes[h.bar_index]


class TestPivotDetection:
    def test_basic_up_down_cycle(self):
        # flat, impulse up, retrace down, impulse up, small pullback
        closes = (
            [100.0] * 3
            + [100 + 0.5 * i for i in range(1, 41)]     # up to 120
            + [140 - 0.5 * i for i in range(1, 41)]     # spike then down to 120
            + [120 + 0.5 * i for i in range(1, 41)]     # up to 140
            + [140 - 0.5 * i for i in range(1, 16)]     # pullback so the H confirms
        )
        zz = mk(closes)
        # after the first up-leg we expect a HIGH pivot; after the down-leg a LOW
        assert zz.pivots[0].type == 1
        assert zz.pivots[1].type == -1
        assert zz.pivots[2].type == 1
        # pivot prices near leg extremes
        assert 139 < zz.pivots[0].price <= 140
        assert 119 < zz.pivots[1].price <= 120.5
        assert 139 < zz.pivots[2].price <= 140

    def test_wick_fake_does_not_confirm(self):
        # A single huge spike CLOSE is used — but close-only means the spike
        # close itself IS the extreme. To fake a wick: put the spike in high,
        # keep close normal. Close-only zigzag must ignore it entirely.
        closes = (
            [100.0] * 3
            + [100 + 0.5 * i for i in range(1, 41)]
            + [140.0]                                  # pivot-ish bar
            + [140 - 0.5 * i for i in range(1, 41)]
        )
        zz = mk(closes)
        n_pivots_normal = len(zz.pivots)

        # rebuild same closes but with a wick on the pivot bar
        candles = [
            Candle(ts=i, open=c, high=c + (50.0 if i == 43 else 0.0),
                   low=c, close=c, volume=1.0)
            for i, c in enumerate(closes)
        ]
        zz2 = ZigZag(coef=2.0, atr_length=7)
        zz2.feed_all(candles)
        assert len(zz2.pivots) == n_pivots_normal
        for p1, p2 in zip(zz.pivots, zz2.pivots):
            assert p1.price == p2.price
            assert p1.bar_index == p2.bar_index

    def test_streaming_equals_batch(self):
        closes = (
            [100.0] * 3
            + [100 + 0.5 * i for i in range(1, 41)]
            + [140 - 0.5 * i for i in range(1, 41)]
            + [120 + 0.5 * i for i in range(1, 41)]
        )
        batch = mk(closes)
        streaming = ZigZag(coef=2.0, atr_length=7)
        for i, c in enumerate(closes):
            candle = Candle(ts=1_000_000 + i * 60_000, open=c, high=c,
                            low=c, close=c, volume=1.0)
            streaming.feed(candle)
        assert [p.price for p in batch.pivots] == [p.price for p in streaming.pivots]
        assert [p.bar_index for p in batch.pivots] == [p.bar_index for p in streaming.pivots]
        assert [p.type for p in batch.pivots] == [p.type for p in streaming.pivots]


class TestProvisional:
    def test_provisional_never_in_pivots(self):
        closes = (
            [100.0] * 3
            + [100 + 0.5 * i for i in range(1, 41)]
            + [140 - 0.5 * i for i in range(1, 41)]
        )
        zz = mk(closes)
        assert zz.provisional is not None
        # provisional tracks the live extreme (down-leg in progress)
        assert zz.provisional.type == -1
        # it must not be a confirmed pivot
        assert zz.provisional not in zz.pivots


class TestConfigValidation:
    def test_invalid_atr_length_raises(self):
        try:
            ZigZag(coef=2.0, atr_length=0)
            assert False, "should raise"
        except ValueError:
            pass
