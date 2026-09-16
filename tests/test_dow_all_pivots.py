"""label_all_pivots (compression-detection extension of vendored dow.py).

The compression detector needs EVERY confirmed pivot in chronological order
with its label when decidable — the initial reference pivots of each side are
label-exempt per the compression spec §5 (they only need to exist and be
within boundary tolerance). The vendored label_pivots() omits unlabeled
pivots, so this parallel function keeps the full sequence.
"""
from compression_detection.dow import PivotLabel, label_pivot
from compression_detection.models import Pivot


def test_label_all_pivots_keeps_every_pivot():
    from compression_detection.dow import label_all_pivots

    p1 = Pivot(type=1, price=100.0, bar_index=1, ts=1)
    p2 = Pivot(type=-1, price=90.0, bar_index=2, ts=2)
    p3 = Pivot(type=1, price=101.0, bar_index=3, ts=3)
    p4 = Pivot(type=-1, price=93.0, bar_index=4, ts=4)
    atr = [None, 2.0, 2.0, 2.0, 2.0]

    out = label_all_pivots([p1, p2, p3, p4], atr)

    assert len(out) == 4                      # full sequence kept
    assert out[0][1] is None                  # first high — no previous
    assert out[1][1] is None                  # first low — no previous
    assert out[2][1] is PivotLabel.EH         # 101 vs 100, within 1*2
    assert out[3][1] is PivotLabel.HL         # 93 vs 90 → > prev + 1*2


def test_label_all_pivots_returns_none_when_atr_missing():
    from compression_detection.dow import label_all_pivots

    p1 = Pivot(type=1, price=100.0, bar_index=1, ts=1)
    p2 = Pivot(type=1, price=110.0, bar_index=2, ts=2)
    out = label_all_pivots([p1, p2], [None, None, None])
    assert [lbl for _, lbl in out] == [None, None]


def test_label_pivots_vendored_behavior_unchanged():
    """Sanity: the vendored function still omits unlabeled pivots."""
    from compression_detection.dow import label_pivots

    p1 = Pivot(type=1, price=100.0, bar_index=1, ts=1)
    p2 = Pivot(type=1, price=110.0, bar_index=2, ts=2)
    out = label_pivots([p1, p2], [None, 2.0, 2.0])
    assert len(out) == 1
    assert out[0].label is PivotLabel.HH


def test_label_all_pivots_chains_anchors_like_vendored():
    """Same-type anchor advanced even when a pivot is unlabeled."""
    from compression_detection.dow import label_all_pivots

    p1 = Pivot(type=1, price=100.0, bar_index=1, ts=1)
    p2 = Pivot(type=1, price=101.0, bar_index=2, ts=2)   # EH vs p1
    p3 = Pivot(type=1, price=102.0, bar_index=3, ts=3)   # EH vs p2 (chained)
    atr = [None, 1.0, 1.0, 1.0]

    out = label_all_pivots([p1, p2, p3], atr)

    assert out[0][1] is None
    assert out[1][1] is PivotLabel.EH
    assert out[2][1] is PivotLabel.EH
