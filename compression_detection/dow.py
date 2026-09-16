"""Dow Theory pivot labelling: HH/LH/EH and HL/LL/EL.

Each confirmed pivot is compared against the previous confirmed pivot of
the SAME TYPE using a second, independent ATR (length 14 by default) as
the significance threshold:

- new high > prev high + 1*ATR14  -> HH
- new high < prev high - 1*ATR14  -> LH
- in between                      -> EH
- new low < prev low - 1*ATR14    -> LL
- new low > prev low + 1*ATR14    -> HL
- in between                      -> EL

ATR14 flavor is configurable (close_only default — matches the zigzag's
close-only philosophy; true_range available). See atr.py.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import List, Optional

from .models import Pivot

# Labels for a high pivot
HH = "HH"
LH = "LH"
EH = "EH"
# Labels for a low pivot
HL = "HL"
LL = "LL"
EL = "EL"

HIGH_LABELS = {HH, LH, EH}
LOW_LABELS = {HL, LL, EL}


class PivotLabel(enum.Enum):
    HH = "HH"
    LH = "LH"
    EH = "EH"
    HL = "HL"
    LL = "LL"
    EL = "EL"

    @property
    def is_high(self) -> bool:
        return self.value in HIGH_LABELS

    @property
    def is_low(self) -> bool:
        return self.value in LOW_LABELS


@dataclass(frozen=True)
class LabeledPivot:
    pivot: Pivot
    label: PivotLabel

    def __repr__(self) -> str:
        return f"{self.label.value}@{self.pivot.price:.6g}(bar={self.pivot.bar_index})"


def label_high(new_price: float, prev_price: float, atr14: float) -> PivotLabel:
    """Label a new high pivot vs the previous high pivot."""
    if new_price > prev_price + atr14:
        return PivotLabel.HH
    if new_price < prev_price - atr14:
        return PivotLabel.LH
    return PivotLabel.EH


def label_low(new_price: float, prev_price: float, atr14: float) -> PivotLabel:
    """Label a new low pivot vs the previous low pivot."""
    if new_price < prev_price - atr14:
        return PivotLabel.LL
    if new_price > prev_price + atr14:
        return PivotLabel.HL
    return PivotLabel.EL


def label_pivot(
    pivot: Pivot,
    prev_same_type: Optional[Pivot],
    atr14: float,
) -> Optional[PivotLabel]:
    """Label a confirmed pivot against the previous same-type pivot.

    Returns None if there's no previous same-type pivot (can't label yet)
    or atr14 is not available.
    """
    if prev_same_type is None or atr14 is None or atr14 <= 0:
        return None
    if pivot.is_high:
        return label_high(pivot.price, prev_same_type.price, atr14)
    return label_low(pivot.price, prev_same_type.price, atr14)


def label_pivots(
    pivots: List[Pivot],
    atr14_series: List[Optional[float]],
) -> List[LabeledPivot]:
    """Label a full pivot list.

    atr14_series is indexed by bar; each pivot uses the ATR14 reading at
    its own bar (ATR14 known at the pivot bar — Wilder RMA through that
    bar). Unlabelable pivots (no prior same-type pivot yet, or ATR14 not
    ready) are omitted.

    """
    out: List[LabeledPivot] = []
    last_high: Optional[Pivot] = None
    last_low: Optional[Pivot] = None
    for p in pivots:
        atr14 = atr14_series[p.bar_index] if p.bar_index < len(atr14_series) else None
        prev = last_high if p.is_high else last_low
        label = label_pivot(p, prev, atr14) if atr14 is not None else None
        # the pivot becomes the new same-type anchor even when it can't be
        # labeled — otherwise the first pivot would poison the whole chain
        if p.is_high:
            last_high = p
        else:
            last_low = p
        if label is None:
            continue
        out.append(LabeledPivot(pivot=p, label=label))
    return out


def label_all_pivots(
    pivots: List[Pivot],
    atr14_series: List[Optional[float]],
):
    """Every pivot with its label or None, chronological, nothing omitted.

    Extension for the compression detector: the full sequence is required
    (initial reference pivots of each side are label-exempt per spec §5).
    Same-type anchors advance exactly like label_pivots, so a pivot that
    cannot be labeled still becomes the next anchor.
    """
    out = []
    last_high: Optional[Pivot] = None
    last_low: Optional[Pivot] = None
    for p in pivots:
        atr14 = atr14_series[p.bar_index] if p.bar_index < len(atr14_series) else None
        prev = last_high if p.is_high else last_low
        label = label_pivot(p, prev, atr14) if atr14 is not None else None
        if p.is_high:
            last_high = p
        else:
            last_low = p
        out.append((p, label))
    return out
