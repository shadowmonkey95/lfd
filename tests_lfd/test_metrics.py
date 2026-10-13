"""BL-0005: DiagonalAverage / PastConceptsAverage on a hand-made matrix."""
import math

import numpy as np
import pytest
from pyclad.metrics.continual.average_continual import ContinualAverage

from lfd import registry
from lfd.metrics import DiagonalAverage, PastConceptsAverage

# M[learned][evaluated]; upper triangle filled with junk that must be ignored
M = [[0.9, 0.1, 0.1],
     [0.6, 0.8, 0.1],
     [0.5, 0.4, 0.7]]


def test_diagonal_average():
    assert DiagonalAverage().compute(M) == pytest.approx((0.9 + 0.8 + 0.7) / 3)


def test_past_concepts_average():
    assert PastConceptsAverage().compute(M) == pytest.approx((0.6 + 0.5 + 0.4) / 3)


def test_split_recombines_to_continual_average():
    n, p = 3, 3
    combined = (n * DiagonalAverage().compute(M) + p * PastConceptsAverage().compute(M)) / (n + p)
    assert combined == pytest.approx(ContinualAverage().compute(M))


def test_upper_triangle_ignored():
    m2 = [row[:] for row in M]
    m2[0][2] = m2[0][1] = m2[1][2] = 0.0
    assert DiagonalAverage().compute(m2) == DiagonalAverage().compute(M)
    assert PastConceptsAverage().compute(m2) == PastConceptsAverage().compute(M)


def test_nan_cells_skipped():
    m2 = [row[:] for row in M]
    m2[1][1] = np.nan
    m2[2][0] = np.nan
    assert DiagonalAverage().compute(m2) == pytest.approx((0.9 + 0.7) / 2)
    assert PastConceptsAverage().compute(m2) == pytest.approx((0.6 + 0.4) / 2)


def test_single_concept_has_no_past():
    assert DiagonalAverage().compute([[0.8]]) == pytest.approx(0.8)
    assert math.isnan(PastConceptsAverage().compute([[0.8]]))


def test_rectangular_rejected():
    with pytest.raises(ValueError):
        DiagonalAverage().compute([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    with pytest.raises(ValueError):
        PastConceptsAverage().compute([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


def test_registered_and_existing_metrics_kept():
    names = [c().name() for c in registry.SUMMARY_METRICS]
    assert names[:5] == ["ContinualAverage", "BackwardTransfer", "ForwardTransfer", "ForgettingMeasureStrict",
                         "FinalStepAverage"]
    assert {"DiagonalAverage", "PastConceptsAverage"} <= set(names)
