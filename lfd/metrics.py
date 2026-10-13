"""Extra continual metrics (BL-0005). Split pyCLAD's ContinualAverage into its two parts.

pyCLAD's ``ContinualAverage`` is the mean of the diagonal plus the strictly-lower triangle of the
concept-level matrix ``M[learned][evaluated]``. Two questions get mixed in that one number:

* ``DiagonalAverage``: mean of ``M[i][i]``. How well the model does on the concept it has *just* learned.
  If continual strategies "accumulate knowledge", this should be higher than ``Naive``.
* ``PastConceptsAverage``: mean of ``M[i][j]`` for ``j < i``. Retention on concepts learned *earlier* only.

With ``N`` concepts and ``P = N(N-1)/2`` past cells:
``ContinualAverage == (N * DiagonalAverage + P * PastConceptsAverage) / (N + P)`` (when no cell is NaN).

Both metrics skip NaN cells. ``PastConceptsAverage`` is NaN for a 1-concept stream, because there is no
past concept to measure. Both accept only square matrices (one concept per training step).
"""
from __future__ import annotations

import numpy as np
from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    SummarizedMetric,
    is_nan,
    validate_square_matrix,
)
from pyclad.metrics.continual.schedule_aware_new_task_acquisition import ScheduleAwareNewTaskAcquisition


class DiagonalAverage(SummarizedMetric):
    """Mean of ``M[i][i]``. Delegates to pyCLAD's ``ScheduleAwareNewTaskAcquisition`` with the identity schedule."""

    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        validate_square_matrix(metric_matrix, self.name())
        return ScheduleAwareNewTaskAcquisition().compute(metric_matrix, list(range(len(metric_matrix))))

    def name(self) -> str:
        return "DiagonalAverage"


class PastConceptsAverage(SummarizedMetric):
    """Mean of the strictly-lower triangle ``M[i][j], j < i``. NaN if there are no past cells."""

    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        validate_square_matrix(metric_matrix, self.name())
        values = [float(metric_matrix[i][j]) for i in range(len(metric_matrix)) for j in range(i)
                  if not is_nan(metric_matrix[i][j])]
        return float(np.mean(values)) if values else float("nan")

    def name(self) -> str:
        return "PastConceptsAverage"
