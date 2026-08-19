"""foldmetrics: unified confidence metrics for structure-prediction models.

Ingests outputs from AlphaFold2/3, ColabFold, Boltz, Chai-1 and Protenix and
computes pTM, ipTM, pLDDT, ipLDDT, PAE statistics, ipSAE, pDockQ, pDockQ2 and
LIS for single models or whole batches.
"""

__version__ = "0.1.7"

from foldmetrics.api import (
    aggregate_by_target,
    evaluate,
    evaluate_full,
    evaluate_interfaces,
    load_predictions,
)
from foldmetrics.metrics import (
    compute_all,
    compute_interfaces,
    compute_summary,
    find_contacts,
)
from foldmetrics.models import Prediction, Token

__all__ = [
    "__version__",
    "Prediction",
    "Token",
    "aggregate_by_target",
    "compute_all",
    "compute_interfaces",
    "compute_summary",
    "evaluate",
    "evaluate_full",
    "evaluate_interfaces",
    "find_contacts",
    "load_predictions",
]
