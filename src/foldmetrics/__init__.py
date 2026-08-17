"""foldmetrics: unified confidence metrics for structure-prediction models.

Ingests outputs from AlphaFold2/3, ColabFold, Boltz, Chai-1 and Protenix and
computes pTM, ipTM, pLDDT, ipLDDT, PAE statistics, ipSAE, pDockQ, pDockQ2 and
LIS for single models or whole batches.
"""

__version__ = "0.1.0"

from foldmetrics.api import (
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
    "compute_all",
    "compute_interfaces",
    "compute_summary",
    "evaluate",
    "evaluate_full",
    "evaluate_interfaces",
    "find_contacts",
    "load_predictions",
]
