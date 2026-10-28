"""CKSO: Constrained Koopman Spectral Operator for echocardiographic cine dynamics.

Reference implementation of the method described in
"CKSO: A Constrained Koopman Spectral Operator for Spectral Estimation of
Echocardiographic Cine Dynamics" (IEEE SPL manuscript).
"""

from .layout import LatentLayout
from .operator import (
    CKSOOperator,
    DenseOperator,
    build_operator,
    omega_max,
    power_norms,
)
from .functionals import (
    trig_coefficients,
    eval_trig_phase,
    soft_extrema,
    exact_extrema,
    ejection_fraction,
)
from .frequency import detect_omega, FrequencyHead
from .model import CKSOModel

__version__ = "1.0.0"

__all__ = [
    "LatentLayout",
    "CKSOOperator",
    "DenseOperator",
    "build_operator",
    "omega_max",
    "power_norms",
    "trig_coefficients",
    "eval_trig_phase",
    "soft_extrema",
    "exact_extrema",
    "ejection_fraction",
    "detect_omega",
    "FrequencyHead",
    "CKSOModel",
]
