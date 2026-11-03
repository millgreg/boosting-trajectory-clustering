"""Baselines for comparison with BTC."""

from .rfc import RFC
from .rfc_factorized import FactorizedRFC

__all__ = ["RFC", "FactorizedRFC"]
