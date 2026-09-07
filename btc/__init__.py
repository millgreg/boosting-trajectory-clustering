"""
Boosting Trajectory Clustering (BTC)
A linear-time representation learning framework for supervised clustering using gradient boosting trajectories.
"""

__version__ = "0.1.0"

from .core import BTC, MultiTaskBTC, auto_select_depth

__all__ = ['BTC', 'MultiTaskBTC', 'auto_select_depth']
