import torch
import torch.nn as nn
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# 1. Time Covariate Encoder
# ═══════════════════════════════════════════════════════════════

class TimeCovariateEncoder:
    """
    Encodes a pd.DatetimeIndex into a Z-score normalized time covariate matrix.

    Covariates:
        - weekday : 0 (Mon) ~ 4 (Fri)
        - month   : 1 ~ 12
        - quarter : 1 ~ 4

    Usage:
        encoder = TimeEncoder()
        encoder.fit(train_dates)          # fit Z-score stats on training set
        tensor = encoder.transform(dates) # apply to any date array
    """

    COVARIATE_NAMES = ["weekday", "month", "quarter"]

    def __init__(self):
        self.means_  = None   # shape: (3,)
        self.stds_   = None   # shape: (3,)
        self._fitted = False

    # ─────────────────────────────────────────
    @staticmethod
    def _extract(dates: pd.DatetimeIndex) -> np.ndarray:
        """Extract raw covariate matrix from dates. Returns shape (N, 3)."""
        weekday = dates.weekday.astype(np.float32)      # 0~4
        month   = dates.month.astype(np.float32)        # 1~12
        quarter = dates.quarter.astype(np.float32)      # 1~4
        return np.stack([weekday, month, quarter], axis=1)  # (N, 3)

    # ─────────────────────────────────────────
    def fit(self, train_dates: pd.DatetimeIndex) -> "TimeCovariateEncoder":
        """
        Compute mean and std from training dates.

        Args:
            train_dates: pd.DatetimeIndex of training period.

        Returns:
            self (for method chaining)
        """
        raw = self._extract(train_dates)           # (N, 3)
        self.means_ = raw.mean(axis=0)             # (3,)
        self.stds_  = raw.std(axis=0) + 1e-8      # (3,)  add epsilon for safety
        self._fitted = True
        return self

    # ─────────────────────────────────────────
    def transform(self, dates: pd.DatetimeIndex) -> torch.Tensor:
        """
        Transform dates to Z-score normalized covariate tensor.

        Args:
            dates: pd.DatetimeIndex of any split (train / val / test).

        Returns:
            torch.Tensor of shape (N, 3), dtype=float32.
        """
        if not self._fitted:
            raise RuntimeError("TimeEncoder must be fit() before transform().")

        raw        = self._extract(dates)                     # (N, 3)
        normalized = (raw - self.means_) / self.stds_         # (N, 3)
        return torch.tensor(normalized, dtype=torch.float32)

    # ─────────────────────────────────────────
    def fit_transform(self, train_dates: pd.DatetimeIndex) -> torch.Tensor:
        """Convenience: fit and transform in one call."""
        return self.fit(train_dates).transform(train_dates)

    # ─────────────────────────────────────────
    def summary(self) -> None:
        """Print fitted statistics."""
        if not self._fitted:
            print("Not fitted yet.")
            return
        print(f"{'Covariate':<12} {'Mean':>8} {'Std':>8}")
        print("─" * 30)
        for name, mu, sigma in zip(self.COVARIATE_NAMES, self.means_, self.stds_):
            print(f"{name:<12} {mu:>8.4f} {sigma:>8.4f}")
