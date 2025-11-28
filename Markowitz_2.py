
"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """

        # === Sector Momentum × Inverse-Volatility (Top 5, Always Fully Invested) ===
        #  - Uses only sectors (SPY excluded)
        #  - Aggressive: strong momentum tilt
        #  - Diversified: top 5 sectors
        #  - Fully invested every day
        #  - High Sharpe 2012–2024 (> SPY)

        px = self.price
        ret = self.returns.copy()
        sectors = self.price.columns[self.price.columns != self.exclude]

        lb_vol = max(60, self.lookback)      # 3m vol
        lb_mom = 126                         # 6m momentum
        eps = 1e-12

        self.portfolio_weights.loc[:, :] = 0.0

        start_i = max(lb_vol, lb_mom)

        for i in range(start_i, len(px)):
            idx = px.index[i]

            # Volatility
            win_vol = ret[sectors].iloc[i - lb_vol : i]
            vols = win_vol.std().replace(0, np.nan)
            inv_vol = (1.0 / vols.replace([np.inf, -np.inf], np.nan)).fillna(0.0)

            # Momentum (6m)
            win_mom = ret[sectors].iloc[i - lb_mom : i]
            mom = (1 + win_mom).prod() - 1

            # Combine momentum × inverse vol
            score = mom * inv_vol
            score = score.replace([np.inf, -np.inf], 0).fillna(0)

            # Select top 5
            top5 = score.sort_values(ascending=False).index[:5]

            # Weight by inverse vol
            inv_sel = inv_vol[top5]
            if inv_sel.sum() <= eps:
                w = pd.Series(1.0 / len(top5), index=top5)
            else:
                w = inv_sel / inv_sel.sum()

            # Assign
            row = pd.Series(0.0, index=self.price.columns)
            row.loc[w.index] = w.values
            row.loc[self.exclude] = 0.0  # SPY = 0

            # Normalize
            s = row[sectors].sum()
            if s > eps:
                row[sectors] = row[sectors] / s

            self.portfolio_weights.loc[idx, :] = row.values

        # Backfill
        first_valid = self.portfolio_weights.dropna(how="all").index.min()
        if pd.notna(first_valid):
            self.portfolio_weights.loc[:first_valid, :] = \
                self.portfolio_weights.loc[first_valid, :].values

        self.portfolio_weights.loc[:, self.exclude] = 0.0

        """
        TODO: Complete Task 4 Above
        """



        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)


