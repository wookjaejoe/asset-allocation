from __future__ import annotations

import pandas as pd
import quantstats as qs
from dataclasses import dataclass
from typing import *


@dataclass
class Portfolio:
    allocations: Dict[str, float]
    description: str

    @classmethod
    def from_series(cls, assets: list, description: str = "") -> Portfolio:
        counts = {}
        for asset in assets:
            if asset in counts:
                counts[asset] += 1
            else:
                counts[asset] = 1

        total = sum(counts.values())
        allocations = {k: v / total for k, v in counts.items()}

        return Portfolio(allocations, description)


class InvestmentStrategy:
    def __init__(
        self,
        chart: pd.DataFrame,
        month_chart: pd.DataFrame,
    ):
        self.chart = chart
        self.month_chart = month_chart
        self.chart_pct_change = self.chart.pct_change()
        self.month_chart_pct_change = self.month_chart.pct_change()
        self._portfolio = None

    @classmethod
    def get_name(cls):
        """
        :return: 전략의 이름
        """
        ...

    @classmethod
    def get_assets(cls) -> set:
        """
        :return: 전략의 관심 ticker 목록
        """
        ...

    def calc_portfolio(self) -> pd.Series:
        """
        :return: 날짜별 보유 ticker 목록
        """
        ...

    @property
    def portfolio(self) -> pd.Series:
        if self._portfolio is None:
            self._portfolio = self.calc_portfolio()

        return self._portfolio

    def backtest(self, output_prefix: str):
        df = pd.DataFrame()
        df["port"] = self.portfolio

        # 자산별 일별 수익률 계산
        df = pd.concat([df, self.chart_pct_change], axis=1)

        def calc_return(row: pd.Series):
            return sum([row[name] * weight for name, weight in row["port"].allocations.items()])

        # 포트폴리오 수익률 계산
        df["return"] = df.dropna().apply(calc_return, axis=1)
        df = df[["port", "return"]].dropna()
        df.to_csv(output_prefix + ".csv")

        pd.DataFrame({
            "allocations": df["port"].apply(lambda x: x.allocations),
            "description": df["port"].apply(lambda x: x.description),
        }).to_csv(output_prefix + "_port.csv")

        qs.reports.html(
            df["return"],
            benchmark=self.chart_pct_change["SPY"].reindex(df.index),
            output=output_prefix + ".html"
        )
