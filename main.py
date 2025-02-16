from strategy import *
from strategy.common import InvestmentStrategy
import yfinance as yf
import pandas as pd


class InvestmentStrategyMerged(InvestmentStrategy):
    strategy_types = [
        InvestmentStrategyBAA,
        InvestmentStrategyBDA,
        InvestmentStrategyLAA,
        InvestmentStrategyMDM
    ]

    @classmethod
    def get_assets(cls) -> set:
        return set.union(*(strategy.get_assets() for strategy in cls.strategy_types))

    def get_portfolio(self) -> pd.Series:
        ports = []
        for strategy_type in self.strategy_types:
            strategy = strategy_type(chart=self.chart, month_chart=self.month_chart)
            port = strategy.get_portfolio()
            ports.append(port)

        return pd.concat(ports).groupby(level=0).sum()


def main():
    chart = yf.download(InvestmentStrategyMerged.get_assets(), start="2007-01-01")
    chart = chart[[col for col in chart.columns if col[0] == "Close"]]
    chart.columns = [col[1] for col in chart.columns]
    chart = chart.dropna()

    month_chart = chart.resample("ME").last()
    month_chart = month_chart.rename(index={month_chart.index[-1]: chart.index[-1]})
    strategy = InvestmentStrategyMerged(chart=chart, month_chart=month_chart)
    port = strategy.get_portfolio()
    port = port.apply(lambda x: sorted(x))
    weight = port.apply(lambda x: pd.Series(x).value_counts(normalize=True).to_dict())
    print(weight)


if __name__ == '__main__':
    main()
