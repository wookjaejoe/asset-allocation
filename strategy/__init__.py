from logger import logger
import pandas as pd
from .common import InvestmentStrategy
import yfinance as yf

from .baa import InvestmentStrategyBAA
from .bda import InvestmentStrategyBDA
from .laa import InvestmentStrategyLAA
from .mdm import InvestmentStrategyMDM


class InvestmentStrategyMerged(InvestmentStrategy):
    strategy_types = [
        InvestmentStrategyBAA,
        InvestmentStrategyBDA,
        InvestmentStrategyLAA,
        InvestmentStrategyMDM
    ]

    @classmethod
    def get_assets(cls) -> set:
        assets = set.union(*(strategy.get_assets() for strategy in cls.strategy_types))
        logger.info(f"Assets to fetch: {assets}")
        return assets

    def calc_portfolio(self) -> pd.Series:
        ports = []
        for strategy_type in self.strategy_types:
            logger.info(f"Processing strategy: {strategy_type.get_name()}")
            strategy = strategy_type(chart=self.chart, month_chart=self.month_chart)
            strategy.backtest(f"./output/{strategy.get_name()}")
            port = strategy.portfolio
            ports.append(port)

        merged_portfolio = pd.concat(ports).groupby(level=0).sum()
        sorted_portfolio = merged_portfolio.apply(lambda x: sorted(x))
        logger.info("Portfolio merging complete.")
        return sorted_portfolio


def fetch_charts() -> (pd.DataFrame, pd.DataFrame):
    assets = InvestmentStrategyMerged.get_assets()
    chart = yf.download(assets, start="2007-01-01")
    chart = chart[[col for col in chart.columns if col[0] == "Close"]]
    chart.columns = [col[1] for col in chart.columns]
    chart = chart.dropna()

    logger.info("Market data successfully fetched and processed.")

    month_chart = chart.resample("ME").last()
    month_chart = month_chart.rename(index={month_chart.index[-1]: chart.index[-1]})

    return chart, month_chart
