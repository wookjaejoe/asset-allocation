from logger import logger
import pandas as pd
from .common import InvestmentStrategy, Portfolio
from lib.price_cache import load_prices_with_cache

from .baa import InvestmentStrategyBAA
from .bda import InvestmentStrategyBDA
from .laa import InvestmentStrategyLAA
from .mdm import InvestmentStrategyMDM


class InvestmentStrategyIntegration(InvestmentStrategy):
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

    def __init__(self, chart: pd.DataFrame, month_chart: pd.DataFrame, run_backtests: bool = True):
        super().__init__(chart=chart, month_chart=month_chart)
        self.run_backtests = run_backtests

    def calc_portfolio(self) -> pd.Series:
        ports = []
        for strategy_type in self.strategy_types:
            logger.info(f"Processing strategy: {strategy_type.get_name()}")
            strategy = strategy_type(chart=self.chart, month_chart=self.month_chart)
            if self.run_backtests:
                strategy.backtest(f"./output/{strategy.get_name()}")
            port = strategy.portfolio

            df = pd.DataFrame({
                "port": strategy.portfolio,
                "strategy": strategy.get_name()  # 전체에 동일한 값으로 들어감
            })
            ports.append(df)

        merged_portfolio = pd.concat(ports).groupby(level=0).apply(self.merge_portfolio)
        logger.info("Portfolio merging complete.")
        return merged_portfolio

    @staticmethod
    def merge_portfolio(ports: pd.DataFrame) -> Portfolio:
        allocations = {}
        description = ""
        for idx, row in ports.iterrows():
            for name, weight in row["port"].allocations.items():
                if name in allocations:
                    allocations[name] += weight
                else:
                    allocations[name] = weight

            description += str(row["strategy"]) + ": " + str(row["port"].allocations) + "\n"

        allocations = {
            k: v / len(ports)
            for k, v in sorted(allocations.items(), key=lambda item: item[1], reverse=True)
        }
        return Portfolio(allocations, description)


def fetch_charts() -> (pd.DataFrame, pd.DataFrame):
    assets = InvestmentStrategyIntegration.get_assets()
    chart = load_prices_with_cache(assets, start="2007-01-01", end=None)
    chart = chart.dropna(how="all").dropna(axis=1, how="all").sort_index()

    logger.info("Market res successfully fetched and processed.")

    month_chart = chart.resample("ME").last()
    month_chart = month_chart.rename(index={month_chart.index[-1]: chart.index[-1]})

    return chart, month_chart
