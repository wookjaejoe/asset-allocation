import pandas as pd
import quantstats as qs


class InvestmentStrategy:
    def __init__(
        self,
        chart: pd.DataFrame,
        month_chart: pd.DataFrame,
    ):
        self.chart = chart
        self.month_chart = month_chart

    def select_tickers(self) -> pd.Series:
        """
        :return: 날짜별 보유 ticker 목록
        """
        ...


def make_report(chart, tickers, output):
    port = pd.DataFrame()
    port["tickers"] = tickers

    # 자산별 일별 수익률 계산
    returns = chart.rolling(2).apply(lambda row: row.iloc[1] / row.iloc[0] - 1)
    port = pd.concat([port, returns], axis=1)

    # 포트폴리오 수익률 계산
    port["return"] = port.dropna().apply(lambda row: row[row["tickers"]].mean(), axis=1)
    port = port[["tickers", "return"]].dropna()

    qs.reports.html(
        port["return"],
        benchmark=returns["SPY"],
        output=output
    )
