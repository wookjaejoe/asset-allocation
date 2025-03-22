from strategy import *


def main():
    chart, month_chart = fetch_charts()
    strategy = InvestmentStrategyIntegration(chart=chart, month_chart=month_chart)
    strategy.backtest("output")


if __name__ == '__main__':
    main()
