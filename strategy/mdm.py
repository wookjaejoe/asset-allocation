"""
✅ 3️⃣ BK-변형듀얼모멘텀 전략 (공격 vs 안전)
▶ **활용 지수 및 ETF 데이터:**
- **공격자산:** SPY, EFA
- **안전자산:** SHY, IEF, TLT, TIP, LQD, HYG, BWX, EMB
▶ 투자 방식
- SPY 12개월 수익률 확인 → 공격자산(100%) or 안전자산(3개 균등) 결정
- SPY의 12개월 수익률이 0 초과인 경우 → SPY 또는 EFA 중 높은 수익률 자산 100% 투자
- SPY 12개월 수익률이 0 이하이면, 안전자산(8개) 중 6개월 수익률 상위 3개 선택
▶ 비교 지수
- S&P 500 (SPY) vs. BK-변형듀얼모멘텀 전략 성과 비교

"""

import numpy as np
import pandas as pd
import yfinance as yf
from common import make_report


class Assets:
    aggressive = ["SPY", "EFA"]
    defensive = ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "BWX", "EMB"]

    @classmethod
    def all(cls):
        return set(cls.aggressive) | set(cls.defensive)


def run():
    chart = yf.download(Assets.all(), start="2007-01-01")
    chart = chart[[col for col in chart.columns if col[0] == "Close"]]
    chart.columns = [col[1] for col in chart.columns]
    chart = chart.dropna()

    month_chart = chart.resample("ME").last()
    month_chart = month_chart.rename(index={month_chart.index[-1]: chart.index[-1]})

    # 12개월 및 6개월 수익률 계산
    returns_12m = month_chart.pct_change(12)
    returns_6m = month_chart.pct_change(6)

    # SPY 12개월 수익률 확인
    spy_return = returns_12m["SPY"]
    is_aggressive = spy_return > 0

    # 공격자산 선택 (SPY vs EFA 중 12개월 수익률이 높은 자산 100%)
    aggressive_choice = returns_12m[Assets.aggressive].idxmax(axis=1)

    # 안전자산 선택 (6개월 수익률 상위 3개 균등 투자)
    defensive_choice = returns_6m[Assets.defensive].apply(lambda x: list(x.nlargest(3).index), axis=1)

    # 최종 포트폴리오 결정
    tickers = pd.concat([aggressive_choice[is_aggressive], defensive_choice[~is_aggressive]]).sort_index()
    tickers = tickers.dropna()

    # 결과 저장
    df = pd.DataFrame({
        "SPY_12m_return": spy_return,
        "tickers": tickers
    })
    df.to_csv("./output/mdm.csv")

    tickers = tickers.fillna(np.nan).reindex(chart.index, method="ffill").shift(1).dropna()
    make_report(chart, tickers, "./output/mdm.html")


if __name__ == '__main__':
    run()
