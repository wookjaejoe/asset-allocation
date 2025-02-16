"""
✅ 4️⃣ BK-채권동적자산배분 전략 (채권 모멘텀 투자)
▶ **활용 지수 및 ETF 데이터:**
- **채권 ETF:** SHY, IEF, TLT, TIP, LQD, HYG, BWX, EMB
▶ 투자 방식
- 8개의 채권 자산 중 최근 6개월 수익률이 높은 상위 3개 선택
- 선택된 채권 중 6개월 수익률이 0 이하이면 해당 비중만큼 현금 보유
▶ 비교 지수
- 미국 국채 (IEF vs. TLT) vs. BK-채권동적자산배분 전략 성과 비교
- 미국 하이일드 회사채 (HYG) vs. BK-채권동적자산배분 전략 성과 비교
"""
import pandas as pd
import yfinance as yf

from common import make_report


class Assets:
    bonds = ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "BWX", "EMB"]
    cash = "BIL"  # 현금 대체 자산

    @classmethod
    def all(cls):
        return set(cls.bonds + [cls.cash, "SPY"])


def run():
    chart = yf.download(Assets.all(), start="2007-01-01")
    chart = chart[[col for col in chart.columns if col[0] == "Close"]]
    chart.columns = [col[1] for col in chart.columns]
    chart = chart.dropna()

    month_chart = chart.resample("ME").last()
    month_chart = month_chart.rename(index={month_chart.index[-1]: chart.index[-1]})

    # 6개월 수익률 계산
    returns_6m = month_chart.pct_change(6)

    # 6개월 수익률 상위 3개 채권 선택
    top_bonds = returns_6m[Assets.bonds].apply(lambda x: list(x.nlargest(3).index), axis=1)

    # 선택된 채권 중 6개월 수익률이 0 이하인 경우 현금 보유
    def filter_negative_returns(row):
        x = [b for b in row["tickers"] if returns_6m.loc[row.name, b] > 0]
        return x + [Assets.cash for _ in range(3 - len(x))]

    tickers = top_bonds.to_frame("tickers").apply(filter_negative_returns, axis=1).rename("tickers")

    # 결과 저장
    df = pd.DataFrame({"tickers": tickers})
    df.to_csv("./output/bda.csv")

    tickers = tickers.dropna().reindex(chart.index, method="ffill").shift(1).dropna()
    make_report(chart, tickers, "./output/bda.html")


if __name__ == '__main__':
    run()
