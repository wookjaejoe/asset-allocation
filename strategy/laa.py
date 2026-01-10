"""
✅ 2️⃣ BK-LAA 전략 (75% 고정 + 25% 동적 배분)
▶ **활용 지수 및 ETF 데이터:**
- **고정 자산 (75%)**: IWD, GLD, IEF
- **동적 자산 (25%)**: SPY, QQQ, SHY
▶ 투자 방식
- 75%는 고정 자산(IWD, GLD, IEF) → 변동 없음
- 25%는 SPY 1개월 이동평균 vs. 200일 이동평균 비교 후 결정
- SPY 1개월 평균이 200일 이동평균 위 → QQQ 투자
- SPY 1개월 평균이 200일 이동평균 아래 → 실업률 데이터 확인 후 QQQ or SHY 선택
▶ 비교 지수
- S&P 500 (SPY) vs. BK-LAA 전략 성과 비교
- 실업률 데이터 (미국 노동통계국 BLS)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from strategy.common import InvestmentStrategy, Portfolio

UNEMPLOYMENT_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"


class Assets:
    static = ["IWD", "GLD", "IEF"]
    dynamic = ["SPY", "QQQ", "SHY"]

    @classmethod
    def all(cls):
        return set(cls.static) | set(cls.dynamic)


class InvestmentStrategyLAA(InvestmentStrategy):
    @classmethod
    def get_name(cls):
        return "LAA"

    @classmethod
    def get_assets(cls) -> set:
        return Assets.all()

    def _load_unemployment(self) -> pd.DataFrame:
        """
        Returns a DataFrame with columns: Date, unemployment
        - Prefers local ./input/unemployment.csv if present (backward compatible)
        - Otherwise fetches UNRATE from FRED (no API key) and caches under .cache/
        """
        local_path = Path("./input/unemployment.csv")
        cache_path = Path(".cache/unemployment_unrate.csv")

        if local_path.exists():
            df = pd.read_csv(local_path)
        else:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.exists():
                df = pd.read_csv(cache_path)
            else:
                df = pd.read_csv(UNEMPLOYMENT_FRED_CSV_URL)
                df.to_csv(cache_path, index=False)

        # Normalize to the schema used by the rest of the strategy
        cols = {c.lower(): c for c in df.columns}
        if "date" in cols and "unrate" in cols:
            df = df.rename(columns={cols["date"]: "Date", cols["unrate"]: "unemployment"})
        elif "observation_date" in cols and "unrate" in cols:
            # FRED CSV commonly uses observation_date + UNRATE
            df = df.rename(columns={cols["observation_date"]: "Date", cols["unrate"]: "unemployment"})
        elif "reported" in cols and "value" in cols:
            df = df.rename(columns={cols["reported"]: "Date", cols["value"]: "unemployment"})

        if "Date" not in df.columns or "unemployment" not in df.columns:
            raise ValueError(f"Unemployment data has unexpected columns: {list(df.columns)}")

        df["Date"] = pd.to_datetime(df["Date"])
        df["unemployment"] = pd.to_numeric(df["unemployment"], errors="coerce")
        df = df.dropna(subset=["Date", "unemployment"]).sort_values("Date").reset_index(drop=True)
        return df[["Date", "unemployment"]]

    def calc_portfolio(self) -> pd.Series:
        unemployment = self._load_unemployment()

        spy_200_ma = self.chart["SPY"].rolling(window=200).mean().resample("ME").last()
        spy_200_ma = spy_200_ma.rename("spy_200_ma")
        spy_200_ma = spy_200_ma.rename(index={spy_200_ma.index[-1]: self.chart.index[-1]})

        df = pd.merge_asof(
            spy_200_ma,  # 기준 데이터 (월말 이동평균)
            unemployment,  # 매핑할 데이터 (실업률)
            on="Date",  # 기준 컬럼 (날짜)
            direction="backward"  # 과거 값 중 가장 가까운 값 선택
        )
        df = df.set_index("Date")
        df["unemployment_12_ma"] = df["unemployment"].rolling(window=12).mean().round(2)
        df["SPY"] = self.month_chart["SPY"]
        df["spy_200_ma"] = spy_200_ma
        df = df[["SPY", "spy_200_ma", "unemployment", "unemployment_12_ma"]]
        df["condition1"] = self.month_chart["SPY"] > spy_200_ma
        df["condition2"] = df["unemployment"] > df["unemployment_12_ma"]
        df["condition"] = df["condition1"] | df["condition2"]
        df["tickers"] = df["condition"].apply(lambda x: Assets.static + (["QQQ"] if x else ["SHY"]))

        df.to_csv("./output/laa.csv")

        result = df["tickers"]
        result = result.apply(lambda row: Portfolio.from_series(row))
        return result.fillna(np.nan).reindex(self.chart.index, method="ffill").shift(1).dropna()
