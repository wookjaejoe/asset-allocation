"""
퀀트 투자 4가지 전략 백테스트 분석 (정확한 투자 방식 포함)
✅ 1️⃣ BK-BAA 전략 (동적 자산 배분)
▶ **활용 지수 및 ETF 데이터:**
- **공격자산:** QQQ, SPY, EFA, EEM, IWM, VGK, EWJ, VWO, VEA, VNQ, DBC, GLD
- **카나리아 자산:** SPY, VWO, VEA, BND
- **안전자산:** SHY, IEF, TLT, TIP, LQD, HYG, BIL
▶ 투자 방식 (정확한 프로세스)
📌 **Step 1: 카나리아 자산 평가 (시장 강세 여부 판단)**
- SPY, VWO, VEA, BND 4개의 자산 모멘텀 스코어 계산 (1-3-6-12 법칙 적용)

- **모멘텀 스코어 계산:** (최근 1개월 수익률 × 12) + (최근 3개월 수익률 × 4) + (최근 6개월 수익률 × 3) + (최근 12개월 수익률 × 1)
- 4개 자산 모두 모멘텀 스코어 > 0이면 → 공격자산 유지
- 4개 중 하나라도 모멘텀 스코어 ≤ 0이면 → 안전자산으로 전환
📌 **Step 2: 공격자산 선정 (12개 중 상위 6개 선택)**
- 공격자산 12개 중 모멘텀 스코어(1-3-6-12 법칙) 기반으로 상위 6개 선정
- 각 16.67%씩 균등 배분
📌 **Step 3: 안전자산 선정 (12개월 이동평균선 대비 최고 수익 3개 선택)**
- 안전자산 7개 중 12개월 이동평균선보다 높은 자산만 고려
- 12개월 이동평균선보다 높은 안전자산 중, 최근 12개월 수익률이 높은 상위 3개 균등 투자
- 3개 미만이면 부족분만큼 현금 보유 (0%, 33%, 66%, 100%)
▶ 비교 지수
- S&P 500 (SPY) vs. BK-BAA 전략 성과 비교
- 나스닥 100 (QQQ) vs. BK-BAA 성과 비교
"""

import numpy as np
import pandas as pd

from strategy.common import InvestmentStrategy

MOMENTUM_WEIGHTS = {period_month: 12 / period_month for period_month in [1, 3, 6, 12]}


class Assets:
    aggressive = ["QQQ", "SPY", "EFA", "EEM", "IWM", "VGK", "EWJ", "VWO", "VEA", "VNQ", "DBC", "GLD"]  # 공격 자산
    canaria = ["SPY", "VWO", "VEA", "BND"]  # 카나리아 자산
    defensive = ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "BIL"]  # 안전 자산

    @classmethod
    def all(cls):
        return set(cls.aggressive) | set(cls.canaria) | set(cls.defensive)


class InvestmentStrategyBAA(InvestmentStrategy):
    """
    BAA - Bold Asset Allocation
    """

    @classmethod
    def get_assets(cls) -> set:
        return Assets.all()

    def get_portfolio(self) -> pd.Series:
        mmt = self.month_chart[list(set(Assets.canaria + Assets.aggressive))].apply(
            self.momentum_score,
            axis=1
        ).dropna()

        aggressive_signal = mmt[Assets.canaria].ge(0).all(axis=1)

        # 1. 공격적 자산 선택
        aggressive_selection: pd.Series = (
            mmt[aggressive_signal][Assets.aggressive]
            .apply(lambda row: row.nlargest(6).index.tolist(), axis=1)
        )

        # 2. 방어적 자산 선택
        defensive_selection: pd.Series = (
            self.month_chart.reindex(aggressive_signal.index)[~aggressive_signal]
            .apply(self.select_defensive, axis=1)
        )

        # 3. 공격적/방어적 선택을 병합하고 정렬
        result: pd.Series = pd.concat([aggressive_selection, defensive_selection]).sort_index()
        result = result.fillna(np.nan)
        result = result.reindex(self.chart.index, method="ffill")
        result = result.shift(1)
        result = result.dropna()
        return result

    def rate_of_return(self, row: pd.Series, months: int) -> pd.Series:
        """
        특정 시점 n개월 수익률 계산
        :param row: 특정 시점 주가
        :param months: 개월 수
        :return: 자산별 수익률
        """
        pos = self.month_chart[row.index].index.get_loc(row.name)
        assert pos - months >= 0, "Not enough data."
        return row / self.month_chart[row.index].iloc[pos - months] - 1

    def momentum_score(self, row: pd.Series):
        """
        특정 시점 모멘텀 스코어 계산
        :param row: 특정 시점 주가
        :return: 자산별 모멘텀 스코어
        """

        try:
            return sum([self.rate_of_return(row, m) * w for m, w in MOMENTUM_WEIGHTS.items()])
        except AssertionError:
            return pd.Series(index=row.index, name=row.name, data=np.nan)

    def select_defensive(self, row: pd.Series):
        """
        방어 자산 중 12개월 수익률 가장 높은 3개 자산 반환
        :param row: 특정 시점 주가
        :return: 선택된 3가지 방어 자산
        """
        pos = self.month_chart.index.get_loc(row.name)
        current = row[Assets.defensive]
        ma = self.month_chart[Assets.defensive].iloc[pos - 11:pos + 1].mean()  # 최근 12개월 이동 평균

        # 12개월 수익률
        return [
            "BIL" if current[name] < ma[name] else name for name, _ in
            self.rate_of_return(current, 12).nlargest(3).items()
        ]
