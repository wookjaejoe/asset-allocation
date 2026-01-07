# 리밸런스 랭킹 백테스트 안내

## 전략이 하는 일
- **대상**: S&P 500 편입 종목. 리밸런스 시점의 실제 편입 종목만 사용합니다.
- **랭킹 기준**: 직전 `lookback` 거래일 누적 수익률을 계산해 순위 매깁니다.
  - `mode=head`: 상승률 상위 종목에 동일 비중 투자(모멘텀).
  - `mode=tail`: 하락률 하위 종목에 동일 비중 투자(리버전/저가 반등 기대).
- **편입 종목 수**: 상위/하위에서 `top`개를 뽑아 모두 똑같은 비중으로 담습니다.
- **리밸런스 주기**: 기본 1개월, 3개월·6개월 간격도 선택 가능합니다.
- **비교 기준(벤치마크)**: 같은 기간 유니버스 전체를 동일 비중으로 묶은 “시장 평균” 수익률과 비교합니다.
- **위험 필터**: 일일 수익률이 너무 큰(설정값 `max_daily_change`, 기본 ±100%) 종목은 해당 기간에서 제외합니다.

## 어떻게 계산되는지 (흐름만 간단히)
1) 리밸런스할 날짜(월말/3개월/6개월 간격)를 정합니다.
2) 각 날짜 직전에 `lookback` 기간 수익률을 계산해 순위를 매기고 `top`개를 고릅니다.
3) 다음 리밸런스까지 보유하며, 매일의 수익률을 더해 구간 수익률을 계산합니다.
4) 같은 구간에 유니버스 전체를 동일 비중으로 가정해 “시장 평균” 수익률도 계산합니다.
5) 매 구간의 포트폴리오 수익률, 벤치마크 수익률, 둘의 차이(초과수익)를 기록하고, 모든 구간을 이어 붙여 일별/월별 결과를 만듭니다.

## 결과 파일(.output 폴더)
각 조합은 `.output/rank_<mode>_lk<lookback>_top<top>_rbm<리밸런스개월>/` 폴더에 저장됩니다.
- `rank_backtest.csv`: 일별 포트폴리오 수익률(그래프, 변동성, 낙폭 계산에 사용).
- `rank_monthly.csv`: 리밸런스 구간별 요약(파일명은 기존 관례). 기간, 보유 종목·매수/매도 가격, `return`(포트), `benchmark_return`(시장 평균), `active_return`(초과 수익)이 포함됩니다.
- `rank_report.html`: 요약 리포트(수익률·변동성·낙폭 등 지표와 차트). 브라우저로 열면 됩니다.
- 여러 조합을 한 번에 돌린 후에는 `.output/rank_summary.csv`와 `.output/rank_summary.md`가 생겨, 모든 조합의 주요 지표(CAGR, 변동성, 최대낙폭, 주기별 초과수익 등)를 한눈에 비교할 수 있습니다.

## 결과 읽는 법
- `return`: 해당 리밸런스 구간(1/3/6개월) 포트 수익률.
- `benchmark_return`: 같은 기간 S&P500 유니버스 전체를 균등 비중으로 잡았을 때 수익률(시장 평균).
- `active_return = return - benchmark_return`: 초과 수익. 양수면 시장보다 나음.
- `rank_summary.csv`의 대표 지표:
  - `CAGR`: 연복리 수익률(장기 성과).
  - `ann_vol`: 연환산 변동성(일별 수익률 기준 흔들림).
  - `max_drawdown`: 최대 낙폭(얼마나 크게 빠질 수 있었는지).
  - `monthly_active_mean`: 리밸런스 주기별 초과 수익 평균(지속적으로 시장을 이겼는지).
해석 시 수익률만 보지 말고 변동성·낙폭도 함께 비교하세요. 리밸런스 주기가 다르면 표준편차(변동성) 비교가 공정하지 않을 수 있으니 같은 주기끼리 보거나 기간 보정된 지표를 참고하세요.

## 리스크와 한계
- **거래비용/세금/슬리피지 미반영**: 백테스트는 이 비용을 고려하지 않습니다. 실전 수익은 더 낮을 수 있습니다.
- **데이터 품질**: 야후 파이낸스 종가를 사용합니다. 지연·수정·결측 가능성이 있습니다.
- **구성 종목 변화**: 과거 시점의 실제 S&P 500 편입 종목을 사용하므로, 현재 지수와 다를 수 있습니다.
- **극단 변동 필터**: `max_daily_change`를 넘는 종목은 제외되어 편입 종목 수가 줄 수 있으며, 그만큼 분산 효과가 떨어질 수 있습니다.
- **동일 비중**: 종목마다 같은 비중으로 담습니다. 시가총액 가중이나 위험 조정 비중은 적용하지 않습니다.

## 도움말
- 리포트는 HTML 파일을 더블클릭해 브라우저로 열면 됩니다.
- CSV는 엑셀·스프레드시트로 열어도 됩니다.

## `rank_summary.csv` 주요 컬럼 설명
- `label`: 실행 조합 이름(예: rank_head_lk60_top20_rbm3).
- `mode`: head(상승 상위 매수) / tail(하락 하위 매수).
- `lookback`: 순위를 매길 때 참고한 거래일 수.
- `lookback_short` / `lookback_mid`: 혼합 전략에서 사용하는 단기/중기 창(해당될 때만 값이 채워짐).
- `top`: 한 번에 담는 종목 수(모두 동일 비중).
- `rebalance_months`: 리밸런스 간격(1=매월, 3=분기, 6=반기).
- `weight_mom` / `weight_rev`: 혼합 전략에서 모멘텀/리버설 스코어 가중치(해당될 때만 값이 채워짐).
- `daily_path` / `monthly_path`: 생성된 상세 CSV 파일 경로.
- `daily_start` / `daily_end`: 일별 수익률 시작/종료 날짜.
- `days`: 일별 수익률 데이터 포인트 수.
- `cagr`: 연복리 수익률(장기 성과).
- `ann_vol`: 연환산 변동성(수익률의 흔들림 정도).
- `max_drawdown`: 최대 낙폭(최고점 대비 얼마나 크게 빠졌는지).
- `months`: 리밸런스 구간 개수(월·분기·반기).
- `period_return_mean`: 리밸런스 구간(선택한 주기) 포트 수익률 평균.
- `period_benchmark_mean`: 같은 구간 유니버스 균등가중 벤치마크 수익률 평균.
- `period_spy_mean`: 같은 구간 SPY 수익률 평균.
- `period_active_mean`: 유니버스 벤치마크 대비 초과 수익 평균(`return - benchmark_return`).
- `period_active_spy_mean`: SPY 대비 초과 수익 평균(`return - spy_return`).
- `period_benchmark_cagr`: 유니버스 벤치마크 연복리 수익률(구간 수익률을 누적 후 전체 기간 기준).
- `period_spy_cagr`: SPY 연복리 수익률(구간 수익률을 누적 후 전체 기간 기준).
- `last_period_return`: 마지막 리밸런스 구간 포트 수익률.
- `last_period_benchmark_return`: 마지막 리밸런스 구간 유니버스 벤치마크 수익률.
- `last_period_spy_return`: 마지막 리밸런스 구간 SPY 수익률.
- `last_period_active_return`: 마지막 구간 유니버스 대비 초과 수익.
- `last_period_active_spy_return`: 마지막 구간 SPY 대비 초과 수익.

## `rank_report.html`(요약 리포트) 안내
TODO

## 분석 스크립트(룩백/Top 경향)
- 집계 파일(.output/rank_summary.csv)이 있으면 `python scripts/analyze_rank_summary.py --summary .output/rank_summary.csv --output-md .output/rank_analysis.md`로 룩백·Top·리밸런스 경향 리포트(Markdown)를 만듭니다.
- 옵션: `--mode head`(또는 tail/mix)로 모드 필터, `--compare-tops 10 50`으로 Top 크기 비교쌍 변경.
- 산출: `.output/rank_analysis.md`에 룩백별 평균, Top 크기 민감도, Sharpe-like 순위, 리밸런스 안정성(평균/표준편차) 테이블을 포함합니다.

---

## 매일 08:00 이메일(운영 리포트) 스펙(초안)
### 이 리포트는 무엇이고, 어떻게 계산되나요?
- **무엇을 알려주나**: “오늘 08:00 KST 기준으로 S&P500에서 어떤 종목을 동일가중으로 보유해야 하는지”를 Head(모멘텀) / Tail(리버설) 각 lookback별 테이블로 보여줍니다.
- **어떻게 계산하나**: 최신 거래일 종가를 `current_price`, `lookback` 영업일 전 종가를 `lookback_price`로 두고, `(current / lookback) - 1`을 lookback_return으로 계산합니다. 이를 정렬해 상위(top) 또는 하위(top)를 뽑습니다. 가격 데이터는 `yfinance` 기반으로 최근 거래일까지 캐시/다운로드합니다.
- **왜 믿을 수 있나**: 유니버스는 해당 기준일의 실제 S&P500 편입 종목만 사용하며, 결측·극단 변동(`max_daily_change`) 종목은 제외합니다. lookback은 영업일 기준이라 공휴일·주말이 자동 보정됩니다.
- **백테스트와 관계**: 일일 이메일은 “지금 무엇을 사야 하나”에 집중하고, 백테스트는 “이 파라미터가 장기적으로 괜찮은가”를 검증합니다. 파라미터를 변경할 때만 백테스트를 다시 돌리면 됩니다.
고객이 매일 아침 “지금 보유해야 할 종목”과 “해당 전략의 백테스트 성과 요약”을 빠르게 확인할 수 있도록, **HTML 본문(요약 + 테이블)** + **CSV 첨부(실행용)** 형태로 보냅니다.

### 1) 포함 전략
- **모멘텀(Head)**: 직전 `lookback` 거래일 누적수익률 **상위** `top`개를 **동일가중** 매수.
- **리버설(Tail)**: 직전 `lookback` 거래일 누적수익률 **하위** `top`개를 **동일가중** 매수.
  - 운영 메일에서는 리버설을 **단기 10영업일/20영업일** 두 창으로 본다.
  - 즉, Tail은 `lookback ∈ {10, 20}` 각각에 대해 “가장 많이 떨어진 종목(top개)” 리스트를 만든다.

### 2) 이메일 제목(Subject) 규칙
- 예: `[Rank] Daily Holdings (2026-01-07 08:00 KST)`
- 예: `[Rank] Daily Holdings (2026-01-07 08:00 KST) | Head lk=60 top=50 | Tail lk=10/20 top=50`

### 3) 본문(HTML) 구성
1. **헤더(요약 3줄 이내)**
   - 기준 시각: `YYYY-MM-DD 08:00 KST` (데이터 기준일은 “가장 최근 거래일”로 별도 표기)
   - 유니버스: S&P 500 (해당 기준일의 실제 편입 종목)
   - 공통 규칙: 동일가중, 결측/극단변동(`max_daily_change`) 필터 적용 여부
2. **오늘 보유 종목(핵심)**
   - `Head (Momentum)` 테이블: `Rank | Ticker | LookbackPrice | CurrentPrice | LookbackReturn`
   - `Tail (Reversal, lk=10)` 테이블: 동일 컬럼
   - `Tail (Reversal, lk=20)` 테이블: 동일 컬럼
   - Name 소스가 없으므로 **Ticker만 표시**한다.
3. **(선택) 전일 대비 변경사항**
   - `Add / Remove`(편입/제외)와 `Weight change`(동일가중이면 보통 0) 요약
   - 전일 신호를 저장해두는 경우에만 표시
4. **백테스트 성과 요약(고정 섹션, 짧게)**
   - “최신 백테스트 산출물”에서 아래 지표를 1줄로 요약(전략별 1줄)
   - 예: `CAGR | ann_vol | max_drawdown | period_active_mean` (+ 필요시 Sharpe)
   - 백테스트는 매일 돌리지 않아도 되며, 고객이 원하는 시점에 재실행/갱신한다.

### 4) 첨부 파일(권장)
1. `signals_YYYYMMDD.csv` (주문/실행용)
   - 컬럼: `asof_date, data_date, strategy, mode, lookback, top, ticker, rank, lookback_price, current_price, lookback_return`
   - `strategy`: `rank_head` / `rank_tail`
   - `mode`: `head` / `tail`
2. (선택) `rank_summary.csv` 또는 `rank_summary.md`
   - 최신 백테스트 주요 지표 표(이미 생성돼 있으면 그대로 첨부)

### 5) 운영 파라미터(기본값 제안)
- `top`: 50 (이메일 보유종목 개수)
- `Head lookback` 후보: 60 / 120 / 250 (백테스트 결과로 1개 선택해 운영 신호에 고정)
- `Tail lookback` 후보: 10 / 20 / 40 (운영 신호는 필요 시 1~3개를 병렬로 제공)

### 6) GitHub Actions 운영(권장)
- **스케줄**: 매일 08:00 KST는 cron 기준 `0 23 * * *`(UTC, 전날 23:00)입니다.
- **산출물**: `.output/daily/YYYYMMDD/email.html`, `.output/daily/YYYYMMDD/signals.csv` (워크플로우 아티팩트로 업로드)
- **발송 방식(추천 순)**:
  1) **SMTP(프로바이더 무관)**: GitHub Secrets에 `SMTP_HOST/PORT/USER/PASS`, `MAIL_FROM`, `MAIL_TO`를 넣고 발송
     - 운영 관점에서 가장 단순하고, SES/사내메일/Google Workspace SMTP 등으로 대체 가능
  2) API 기반(예: SendGrid/Mailgun): 토큰 기반이지만 공급자 종속
- **추천 조합**: AWS SES(SMTP 자격증명) 또는 회사 SMTP (OAuth 없이 자동화가 쉬움)
