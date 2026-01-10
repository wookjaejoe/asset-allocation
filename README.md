

## Output
- output/.csv: 전략별/일자별 포트폴리오 구성 및 수익률
- output/.html: 전략별 백테스트 결과
- output/final.csv: 최종 전략의 일자별 포트폴리오 구성 및 수익률
- output/final.html: 최종 전략 백테스트 결과
- .output/daily/YYYYMMDD/: (Rank) 데일리 이메일 산출물 (`scripts/daily_rank_report.py`)
- .output/daily_asset_allocation/YYYYMMDD/: (Asset Allocation) 데일리 이메일 산출물 (`scripts/daily_asset_allocation_report.py`)


## Build
```
pyinstaller main.spec
```

## Trouble shooting

quantstats:0.0.64 `UnsupportedFunctionCall: numpy operations are not valid with resample. Use .resample(...).sum() instead` 오류 발생한다. 아래와 같이 수정:

https://github.com/bbalouki/quantstats/blob/bbs-dev/quantstats/_plotting/core.py
Line 292
```
if resample:
    returns = returns.resample(resample).last() if compound else returns.resample(resample).sum()
    if isinstance(benchmark, _pd.Series):
        benchmark = benchmark.resample(resample).last() if compound else benchmark.resample(resample).sum()
```

---

- `dist/main/_internal/input` -> `dist/main/input` 으로 이동
- `dist/main/_internal/quantstats/report.html` -> `reports.html` 으로 rename 
