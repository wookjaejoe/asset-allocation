"""
Analyze rank backtest results (lookback/top trends, top-size sensitivity, stability).

Reads .output/rank_summary.csv produced by aggregate_rank_results.py and writes an
HTML report with grouped metrics to help pick robust parameters.

If aggregate_rank_results.py has not been run yet, this script will invoke it
automatically unless --skip-aggregate is specified.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd




DEFAULT_METRICS: Sequence[str] = (
    "cagr",
    "ann_vol",
    "max_drawdown",
    "period_active_mean",
    "period_active_spy_mean",
)

INTERPRETATION_GUIDE = [
    "목표: 룩백 길이, 종목 수(top), 리밸런스 주기에 따른 성과/리스크 민감도를 보고 일관적이고 재현 가능한 파라미터를 고릅니다.",
    "우선순위: <code>cagr</code>·<code>period_active_mean</code>(수익), <code>ann_vol</code>·<code>max_drawdown</code>(리스크), <code>sharpe_like</code>(위험조정), <code>period_active_mean</code>의 <code>std</code>(안정성) 순으로 밸런스를 확인하세요.",
    "리밸런스 고려: 모든 표는 rebalance_months를 평균한 값이지만 마지막 섹션에서 주기별 변동성(민감도)을 다시 점검합니다.",
    "투자 적용 팁: 수익/리스크가 균형 잡힌 룩백·top 조합을 기본값으로 삼고, 주기 민감도가 낮은(안정적인) 조합을 우선 사용합니다. 집중도가 높은 전략(top10)이 유리해도 변동성이 크게 늘면 분산형(top50)과 혼합해 운용할 수 있습니다.",
]

METRIC_EXPLANATIONS = [
    "<code>cagr</code>: 연평균 성장률. 높을수록 장기 복리 성과가 좋음. 다만 변동성·최대낙폭과 함께 봐서 과도한 리스크를 걸러냅니다.",
    "<code>ann_vol</code>: 연환산 변동성. 낮을수록 수익 경로가 매끄럽고 포지션 사이징에 유리합니다.",
    "<code>max_drawdown</code>: 백테스트 기간 중 최대 낙폭. 단기/심리적 리스크 허용도와 맞는지 확인합니다.",
    "<code>period_active_mean</code>: 벤치마크 대비 구간별 초과수익 평균. 양수·높을수록 알파가 꾸준히 발생한 것.",
    "<code>period_active_spy_mean</code>: S&amp;P500 대비 초과수익 평균(데이터에 있을 때). 시장 방향성에 대한 상대적 우위 판단에 사용합니다.",
    "<code>sharpe_like</code>: <code>cagr/ann_vol</code>로 계산한 간이 위험조정 성과. 동일 mode/top 내에서 높은 값이 효율적입니다.",
    "<code>mean</code>·<code>std</code>(stability): <code>period_active_mean</code>의 평균·표준편차. 평균↑·표준편차↓일수록 리밸런스 주기 변화에도 알파가 안정적입니다.",
    "<code>*_SMALL-LARGE</code> deltas: 작은 top(집중) 대비 큰 top(분산)의 성과/리스크 차이. 수익 이득이 리스크 증가를 정당화하는지 확인합니다.",
]

SECTION_GUIDES = {
    "lookback_trend": [
        "의미: 동일 rebalance_months 평균을 통해 룩백·top 조합별 전반적 성향을 봅니다.",
        "해석 팁: <code>cagr</code>·<code>period_active_mean</code>은 높고 <code>ann_vol</code>·<code>max_drawdown</code>은 낮은 지점을 찾고, <code>sharpe_like</code>가 함께 높으면 효율적입니다.",
        "투자 활용: 단기/중기/장기 룩백 중 어떤 구간이 시장에 더 적합했는지 파악해 기본 파라미터 후보로 삼습니다.",
    ],
    "lookback_agg": [
        "의미: top 크기를 평균해 룩백 길이 자체의 구조적 우위를 확인합니다.",
        "해석 팁: 특정 룩백이 꾸준히 우수하면 top 크기를 조절하면서도 해당 룩백을 유지하는 전략이 유리할 수 있습니다.",
        "투자 활용: 리밸런스나 top 변경 시에도 유지할 기본 룩백 길이를 선택할 때 참고합니다.",
    ],
    "top_deltas": [
        "의미: 집중(top small) vs 분산(top large) 전략의 성과/리스크 차이를 수치로 비교합니다.",
        "해석 팁: <code>cagr_</code>와 <code>active_</code>가 양수이면 집중의 성과가 우수. 동시에 <code>vol_</code>·<code>mdd_</code>가 얼마나 늘었는지 확인해 위험 대비 매력도를 판단합니다.",
        "투자 활용: 집중도가 높을수록 포지션 사이징이나 헷지 비중을 보수적으로 조정하고, 분산형은 변동성 관리가 수월합니다.",
    ],
    "sharpe_rank": [
        "의미: 위험조정 효율 순위를 mode/top별로 제시합니다.",
        "해석 팁: <code>sharpe_like</code> 상위 룩백은 같은 top 내에서 더 적은 변동성으로 성과를 냈음을 뜻합니다. <code>cagr</code>가 낮은데 순위가 높다면 방어적 성향.",
        "투자 활용: 기본 전략 선택 시 최우선 후보군이며, 리스크 한도를 줄여야 할 때 대체 옵션으로 활용합니다.",
    ],
    "stability": [
        "의미: 리밸런스 주기 변화에 대한 알파 안정성을 mean/std로 측정합니다.",
        "해석 팁: mean이 높고 std가 작으면 주기 선택에 덜 민감하므로 실거래에서 유지하기 쉬운 조합입니다.",
        "투자 활용: 운영/거래 비용 제약으로 리밸런스 빈도를 바꿔야 할 때 안정성이 높은 룩백·top을 우선 고려합니다.",
    ],
    "rebalance_split": [
        "의미: 1/3/6개월 등 리밸런스 주기별 성과 차이를 직접 비교합니다.",
        "해석 팁: 특정 주기에서만 성과가 나는 경우 오버핏 위험이 있으니 다른 주기에서도 과도하게 나빠지지 않는지 확인합니다.",
        "투자 활용: 거래비용, 세금, 운용 규모에 맞춰 적정 리밸런스 주기를 결정하고, 민감도가 낮은 전략을 기본으로 삼습니다.",
    ],
}


def section_with_guide(title: str, guide_key: str, body: str) -> str:
    guide_lines = SECTION_GUIDES.get(guide_key, [])
    guide_html = "".join(f"<p>{line}</p>" for line in guide_lines)
    return f"<h2>{title}</h2>\n<div class='guide'>{guide_html}</div>\n{body}"


def format_df(df: pd.DataFrame, float_cols: Iterable[str], digits: int = 3) -> pd.DataFrame:
    """Round selected float columns for readable Markdown output."""
    df = df.copy()
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float).round(digits)
    return df


def ensure_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in summary CSV: {', '.join(missing)}")
    return df


def add_sharpe_like(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Avoid division by zero; set sharpe_like to 0 when ann_vol is 0 or very small
    df["sharpe_like"] = df.apply(
        lambda row: row["cagr"] / row["ann_vol"] if row["ann_vol"] > 1e-9 else 0.0,
        axis=1,
    )
    return df


def filter_modes(df: pd.DataFrame, modes: Sequence[str] | None) -> pd.DataFrame:
    if not modes:
        return df
    missing = [m for m in modes if m not in set(df["mode"])]
    if missing:
        print(f"[warn] requested modes not in data: {missing}")
    return df[df["mode"].isin(modes)]


def build_tables(df: pd.DataFrame, compare_tops: Sequence[int]) -> List[str]:
    sections: List[str] = []
    # Filter DEFAULT_METRICS to only include columns present in the data
    available_metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    df = ensure_cols(df, ["mode", "lookback", "top", "rebalance_months"])
    df = add_sharpe_like(df)
    metrics = available_metrics + ["sharpe_like"]

    # 1) Lookback trend per mode/top (mean across rebalance months)
    grouped = (
        df.groupby(["mode", "lookback", "top"])[metrics]
        .mean()
        .reset_index()
        .sort_values(["mode", "lookback", "top"])
    )
    sections.append(
        section_with_guide(
            "Lookback trend (mean across rebalance months)",
            "lookback_trend",
            df_to_html(format_df(grouped, metrics)),
        )
    )

    # 2) Lookback trend aggregated across top sizes (mean across top & rebalance)
    agg_lb = (
        df.groupby(["mode", "lookback"])[metrics]
        .mean()
        .reset_index()
        .sort_values(["mode", "lookback"])
    )
    sections.append(
        section_with_guide(
            "Lookback trend aggregated across tops",
            "lookback_agg",
            df_to_html(format_df(agg_lb, metrics)),
        )
    )

    # 3) Top-size sensitivity: deltas between two tops for each mode/lookback
    compare_tops = list(compare_tops)
    if len(compare_tops) != 2:
        raise SystemExit("--compare-tops expects exactly two integers (e.g., 10 50)")
    small, large = compare_tops
    available_tops = set(df["top"].unique())
    if small not in available_tops or large not in available_tops:
        print(f"[warn] compare-tops ({small}, {large}) not fully in data tops: {sorted(available_tops)}")
    deltas = []
    base = (
        df.groupby(["mode", "lookback", "top"])[metrics]
        .mean()
        .reset_index()
    )
    for mode in base["mode"].unique():
        for lb in sorted(base[base["mode"] == mode]["lookback"].unique()):
            sub = base[(base["mode"] == mode) & (base["lookback"] == lb)]
            if {small, large}.issubset(set(sub["top"])):
                m_small = sub[sub["top"] == small].iloc[0]
                m_large = sub[sub["top"] == large].iloc[0]
                deltas.append(
                    {
                        "mode": mode,
                        "lookback": lb,
                        f"cagr_{small}-{large}": m_small["cagr"] - m_large["cagr"],
                        f"active_{small}-{large}": m_small["period_active_mean"] - m_large["period_active_mean"],
                        f"vol_{small}-{large}": m_small["ann_vol"] - m_large["ann_vol"],
                        f"mdd_{small}-{large}": m_small["max_drawdown"] - m_large["max_drawdown"],
                        f"sharpe_like_{small}-{large}": m_small["sharpe_like"] - m_large["sharpe_like"],
                    }
                )
    if deltas:
        delta_df = pd.DataFrame(deltas).sort_values(["mode", "lookback"])
        sections.append(
            section_with_guide(
                f"Top-size deltas (top{small} minus top{large}, mean across rebalance months)",
                "top_deltas",
                df_to_html(format_df(delta_df, delta_df.columns.difference(["mode", "lookback"]))),
            )
        )
    else:
        sections.append(
            section_with_guide(
                "Top-size deltas",
                "top_deltas",
                f"No pairs with both top{small} and top{large} found.",
            )
        )

    # 4) Sharpe-like ranking per mode/top (mean across rebalance months)
    sharpe_rank = (
        base.sort_values(["mode", "top", "sharpe_like"], ascending=[True, True, False])
        .groupby(["mode", "top"])
    )
    sharpe_sections: List[str] = []
    sharpe_cols = ["lookback", "sharpe_like", "cagr", "ann_vol", "max_drawdown"]
    if "period_active_mean" in base.columns:
        sharpe_cols.append("period_active_mean")
    sharpe_float_cols = [c for c in sharpe_cols if c not in ("lookback",)]
    for (mode, top), sub in sharpe_rank:
        sharpe_sections.append(
            f"<h3>{mode} top{top}</h3>\n"
            + df_to_html(
                format_df(
                    sub[[c for c in sharpe_cols if c in sub.columns]],
                    sharpe_float_cols,
                )
            )
        )
    sections.append(
        section_with_guide(
            "Sharpe-like ranking per mode/top",
            "sharpe_rank",
            "".join(sharpe_sections),
        )
    )

    # 5) Stability across rebalance months: mean/std of period_active_mean
    stability_metric = "period_active_mean" if "period_active_mean" in df.columns else "cagr"
    robust = (
        df.groupby(["mode", "lookback", "top"])[stability_metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["mode", "lookback", "top"])
    )
    # Fill NaN std (when only 1 data point) with 0
    robust["std"] = robust["std"].fillna(0.0)
    sections.append(
        section_with_guide(
            f"{stability_metric} mean/std by mode/lookback/top",
            "stability",
            df_to_html(format_df(robust, ["mean", "std"])),
        )
    )

    # 6) Rebalance-month split (optional detail)
    per_rbm = (
        df.groupby(["mode", "lookback", "top", "rebalance_months"])[metrics]
        .mean()
        .reset_index()
        .sort_values(["mode", "lookback", "top", "rebalance_months"])
    )
    sections.append(
        section_with_guide(
            "Metrics per rebalance_months",
            "rebalance_split",
            df_to_html(format_df(per_rbm, metrics)),
        )
    )

    return sections


def df_to_html(df: pd.DataFrame) -> str:
    """Return HTML table."""
    return df.to_html(index=False, classes="data-table", border=0)


def build_backtest_info(df: pd.DataFrame) -> str:
    """Generate HTML block summarizing backtest settings from data."""
    info_items = []
    
    # Backtest period
    if "daily_start" in df.columns and "daily_end" in df.columns:
        starts = pd.to_datetime(df["daily_start"].dropna())
        ends = pd.to_datetime(df["daily_end"].dropna())
        if not starts.empty and not ends.empty:
            period_start = starts.min().strftime("%Y-%m-%d")
            period_end = ends.max().strftime("%Y-%m-%d")
            info_items.append(f"<li><strong>백테스트 기간:</strong> {period_start} ~ {period_end}</li>")
    
    # Data counts
    if "days" in df.columns:
        min_days = int(df["days"].min())
        max_days = int(df["days"].max())
        info_items.append(f"<li><strong>거래일 수:</strong> {min_days} ~ {max_days}일</li>")
    
    # Modes tested
    if "mode" in df.columns:
        modes = sorted(df["mode"].unique())
        info_items.append(f"<li><strong>모드:</strong> {', '.join(modes)}</li>")
    
    # Lookback values
    if "lookback" in df.columns:
        lookbacks = sorted(df["lookback"].unique())
        info_items.append(f"<li><strong>룩백 기간:</strong> {', '.join(map(str, lookbacks))}일</li>")
    
    # Top sizes
    if "top" in df.columns:
        tops = sorted(df["top"].unique())
        info_items.append(f"<li><strong>종목 수 (top):</strong> {', '.join(map(str, tops))}</li>")
    
    # Rebalance months
    if "rebalance_months" in df.columns:
        rbms = sorted(df["rebalance_months"].unique())
        info_items.append(f"<li><strong>리밸런스 주기:</strong> {', '.join(map(str, rbms))}개월</li>")
    
    # Total configurations
    info_items.append(f"<li><strong>총 설정 조합 수:</strong> {len(df)}개</li>")
    
    # Report generation time
    info_items.append(f"<li><strong>리포트 생성 시각:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>")
    
    return f"<ul>{''.join(info_items)}</ul>"


def run_aggregate_if_needed(summary_path: Path, output_root: str, force: bool = False) -> bool:
    """Run aggregate_rank_results.py if summary CSV doesn't exist or force is True.
    
    Returns True if aggregate was run, False otherwise.
    """
    if not force and summary_path.exists():
        return False
    
    script_dir = Path(__file__).parent
    aggregate_script = script_dir / "aggregate_rank_results.py"
    
    if not aggregate_script.exists():
        print(f"[warn] aggregate_rank_results.py not found at {aggregate_script}")
        return False
    
    print(f"Running aggregate_rank_results.py to generate {summary_path}...")
    try:
        result = subprocess.run(
            [sys.executable, str(aggregate_script), "--output-root", output_root, "--summary-csv", str(summary_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[error] aggregate_rank_results.py failed:\n{result.stderr}")
            return False
        print(result.stdout)
        return True
    except Exception as e:
        print(f"[error] Failed to run aggregate_rank_results.py: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Systematic trend analysis for rank backtests (lookback/top sensitivity).")
    parser.add_argument(
        "--summary",
        default=".output/rank_summary.csv",
        help="Path to rank_summary.csv generated by aggregate_rank_results.py",
    )
    parser.add_argument(
        "--output",
        default=".output/rank_analysis.html",
        help="Where to write the HTML report.",
    )
    parser.add_argument(
        "--output-root",
        default=".output",
        help="Root folder containing rank_* subfolders (passed to aggregate_rank_results.py).",
    )
    parser.add_argument(
        "--mode",
        action="append",
        dest="modes",
        choices=["head", "tail", "mix"],
        help="Filter to specific modes. Repeatable. Default: all modes present.",
    )
    parser.add_argument(
        "--compare-tops",
        nargs=2,
        type=int,
        default=[10, 50],
        metavar=("SMALL", "LARGE"),
        help="Two top sizes to compare for deltas (default: 10 50).",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip running aggregate_rank_results.py even if summary CSV is missing.",
    )
    parser.add_argument(
        "--force-aggregate",
        action="store_true",
        help="Force re-run aggregate_rank_results.py even if summary CSV exists.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    
    # Run aggregate if needed (unless --skip-aggregate)
    if not args.skip_aggregate:
        run_aggregate_if_needed(summary_path, args.output_root, force=args.force_aggregate)
    
    if not summary_path.exists():
        raise SystemExit(f"Summary CSV not found: {summary_path}\nRun aggregate_rank_results.py first or remove --skip-aggregate.")

    df = pd.read_csv(summary_path)
    df_full = df.copy()  # Keep full data for backtest info
    df = filter_modes(df, args.modes)
    if df.empty:
        raise SystemExit("Filtered data is empty; check --mode or summary file.")

    sections = build_tables(df, args.compare_tops)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert guide texts to HTML
    interpretation_html = "".join(f"<p>{line}</p>" for line in INTERPRETATION_GUIDE)
    metric_html = "".join(f"<p>{line}</p>" for line in METRIC_EXPLANATIONS)
    
    # Build backtest info from full data (before mode filtering)
    backtest_info_html = build_backtest_info(df_full)
    
    content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rank Backtest Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
        h2 {{ margin-top: 40px; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
        h3 {{ margin-top: 20px; color: #555; }}
        .guide {{ background: #f8f8f8; padding: 10px 15px; margin: 10px 0; border-left: 3px solid #666; }}
        .guide p {{ margin: 5px 0; }}
        .info-box {{ background: #e8f4fd; padding: 15px; margin: 15px 0; border-radius: 5px; border: 1px solid #b8d4e8; }}
        .info-box ul {{ margin: 0; padding-left: 20px; }}
        .info-box li {{ margin: 5px 0; }}
        .data-table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 14px;
        }}
        .data-table th, .data-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: right;
        }}
        .data-table th {{
            background: #f5f5f5;
            font-weight: 600;
        }}
        .data-table tr:nth-child(even) {{ background: #fafafa; }}
        .data-table tr:hover {{ background: #f0f0f0; }}
    </style>
</head>
<body>
    <h1>Rank Backtest Analysis</h1>
    
    <h2>백테스트 설정 요약</h2>
    <div class="info-box">{backtest_info_html}</div>
    
    <h2>How to read this report</h2>
    <div class="guide">{interpretation_html}</div>
    
    <h2>Metric meaning &amp; interpretation tips</h2>
    <div class="guide">{metric_html}</div>
    
    {"".join(sections)}
</body>
</html>
"""
    output_path.write_text(content, encoding="utf-8")
    print(f"Wrote analysis to {output_path}")


if __name__ == "__main__":
    main()
