"""
자산배분 전략 백테스트 결과 분석

.output/asset_allocation/ 디렉토리의 백테스트 결과를 분석하여
HTML 리포트를 생성합니다.

분석 내용:
1. 전략별 성과 비교 (CAGR, Volatility, MDD, Sharpe)
2. 연도별 수익률 비교
3. 월별 수익률 히트맵
4. 상관관계 분석
5. 롤링 성과 분석

출력:
- .output/aa_analysis.html: 분석 리포트
- .output/aa_summary.csv: 요약 CSV (이미 run_aa_backtests.sh에서 생성)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze asset allocation backtest results.")
    p.add_argument(
        "--input-dir",
        default=".output/asset_allocation",
        help="Directory containing backtest results (default: .output/asset_allocation)",
    )
    p.add_argument(
        "--output",
        default=".output/aa_analysis.html",
        help="Output HTML report path (default: .output/aa_analysis.html)",
    )
    p.add_argument(
        "--summary-csv",
        default=None,
        help="Output summary CSV path (default: {input-dir}/summary.csv)",
    )
    return p.parse_args()


def max_drawdown(series: pd.Series) -> float:
    """Calculate maximum drawdown from a return series."""
    if series.empty:
        return float("nan")
    cum = (1 + series.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    return dd.min()


def load_returns(input_dir: Path) -> Dict[str, pd.Series]:
    """Load daily returns for each strategy."""
    returns = {}
    
    # Strategy files to look for
    strategies = ["BAA", "BDA", "LAA", "MDM", "final"]
    
    for name in strategies:
        csv_path = input_dir / f"{name}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if "return" in df.columns:
                    display_name = "Integration" if name == "final" else name
                    returns[display_name] = df["return"].fillna(0)
            except Exception as e:
                print(f"[warn] Failed to load {csv_path}: {e}")
    
    return returns


def compute_metrics(returns: pd.Series, name: str) -> dict:
    """Compute performance metrics from daily returns."""
    if returns.empty:
        return {
            "strategy": name,
            "start_date": None,
            "end_date": None,
            "days": 0,
            "years": 0,
            "cagr": float("nan"),
            "ann_vol": float("nan"),
            "max_drawdown": float("nan"),
            "sharpe_like": float("nan"),
            "total_return": float("nan"),
            "best_year": float("nan"),
            "worst_year": float("nan"),
        }
    
    days = len(returns)
    years = days / 252
    total_return = (1 + returns).prod() - 1
    cagr = (1 + total_return) ** (252 / days) - 1 if days > 0 else float("nan")
    ann_vol = returns.std() * sqrt(252)
    mdd = max_drawdown(returns)
    sharpe_like = cagr / ann_vol if ann_vol > 1e-9 else 0.0
    
    # Yearly returns
    yearly = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    best_year = yearly.max() if not yearly.empty else float("nan")
    worst_year = yearly.min() if not yearly.empty else float("nan")
    
    return {
        "strategy": name,
        "start_date": returns.index.min().date().isoformat(),
        "end_date": returns.index.max().date().isoformat(),
        "days": days,
        "years": round(years, 1),
        "cagr": cagr,
        "ann_vol": ann_vol,
        "max_drawdown": mdd,
        "sharpe_like": sharpe_like,
        "total_return": total_return,
        "best_year": best_year,
        "worst_year": worst_year,
    }


def build_yearly_returns_table(returns_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """Build yearly returns comparison table."""
    if not returns_dict:
        return pd.DataFrame()
    
    yearly_data = {}
    for name, rets in returns_dict.items():
        yearly = rets.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        yearly.index = yearly.index.year
        yearly_data[name] = yearly
    
    df = pd.DataFrame(yearly_data)
    df.index.name = "Year"
    return df


def build_monthly_returns_table(returns: pd.Series, name: str) -> pd.DataFrame:
    """Build monthly returns pivot table for heatmap."""
    if returns.empty:
        return pd.DataFrame()
    
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_df = monthly.to_frame("return")
    monthly_df["year"] = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month
    
    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return pivot


def build_correlation_matrix(returns_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """Build correlation matrix between strategies."""
    if not returns_dict:
        return pd.DataFrame()
    
    df = pd.DataFrame(returns_dict)
    return df.corr()


def build_rolling_sharpe(returns_dict: Dict[str, pd.Series], window: int = 252) -> pd.DataFrame:
    """Build rolling Sharpe ratio (annualized)."""
    if not returns_dict:
        return pd.DataFrame()
    
    rolling_sharpe = {}
    for name, rets in returns_dict.items():
        rolling_mean = rets.rolling(window).mean() * 252
        rolling_std = rets.rolling(window).std() * sqrt(252)
        rolling_sharpe[name] = rolling_mean / rolling_std.replace(0, np.nan)
    
    return pd.DataFrame(rolling_sharpe)


def df_to_html(df: pd.DataFrame, float_format: str = ".2%", classes: str = "data-table") -> str:
    """Convert DataFrame to HTML with formatting."""
    if df.empty:
        return "<p><i>No data</i></p>"
    
    styled = df.copy()
    for col in styled.select_dtypes(include=[np.floating]).columns:
        styled[col] = styled[col].apply(lambda x: f"{x:{float_format}}" if pd.notna(x) else "NA")
    
    return styled.to_html(classes=classes, border=0, na_rep="NA")


def fmt_pct(x: float, digits: int = 2) -> str:
    """Format as percentage."""
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}%}"


def fmt_float(x: float, digits: int = 2) -> str:
    """Format as float."""
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def build_summary_table_html(metrics_list: List[dict]) -> str:
    """Build summary metrics table HTML."""
    if not metrics_list:
        return "<p><i>No data</i></p>"
    
    rows = []
    for m in metrics_list:
        row = f"""
        <tr>
            <td><strong>{m.get('strategy', 'NA')}</strong></td>
            <td>{fmt_pct(m.get('cagr', float('nan')))}</td>
            <td>{fmt_pct(m.get('ann_vol', float('nan')))}</td>
            <td>{fmt_pct(m.get('max_drawdown', float('nan')))}</td>
            <td>{fmt_float(m.get('sharpe_like', float('nan')))}</td>
            <td>{fmt_pct(m.get('total_return', float('nan')))}</td>
            <td>{fmt_pct(m.get('best_year', float('nan')))}</td>
            <td>{fmt_pct(m.get('worst_year', float('nan')))}</td>
            <td>{m.get('years', 'NA')}</td>
        </tr>
        """
        rows.append(row)
    
    return f"""
    <table class="data-table">
        <thead>
            <tr>
                <th>Strategy</th>
                <th>CAGR</th>
                <th>Ann. Vol</th>
                <th>Max DD</th>
                <th>Sharpe</th>
                <th>Total Return</th>
                <th>Best Year</th>
                <th>Worst Year</th>
                <th>Years</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    """


def build_yearly_table_html(yearly_df: pd.DataFrame) -> str:
    """Build yearly returns comparison table HTML with color coding."""
    if yearly_df.empty:
        return "<p><i>No data</i></p>"
    
    def color_cell(val):
        if pd.isna(val):
            return "NA", ""
        if val > 0.15:
            return f"{val:.1%}", "background: #c6efce; color: #006100;"
        elif val > 0:
            return f"{val:.1%}", "background: #e2f0d9;"
        elif val > -0.10:
            return f"{val:.1%}", "background: #fce4d6;"
        else:
            return f"{val:.1%}", "background: #ffc7ce; color: #9c0006;"
    
    header = "<tr><th>Year</th>" + "".join(f"<th>{col}</th>" for col in yearly_df.columns) + "</tr>"
    
    rows = []
    for year, row in yearly_df.iterrows():
        cells = [f"<td><strong>{year}</strong></td>"]
        for val in row:
            text, style = color_cell(val)
            cells.append(f'<td style="{style}">{text}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    
    return f"""
    <table class="data-table">
        <thead>{header}</thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    """


def build_correlation_html(corr_df: pd.DataFrame) -> str:
    """Build correlation matrix HTML with color coding."""
    if corr_df.empty:
        return "<p><i>No data</i></p>"
    
    def color_cell(val, is_diagonal=False):
        if pd.isna(val):
            return "NA", ""
        if is_diagonal:
            return f"{val:.2f}", "background: #f5f5f5;"
        if val > 0.8:
            return f"{val:.2f}", "background: #c6efce;"
        elif val > 0.5:
            return f"{val:.2f}", "background: #e2f0d9;"
        elif val > 0.2:
            return f"{val:.2f}", "background: #fff2cc;"
        else:
            return f"{val:.2f}", "background: #fce4d6;"
    
    header = "<tr><th></th>" + "".join(f"<th>{col}</th>" for col in corr_df.columns) + "</tr>"
    
    rows = []
    for row_name, row in corr_df.iterrows():
        cells = [f"<td><strong>{row_name}</strong></td>"]
        for col_name, val in row.items():
            is_diagonal = row_name == col_name
            text, style = color_cell(val, is_diagonal)
            cells.append(f'<td style="{style}">{text}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    
    return f"""
    <table class="data-table" style="max-width: 600px;">
        <thead>{header}</thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    """


def build_monthly_heatmap_html(monthly_df: pd.DataFrame, strategy_name: str) -> str:
    """Build monthly returns heatmap HTML."""
    if monthly_df.empty:
        return f"<p><i>No monthly data for {strategy_name}</i></p>"
    
    def color_cell(val):
        if pd.isna(val):
            return "NA", ""
        if val > 0.05:
            return f"{val:.1%}", "background: #006100; color: white;"
        elif val > 0.02:
            return f"{val:.1%}", "background: #c6efce; color: #006100;"
        elif val > 0:
            return f"{val:.1%}", "background: #e2f0d9;"
        elif val > -0.02:
            return f"{val:.1%}", "background: #fce4d6;"
        elif val > -0.05:
            return f"{val:.1%}", "background: #ffc7ce; color: #9c0006;"
        else:
            return f"{val:.1%}", "background: #9c0006; color: white;"
    
    header = "<tr><th>Year</th>" + "".join(f"<th>{col}</th>" for col in monthly_df.columns) + "</tr>"
    
    rows = []
    for year, row in monthly_df.iterrows():
        cells = [f"<td><strong>{year}</strong></td>"]
        for val in row:
            text, style = color_cell(val)
            cells.append(f'<td style="{style}">{text}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    
    return f"""
    <h4>{strategy_name}</h4>
    <table class="data-table" style="font-size: 11px;">
        <thead>{header}</thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    """


STRATEGY_DESCRIPTIONS = {
    "BAA": "Bold Asset Allocation: 카나리아 자산(SPY,VWO,VEA,BND) 모멘텀으로 시장 판단, 공격/방어 자산 선택",
    "BDA": "Bond Dynamic Asset Allocation: 채권 ETF 중 6개월 수익률 상위 3개 선택",
    "LAA": "Lethargic Asset Allocation: IWD/GLD/IEF 고정 + SPY 200일선·실업률 조건부 QQQ/SHY",
    "MDM": "Modified Dual Momentum: SPY 12개월 수익률 > 0이면 공격(SPY/EFA), 아니면 방어(채권)",
    "Integration": "4가지 전략 동일가중(25%씩) 통합 포트폴리오",
}


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}\nRun scripts/run_aa_backtests.sh first.")
    
    print(f"Loading backtest results from {input_dir}...")
    returns_dict = load_returns(input_dir)
    
    if not returns_dict:
        raise SystemExit("No backtest results found. Run scripts/run_aa_backtests.sh first.")
    
    print(f"Found {len(returns_dict)} strategies: {list(returns_dict.keys())}")
    
    # Compute metrics
    metrics_list = [compute_metrics(rets, name) for name, rets in returns_dict.items()]
    
    # Build analysis components
    yearly_df = build_yearly_returns_table(returns_dict)
    corr_df = build_correlation_matrix(returns_dict)
    
    # Build monthly heatmaps for each strategy
    monthly_heatmaps = []
    for name, rets in returns_dict.items():
        monthly_df = build_monthly_returns_table(rets, name)
        if not monthly_df.empty:
            monthly_heatmaps.append(build_monthly_heatmap_html(monthly_df, name))
    
    # Strategy descriptions
    desc_html = "<ul>"
    for name in returns_dict.keys():
        desc = STRATEGY_DESCRIPTIONS.get(name, "")
        desc_html += f"<li><strong>{name}</strong>: {desc}</li>"
    desc_html += "</ul>"
    
    # Build HTML report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Backtest info
    if metrics_list:
        sample = metrics_list[0]
        backtest_info = f"""
        <ul>
            <li><strong>백테스트 기간:</strong> {sample.get('start_date', 'NA')} ~ {sample.get('end_date', 'NA')}</li>
            <li><strong>분석 전략 수:</strong> {len(metrics_list)}개 (BAA, BDA, LAA, MDM, Integration)</li>
            <li><strong>리포트 생성 시각:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
        </ul>
        """
    else:
        backtest_info = "<p>No metrics available</p>"
    
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Asset Allocation Backtest Analysis</title>
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
        h4 {{ margin-top: 15px; color: #666; font-size: 14px; }}
        .info-box {{ background: #e8f4fd; padding: 15px; margin: 15px 0; border-radius: 5px; border: 1px solid #b8d4e8; }}
        .info-box ul {{ margin: 0; padding-left: 20px; }}
        .info-box li {{ margin: 5px 0; }}
        .guide {{ background: #f8f8f8; padding: 10px 15px; margin: 10px 0; border-left: 3px solid #666; }}
        .guide p {{ margin: 5px 0; }}
        .data-table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 13px;
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
        .highlight-box {{
            background: #f0fff4;
            border: 1px solid #9ae6b4;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }}
        .highlight-box h3 {{
            color: #276749;
            margin-top: 0;
        }}
        .strategy-desc {{
            background: #fffaf0;
            border: 1px solid #fbd38d;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }}
        .strategy-desc ul {{ margin: 0; padding-left: 20px; }}
        .strategy-desc li {{ margin: 8px 0; }}
    </style>
</head>
<body>
    <h1>Asset Allocation Backtest Analysis</h1>
    
    <h2>백테스트 정보</h2>
    <div class="info-box">{backtest_info}</div>
    
    <h2>전략 설명</h2>
    <div class="strategy-desc">{desc_html}</div>
    
    <h2>1. 전략별 성과 비교</h2>
    <div class="guide">
        <p><strong>CAGR:</strong> 연평균 복리 수익률. 높을수록 좋음.</p>
        <p><strong>Ann. Vol:</strong> 연환산 변동성. 낮을수록 안정적.</p>
        <p><strong>Max DD:</strong> 최대 낙폭. 절대값이 작을수록 리스크가 낮음.</p>
        <p><strong>Sharpe:</strong> 위험조정 수익률(CAGR/Vol). 높을수록 효율적.</p>
    </div>
    {build_summary_table_html(metrics_list)}
    
    <h2>2. 연도별 수익률 비교</h2>
    <div class="guide">
        <p>각 전략의 연도별 수익률을 비교합니다. 녹색은 양수, 빨간색은 음수 수익률입니다.</p>
    </div>
    {build_yearly_table_html(yearly_df)}
    
    <h2>3. 전략 간 상관관계</h2>
    <div class="guide">
        <p>전략 간 일별 수익률 상관계수입니다. 낮은 상관관계(0.5 미만)의 전략을 조합하면 분산 효과가 있습니다.</p>
    </div>
    {build_correlation_html(corr_df)}
    
    <h2>4. 월별 수익률 히트맵</h2>
    <div class="guide">
        <p>각 전략의 월별 수익률 분포입니다. 진한 녹색은 +5% 이상, 진한 빨간색은 -5% 이하입니다.</p>
    </div>
    {"".join(monthly_heatmaps)}
    
    <h2>5. 투자 시사점</h2>
    <div class="highlight-box">
        <h3>분석 요약</h3>
        <ul>
            <li><strong>Integration(통합):</strong> 4개 전략을 동일가중하여 개별 전략 대비 변동성이 낮고 안정적인 성과를 기대할 수 있습니다.</li>
            <li><strong>분산 효과:</strong> 상관관계가 낮은 전략들의 조합으로 리스크 대비 수익률이 개선됩니다.</li>
            <li><strong>시장 적응:</strong> BAA/MDM은 시장 상황에 따라 공격/방어 전환, LAA는 대체로 안정적, BDA는 채권 중심입니다.</li>
        </ul>
    </div>
    
    <hr/>
    <p style="font-size: 12px; color: #888;">
        Generated by scripts/analyze_aa_summary.py at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>
</body>
</html>
"""
    
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Wrote analysis report to {output_path}")
    
    # Also write/update summary CSV
    summary_csv_path = Path(args.summary_csv) if args.summary_csv else input_dir / "summary.csv"
    summary_df = pd.DataFrame(metrics_list)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Wrote summary CSV to {summary_csv_path}")


if __name__ == "__main__":
    main()
