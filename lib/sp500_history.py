from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "res" / "sp500_ticker_start_end.csv"


def _resolve(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


@lru_cache(maxsize=None)
def _load_dataframe(resolved_csv: str) -> pd.DataFrame:
    df = pd.read_csv(resolved_csv, parse_dates=["start_date", "end_date"])
    df["ticker"] = df["ticker"].str.strip().str.upper()
    return df


def _get_df(csv_path: Path | str = DATA_PATH) -> pd.DataFrame:
    resolved = _resolve(csv_path)
    return _load_dataframe(str(resolved))


class SP500History:
    """
    Utility for reading S&P 500 membership intervals from res/sp500_ticker_start_end.csv.
    """

    def __init__(self, csv_path: Path | str = DATA_PATH):
        self.csv_path = csv_path

    @property
    def df(self) -> pd.DataFrame:
        return _get_df(self.csv_path)

    def constituents(self, ref_date: pd.Timestamp | str) -> List[str]:
        ref = pd.to_datetime(ref_date).normalize()
        active = self.df[
            (self.df["start_date"] <= ref)
            & (self.df["end_date"].isna() | (self.df["end_date"] >= ref))
        ]
        return sorted(active["ticker"].unique())

    def periods(self, tickers: Iterable[str]) -> pd.DataFrame:
        norm = {t.strip().upper() for t in tickers}
        return self.df[self.df["ticker"].isin(norm)].copy()


def get_sp500_constituents(ref_date: pd.Timestamp | str, csv_path: Path | str = DATA_PATH) -> List[str]:
    """
    Convenience wrapper to fetch constituents for a date without instantiating explicitly.
    """
    return SP500History(csv_path).constituents(ref_date)


def get_sp500_periods(tickers: Iterable[str], csv_path: Path | str = DATA_PATH) -> pd.DataFrame:
    """
    Convenience wrapper to fetch membership periods for specific tickers.
    """
    return SP500History(csv_path).periods(tickers)
