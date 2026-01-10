from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

from logger import logger


@dataclass
class CacheMeta:
    created_at: datetime
    start: Optional[str]
    end: Optional[str]
    universe_size: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "created_at": self.created_at.isoformat(),
                "start": self.start,
                "end": self.end,
                "universe_size": self.universe_size,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def from_file(path: Path) -> Optional["CacheMeta"]:
        try:
            data = json.loads(path.read_text())
            return CacheMeta(
                created_at=datetime.fromisoformat(data["created_at"]),
                start=data.get("start"),
                end=data.get("end"),
                universe_size=int(data.get("universe_size", 0)),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load cache meta {path}: {e}")
            return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _cache_paths(base: Path) -> tuple[Path, Path]:
    return base / "prices.csv", base / "prices.meta.json"


def _download(tickers: list[str], start: str, end: Optional[str]) -> pd.DataFrame:
    logger.info(f"Downloading prices for {len(tickers)} tickers from {start} to {end or 'today'}")
    data = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, progress=True)
    closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    closes = closes.dropna(how="all").dropna(axis=1, how="all").sort_index()
    logger.info(f"Price matrix shape: {closes.shape}")
    return closes


def load_prices_with_cache(
    tickers: Iterable[str],
    start: str,
    end: Optional[str],
    cache_dir: Path = Path(".cache"),
    full_refresh_age: timedelta = timedelta(days=1),
    partial_refresh_age: timedelta = timedelta(hours=1),
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = _cache_paths(cache_dir)

    universe = sorted(set(tickers))
    now = _now()

    meta = CacheMeta.from_file(meta_path) if meta_path.exists() else None
    prices: Optional[pd.DataFrame] = None

    if csv_path.exists():
        try:
            prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            prices = prices.sort_index()
            logger.info(f"Loaded cached prices: {prices.shape}, created at {meta.created_at if meta else 'unknown'}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to read cache CSV {csv_path}: {e}")

    needs_full = True
    needs_partial = False
    if prices is not None and meta is not None:
        age = now - meta.created_at
        needs_full = age >= full_refresh_age
        needs_partial = (age >= partial_refresh_age) and not needs_full

    # If cache is missing any requested tickers, force a full refresh with the requested universe.
    if prices is not None:
        missing_cols = [t for t in universe if t not in prices.columns]
        if missing_cols:
            logger.info(f"Cache missing {len(missing_cols)} requested tickers; forcing full refresh")
            needs_full = True
            needs_partial = False

    # Full refresh
    if needs_full:
        logger.info("Cache older than full refresh threshold; downloading full dataset")
        try:
            fresh = _download(universe, start, end)
            if not fresh.empty:
                fresh.to_csv(csv_path)
                meta = CacheMeta(created_at=now, start=start, end=end, universe_size=len(universe))
                meta_path.write_text(meta.to_json())
                prices = fresh
                logger.info(f"Replaced cache with fresh download {prices.shape}")
                needs_partial = False
        except Exception as e:  # noqa: BLE001
            logger.error(f"Full refresh failed, keeping existing cache: {e}")

    # Partial refresh (append recent data)
    if prices is not None and needs_partial:
        try:
            last_date = prices.index.max()
            partial_start = (last_date + pd.Timedelta(days=1)).date().isoformat()
            logger.info(f"Partial update from {partial_start} to {end or 'today'}")
            recent = _download(universe, partial_start, end)
            if not recent.empty:
                merged = pd.concat([prices, recent])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                merged.to_csv(csv_path)
                meta = CacheMeta(created_at=now, start=start, end=end, universe_size=len(universe))
                meta_path.write_text(meta.to_json())
                prices = merged
                logger.info(f"Cache updated with recent data, new shape {prices.shape}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Partial refresh failed, keeping existing cache: {e}")

    if prices is None:
        # no usable cache, attempt one-time download
        prices = _download(universe, start, end)
        if prices.empty:
            raise RuntimeError("Price data is empty after download attempt.")
        prices.to_csv(csv_path)
        meta = CacheMeta(created_at=now, start=start, end=end, universe_size=len(universe))
        meta_path.write_text(meta.to_json())

    # Ensure returned frame contains at least the requested tickers (others may exist from prior cache runs).
    keep = [t for t in universe if t in prices.columns]
    return prices[keep].copy()
