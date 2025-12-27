from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional

import yfinance as yf

from logger import logger


def normalize_ticker(ticker: str) -> str:
    """
    야후 파이낸스 호환 티커 형태로 정규화 (예: BRK.B -> BRK-B).
    """
    return ticker.strip().replace(".", "-").upper()


def _get_market_cap(ticker: str) -> Optional[float]:
    try:
        tk = yf.Ticker(ticker)
        fi = getattr(tk, "fast_info", None)
        if fi:
            cap = getattr(fi, "market_cap", None)
            if cap is None and isinstance(fi, dict):
                cap = fi.get("market_cap")
            if cap is not None:
                return cap

        info = tk.info
        return info.get("marketCap")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to fetch marketCap for {ticker}: {e}")
        return None


def fetch_market_caps(tickers: Iterable[str], max_workers: int = 8, retries: int = 1) -> Dict[str, float]:
    """
    복수 티커의 시가총액을 멀티스레드로 조회.

    :param tickers: 티커 목록
    :param max_workers: 동시 요청 스레드 수
    :param retries: 실패 시 재시도 횟수
    :return: {정규화된 티커: 시가총액} 딕셔너리
    """
    normed: List[str] = []
    seen = set()
    for t in tickers:
        nt = normalize_ticker(t)
        if nt not in seen:
            seen.add(nt)
            normed.append(nt)

    def task(t: str):
        cap = None
        for _ in range(retries + 1):
            cap = _get_market_cap(t)
            if cap is not None:
                break
        if cap is None:
            logger.debug(f"marketCap missing for {t}")
        return t, cap

    results: Dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(task, t): t for t in normed}
        for fut in as_completed(futures):
            t, cap = fut.result()
            if cap is not None:
                results[t] = cap
    logger.info(f"Fetched marketCap for {len(results)}/{len(normed)} tickers.")
    return results
