"""
data_utils.py
================
历史数据加载与重采样工具。

数据来源:
  gold2012_2024.csv  —— 1 分钟 OHLC, 列: name,data(YYYYMMDD),time(HHMMSS),open,high,low,close,vol
  时间被视为 UTC (经纪商常用 GMT/GMT+2/3, 仅影响交易时段过滤; 可在 config 调整 tz_shift)。
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


# 历史数据默认路径 (与脚本同级或 XAUUSD 目录)
DEFAULT_M1_CSV = r"E:\fun\XAUUSD\gold2012_2024.csv"


def load_m1(csv_path: str = DEFAULT_M1_CSV, tz_shift_hours: int = 0) -> pd.DataFrame:
    """
    加载 1 分钟 OHLC 数据, 返回带 DatetimeIndex 的 DataFrame[open,high,low,close,volume]。

    tz_shift_hours: 把原始时间平移到 UTC 的小时数(用于交易时段过滤)。
                    若原始数据已是 UTC 则填 0。
    """
    df = pd.read_csv(
        csv_path,
        dtype={"data": str, "time": str},
    )
    # 兼容列名
    cols = {c.lower(): c for c in df.columns}
    df = df.rename(columns={cols.get("open", "open"): "open",
                            cols.get("high", "high"): "high",
                            cols.get("low", "low"): "low",
                            cols.get("close", "close"): "close"})

    dt = pd.to_datetime(df["data"] + df["time"].str.zfill(6), format="%Y%m%d%H%M%S")
    if tz_shift_hours:
        dt = dt + pd.Timedelta(hours=tz_shift_hours)
    dt = pd.DatetimeIndex(dt.to_numpy())  # 转 numpy 避免索引对齐问题(pandas 会按索引对齐导致 NaN)

    # 注意: 必须用 .to_numpy() 取值, 否则 Series 自带的 RangeIndex 会与 dt 对齐而产生 NaN
    out = pd.DataFrame({
        "open": df["open"].astype(float).to_numpy(),
        "high": df["high"].astype(float).to_numpy(),
        "low": df["low"].astype(float).to_numpy(),
        "close": df["close"].astype(float).to_numpy(),
        "volume": (df["vol"].astype(float).to_numpy() if "vol" in df.columns
                   else np.zeros(len(df))),
    }, index=dt)
    out.index.name = "time"
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def resample(df_m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    """把 1 分钟数据重采样为更大周期。rule 例: '5min', '15min', '1h'。"""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df_m1.resample(rule, label="right", closed="right").agg(agg).dropna(subset=["open"])
    return out


def timeframe_to_rule(tf: str) -> str:
    return {"M1": "1min", "M5": "5min", "M15": "15min"}[tf]


def infer_tf_minutes(index: pd.DatetimeIndex) -> int:
    """从 DatetimeIndex 推断工作周期(分钟) = 最小正的相邻时间差。"""
    diffs = np.diff(index.values).astype("timedelta64[s]").astype(np.int64)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1
    return max(1, int(np.min(diffs)) // 60)


def mtf_ema_aligned(df_working: pd.DataFrame, base_min: int,
                    tf_min: int, period: int) -> np.ndarray:
    """
    计算"真正的多周期均线"并因果对齐到工作周期的每根K线。
    在 tf_min 周期(右沿标注的已收盘K线)上算 EMA(period), 再用 merge_asof(backward)
    投影到工作周期 index: 每根工作K线收盘时, 取"已收盘的最近一根 tf_min K线"的均线值。
    -> 不使用未来数据 (causal)。

    若 tf_min < base_min (比工作周期还细), 无法由工作数据合成, 返回全 NaN。
    """
    n = len(df_working)
    if tf_min < base_min:
        return np.full(n, np.nan)
    from smc_core import ema  # 局部导入避免循环
    if tf_min == base_min:
        e = np.array(ema(df_working["close"].to_numpy(dtype=float), period), dtype=float)
        e[:period] = np.nan                            # 预热期不可用
        return e

    r = resample(df_working, f"{tf_min}min")          # 右沿标注, 收盘对齐
    if len(r) == 0:
        return np.full(n, np.nan)
    e = np.array(ema(r["close"].to_numpy(dtype=float), period), dtype=float)
    e[:period] = np.nan                                # 高周期均线预热期置 NaN
    right = pd.DataFrame({"t": r.index.values, "v": e}).sort_values("t")
    left = pd.DataFrame({"t": df_working.index.values}).sort_values("t")
    merged = pd.merge_asof(left, right, on="t", direction="backward")
    return merged["v"].to_numpy()


def slice_dates(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


if __name__ == "__main__":
    # 快速自检
    d = load_m1()
    print("M1 范围:", d.index.min(), "->", d.index.max(), "共", len(d), "根")
    for tf in ("M5", "M15"):
        r = resample(d, timeframe_to_rule(tf))
        print(tf, "共", len(r), "根")
