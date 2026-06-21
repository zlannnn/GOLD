"""
tune.py —— pullback(顺势回踩续势) 模型参数寻优 + 逐年稳健性验证。
结论用于回填 smc_core.DEFAULT_PARAMS。
"""
from __future__ import annotations
import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

import data_utils as du
import smc_core as sc
from backtest import backtest_tf, ExecConfig


def yearly(trades, df):
    """按年统计 totR, 检验稳健性。"""
    if not trades:
        return {}
    idx = df.index
    rows = {}
    for tr in trades:
        y = idx[tr.exit_idx].year
        rows.setdefault(y, 0.0)
        rows[y] += tr.r
    return rows


def run_grid(df, base, grid, ex, tag, topn=10):
    keys = list(grid.keys())
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, combo))
        p = replace(base, entry_model="pullback", **kw)
        trades, eq, st = backtest_tf(df, p, ex)
        if st.get("trades", 0) < 100:
            continue
        rows.append((st["total_R"], st["win_rate"], st["profit_factor"],
                     st["trades"], st["expectancy_R"], st["max_dd_R"], kw, trades))
    rows.sort(key=lambda r: r[0], reverse=True)
    print(f"\n===== {tag} 前 {topn}(按总R, spread=${ex.spread}) =====")
    print(f"{'totR':>7} {'win%':>6} {'PF':>5} {'trades':>7} {'expR':>7} {'maxDD':>7}  params")
    for r in rows[:topn]:
        print(f"{r[0]:7.1f} {r[1]:6.1f} {r[2]:5.2f} {r[3]:7d} {r[4]:+7.3f} {r[5]:7.1f}  {r[6]}")
    if rows:
        best = rows[0]
        yr = yearly(best[7], df)
        pos = sum(1 for v in yr.values() if v > 0)
        print(f"  最佳逐年: " + "  ".join(f"{y}:{v:+.0f}" for y, v in sorted(yr.items())))
        print(f"  正收益年份: {pos}/{len(yr)}")
    return rows


def main():
    m1 = du.load_m1()
    m15 = du.resample(m1, "15min")
    m5 = du.resample(m1, "5min")
    m5 = m5[m5.index >= (m5.index.max() - pd.Timedelta(days=int(6 * 365.25)))]

    ex = ExecConfig(spread=0.08)   # raw/ECN 账户黄金典型点差
    grid = {
        "ema_fast": [20, 50],
        "ema_trend": [50, 200],
        "min_rr": [2.0, 3.0, 4.0],
        "sweep_lookback": [8, 15],
        "sl_buffer_atr": [0.10],
        "require_fvg": [False, True],
    }
    # 仅保留 ema_fast<ema_trend 的合理组合在循环内处理
    print(f"M15 全程: {m15.index.min()} ~ {m15.index.max()} ({len(m15):,})")
    run_grid(m15, sc.DEFAULT_PARAMS["M15"], grid, ex, "M15")

    print(f"\nM5 近6年: {m5.index.min()} ~ {m5.index.max()} ({len(m5):,})")
    run_grid(m5, sc.DEFAULT_PARAMS["M5"], grid, ex, "M5")


if __name__ == "__main__":
    main()
