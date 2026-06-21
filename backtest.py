"""
backtest.py
================
事件驱动回测引擎, 用于验证 smc_core 的交易逻辑在 1/5/15min 三个周期上的真实表现。
回测与实盘共用 smc_core.evaluate_entry, 保证逻辑一致。

严格遵守用户三原则:
  1. 开仓即带止盈止损, 且每笔的盈亏比 RR >= min_rr (>=1)。
  2. 允许加仓(金字塔式), 但新仓止损被钳制为"不差于初始仓位止损";
     若钳制后 RR 仍 >= min_rr 才允许加仓, 否则放弃该次加仓。
  3. 仅在每根 K 线收盘瞬间评估进场(evaluate_entry 只看已收盘 K 线);
     成交/止盈止损在随后 K 线用其最高最低价撮合(同根内若同时触及, 保守按先止损处理)。

输出 (reports/):
  - backtest_report.md
  - equity_<TF>.png

运行:
  python backtest.py                # 用各周期默认时间范围
  python backtest.py --years 6      # 只回测最近 6 年(更快)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_utils as du
import smc_core as sc

REPORT_DIR = r"E:\fun\GOLD\reports"
os.makedirs(REPORT_DIR, exist_ok=True)


@dataclass
class ExecConfig:
    """撮合 / 成本设置。"""
    spread: float = 0.10          # 点差成本(美元/笔). raw/ECN 账户黄金约 0.05~0.15; 标准账户约 0.20~0.35
    scale_trigger_r: float = 0.5  # 加仓条件: 浮盈达到 0.5R 才允许金字塔加仓
    sl_first: bool = True         # 同根同时触及止盈止损时, 保守按先止损


@dataclass
class Trade:
    side: str
    entry_idx: int
    entry: float
    sl: float
    tp: float
    risk: float        # 初始风险(美元) = |entry - sl|
    is_scale: bool
    exit_idx: int = -1
    exit: float = np.nan
    r: float = np.nan  # 盈亏(以初始风险 R 计)
    reason: str = ""


def backtest_tf(df: pd.DataFrame, p, ex: ExecConfig, strat=sc):
    """对单一周期做回测, 返回 (trades, equity_series, stats)。

    strat: 提供 precompute(df, p) 与 evaluate_entry(arr, t, p) 的策略模块,
           默认 smc_core; 短线 3K 线策略传入 three_candle 即可复用本引擎。
    """
    arr = strat.precompute(df, p)
    n = arr["n"]
    h, l, c = arr["high"], arr["low"], arr["close"]
    half_sp = ex.spread / 2.0

    open_trades: list[Trade] = []
    closed: list[Trade] = []
    pending: list[dict] = []   # 挂单(stop突破单): 仅在有效期内等待触发, 超时撤单

    def initial_sl_for_side(side):
        """同方向已有仓位中, 最初(最差)的止损价。"""
        sls = [t.sl for t in open_trades if t.side == side]
        if not sls:
            return None
        # 做多: 初始止损是最低的那个(最差); 做空: 最高的那个
        return min(sls) if side == "long" else max(sls)

    for t in range(n):
        # ---- 1) 先用当根 K 线撮合已有持仓的止盈止损 ----
        still_open = []
        for tr in open_trades:
            if t <= tr.entry_idx:
                still_open.append(tr)
                continue
            hit_sl = hit_tp = False
            if tr.side == "long":
                hit_sl = l[t] <= tr.sl
                hit_tp = h[t] >= tr.tp
            else:
                hit_sl = h[t] >= tr.sl
                hit_tp = l[t] <= tr.tp
            if hit_sl and hit_tp:
                # 同根同时触及 -> 保守按先止损
                if ex.sl_first:
                    hit_tp = False
                else:
                    hit_sl = False
            if hit_sl:
                tr.exit_idx = t; tr.exit = tr.sl; tr.reason = "SL"
                pnl = (tr.exit - tr.entry) if tr.side == "long" else (tr.entry - tr.exit)
                tr.r = (pnl - ex.spread) / tr.risk
                closed.append(tr)
            elif hit_tp:
                tr.exit_idx = t; tr.exit = tr.tp; tr.reason = "TP"
                pnl = (tr.exit - tr.entry) if tr.side == "long" else (tr.entry - tr.exit)
                tr.r = (pnl - ex.spread) / tr.risk
                closed.append(tr)
            else:
                still_open.append(tr)
        open_trades = still_open

        # ---- 1b) 处理挂单(stop突破单): 当根是否触发 / 是否超时撤单 ----
        if pending:
            keep = []
            for od in pending:
                if t > od["expire"]:
                    continue   # 超时撤单
                triggered = (h[t] >= od["trigger"]) if od["side"] == "long" \
                    else (l[t] <= od["trigger"])
                if not triggered:
                    keep.append(od)
                    continue
                # 不与反向持仓对冲; 同向已满仓则放弃
                opp = "short" if od["side"] == "long" else "long"
                if any(tr.side == opp for tr in open_trades):
                    continue
                if any(tr.side == od["side"] for tr in open_trades):
                    continue
                # 跳空则按开盘价成交, 否则按触发价
                if od["side"] == "long":
                    entry = max(arr["open"][t], od["trigger"])
                else:
                    entry = min(arr["open"][t], od["trigger"])
                risk = abs(entry - od["sl"])
                if risk <= 0:
                    continue
                open_trades.append(Trade(od["side"], t, entry, od["sl"], od["tp"],
                                         risk, False, reason=od["reason"]))
            pending = keep

        # ---- 2) 当根收盘评估进场信号 ----
        sig = strat.evaluate_entry(arr, t, p)
        if sig is None:
            continue

        # stop 突破单: 不立即成交, 登记为挂单, 等随后 valid_bars 根内触发(超时撤单)
        if getattr(sig, "entry_type", "market") == "stop":
            opp = "short" if sig.side == "long" else "long"
            if any(tr.side == opp for tr in open_trades):
                continue
            if any(tr.side == sig.side for tr in open_trades):
                continue   # 已有同向持仓(max_positions=1 短线), 不重复挂单
            if any(od["side"] == sig.side for od in pending):
                continue   # 已有同向挂单, 不叠加
            if np.isfinite(sig.trigger):
                pending.append(dict(side=sig.side, trigger=sig.trigger, sl=sig.sl,
                                    tp=sig.tp, reason=sig.reason,
                                    expire=t + int(sig.valid_bars)))
            continue

        # 不与反方向持仓对冲: 若已有反向持仓则跳过
        opp = "short" if sig.side == "long" else "long"
        if any(tr.side == opp for tr in open_trades):
            continue

        same = [tr for tr in open_trades if tr.side == sig.side]

        # 说明: RR 在信号层(中间价)已保证 >= min_rr; 点差作为成本在结算时扣除,
        # 不在此处用滑点后的价格重算 RR(否则会错误地拒绝固定 RR 的信号)。
        if not same:
            # 首仓: 按信号给定的(中间价) entry/sl/tp
            entry, sl, tp = sig.entry, sig.sl, sig.tp
            risk = abs(entry - sl)
            if risk <= 0:
                continue
            open_trades.append(Trade(sig.side, t, entry, sl, tp, risk, False, reason=sig.reason))
        else:
            # 加仓(金字塔)
            if not p.allow_scale_in or len(same) >= p.max_positions:
                continue
            first = same[0]
            # 条件: 价格已朝有利方向走出 >= scale_trigger_r 的浮盈
            favor = (sig.entry - first.entry) if sig.side == "long" else (first.entry - sig.entry)
            if favor < ex.scale_trigger_r * first.risk:
                continue
            # 原则2: 钳制新仓止损, 不得差于初始仓位止损
            init_sl = initial_sl_for_side(sig.side)
            entry = sig.entry
            sl = max(sig.sl, init_sl) if sig.side == "long" else min(sig.sl, init_sl)
            tp = sig.tp
            risk = abs(entry - sl)
            if risk <= 0:
                continue
            rr = (tp - entry) / risk if sig.side == "long" else (entry - tp) / risk
            if rr < p.min_rr:
                continue  # 钳制止损后盈亏比不足, 放弃加仓(原则1)
            open_trades.append(Trade(sig.side, t, entry, sl, tp, risk, True, reason="scale:" + sig.reason))

    # 回测结束, 未平仓的按最后收盘价结算(标记)
    for tr in open_trades:
        tr.exit_idx = n - 1; tr.exit = c[-1]; tr.reason = "EOD"
        pnl = (tr.exit - tr.entry) if tr.side == "long" else (tr.entry - tr.exit)
        tr.r = (pnl - ex.spread) / tr.risk
        closed.append(tr)

    closed.sort(key=lambda x: x.exit_idx)
    stats = compute_stats(closed, df)
    eq = np.cumsum([tr.r for tr in closed]) if closed else np.array([])
    return closed, eq, stats


def compute_stats(trades: list[Trade], df: pd.DataFrame) -> dict:
    if not trades:
        return {"trades": 0}
    rs = np.array([tr.r for tr in trades])
    wins = rs[rs > 0]; losses = rs[rs <= 0]
    eq = np.cumsum(rs)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    n_first = sum(1 for tr in trades if not tr.is_scale)
    n_scale = sum(1 for tr in trades if tr.is_scale)
    longs = [tr for tr in trades if tr.side == "long"]
    shorts = [tr for tr in trades if tr.side == "short"]
    span_years = max((df.index.max() - df.index.min()).days / 365.25, 1e-9)
    return {
        "trades": len(trades),
        "first_entries": n_first,
        "scale_ins": n_scale,
        "win_rate": len(wins) / len(rs) * 100,
        "avg_R": float(rs.mean()),
        "expectancy_R": float(rs.mean()),
        "total_R": float(rs.sum()),
        "profit_factor": float(pf),
        "max_dd_R": float(dd.max()) if len(dd) else 0.0,
        "avg_win_R": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_R": float(losses.mean()) if len(losses) else 0.0,
        "trades_per_year": len(trades) / span_years,
        "long_n": len(longs), "long_total_R": float(sum(t.r for t in longs)),
        "short_n": len(shorts), "short_total_R": float(sum(t.r for t in shorts)),
        "start": df.index.min(), "end": df.index.max(),
    }


def plot_equity(eq: np.ndarray, tf: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq, color="#1565c0")
    ax.set_title(f"{tf} 累计权益曲线 (单位: R, 每笔风险=1R)")
    ax.set_xlabel("交易序号"); ax.set_ylabel("累计 R")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, f"equity_{tf}.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=None, help="只回测最近 N 年")
    ap.add_argument("--csv", type=str, default=du.DEFAULT_M1_CSV)
    args = ap.parse_args()

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    print("加载 1 分钟历史数据 ...")
    m1 = du.load_m1(args.csv)
    print(f"  完成: {m1.index.min()} -> {m1.index.max()}, 共 {len(m1):,} 根\n")

    # 各周期默认回测范围(控制运行时间; M1 数据量大默认仅近 2 年)
    default_years = {"M1": 2.0, "M5": 6.0, "M15": 12.0}

    tf_defs = [
        ("M5", du.resample(m1, "5min")),
        ("M15", du.resample(m1, "15min")),
        ("M1", m1),
    ]

    ex = ExecConfig()
    lines = ["# 黄金 SMC/ICT 策略回测报告\n",
             "> 规则: 开仓即带止盈止损, 每笔 RR≥min_rr; 加仓止损不差于初始仓位; 仅K线收盘评估。",
             f"> 成本: 点差 ${ex.spread}/笔, 加仓触发浮盈 {ex.scale_trigger_r}R。\n"]

    for tf, df in tf_defs:
        yrs = args.years if args.years else default_years[tf]
        if yrs:
            cutoff = df.index.max() - pd.Timedelta(days=int(yrs * 365.25))
            df = df[df.index >= cutoff]
        p = sc.DEFAULT_PARAMS[tf]
        print(f"==== 回测 {tf}: {df.index.min()} ~ {df.index.max()} ({len(df):,} 根) ====")
        trades, eq, st = backtest_tf(df, p, ex)
        path = plot_equity(eq, tf) if len(eq) else ""
        print(f"  交易 {st.get('trades',0)} 笔, 胜率 {st.get('win_rate',0):.1f}%, "
              f"总R {st.get('total_R',0):.1f}, PF {st.get('profit_factor',0):.2f}\n")

        if st.get("trades", 0) == 0:
            lines += [f"\n## {tf}: 无交易(检查参数)"]
            continue

        lines += [
            f"\n## {tf} 周期回测结果",
            f"- 区间: {st['start']} ~ {st['end']}   参数: swing={p.swing_left}/{p.swing_right}, "
            f"ATR={p.atr_period}, min_RR={p.min_rr}, 时段UTC={p.sessions_utc}",
            f"- 总交易: **{st['trades']}** 笔 (首仓 {st['first_entries']}, 加仓 {st['scale_ins']}) "
            f"≈ {st['trades_per_year']:.0f} 笔/年",
            f"- **胜率: {st['win_rate']:.1f}%**   期望值: **{st['expectancy_R']:.3f} R/笔**   "
            f"盈利因子 PF: **{st['profit_factor']:.2f}**",
            f"- 累计收益: **{st['total_R']:.1f} R**   最大回撤: {st['max_dd_R']:.1f} R",
            f"- 平均盈利 {st['avg_win_R']:.2f}R / 平均亏损 {st['avg_loss_R']:.2f}R",
            f"- 多头 {st['long_n']} 笔(累计 {st['long_total_R']:.1f}R) / "
            f"空头 {st['short_n']} 笔(累计 {st['short_total_R']:.1f}R)",
            f"- 权益曲线: {os.path.basename(path)}",
        ]

    report = os.path.join(REPORT_DIR, "backtest_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"回测报告已写入: {report}")


if __name__ == "__main__":
    main()
