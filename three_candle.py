"""
three_candle.py
================
短线 (M1 / M5) "关键位 + 3根K线" 反应式进场策略。

设计动机 (用户需求)
--------------------
- 在 *大周期* (H1 / H4) 上取 **关键价格** (摆动高低点 = 流动性/支撑阻力);
- 价格经过 *较长一段上涨或下跌* 后, 来到关键位附近;
- 出现 3 根 K 线形态:
    第1根: 到位/试探 K 线 (确立一个区间);
    第2根: 收盘落在第1根区间内 (停顿/收缩, 多空暂时平衡);
    第3根: 一根 **饱满** 的顺势 K 线 (实体占比大、有体量, 收盘突破前两根基底);
- 此时顺第3根方向 *介入一笔*, **止损放在形态外侧(很小)**, 以小止损博取较高盈亏比 (RR)。

为什么关键位必须来自大周期?
- 小周期(1/5min)噪声极大, 任意位置的 3K 线信号几乎随机;
- 只有当形态发生在 H1/H4 这种 *有意义的关键位* 上时, 反应才具备统计意义(机构流动性所在)。

与项目其余部分的关系
- 完全复用 smc_core 的原语 (atr / find_pivots / Signal) 和 data_utils 的重采样;
- precompute / evaluate_entry 的签名与 smc_core 一致, 因此可直接传给
  backtest.backtest_tf(df, p, ex, strat=three_candle) 复用同一套回测引擎与三原则结算;
- generate_signal(df, p) 供实盘机器人调用 (与 smc_core.generate_signal 同形)。

严格无未来函数
- H1/H4 摆动点在其右侧 `right` 根高周期K线收盘后才"确认", 通过 searchsorted 映射到
  "第一根能看到它的工作K线", 之后才纳入关键位集合;
- 形态只用已收盘 K 线 (c1=t-2, c2=t-1, c3=t), 在第3根收盘瞬间判定。
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import data_utils as du
from smc_core import atr, find_pivots, Signal


# ----------------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------------

@dataclass
class ThreeCandleParams:
    """3K线短线策略参数 (与 smc_core.StrategyParams 字段名兼容回测引擎所需部分)。"""
    name: str = "M5_3C"

    # ---- 大周期关键位: (周期分钟, 左, 右) 列表; 取这些周期上的摆动高/低点作为关键价 ----
    htf_levels: tuple = ((60, 3, 3), (240, 3, 3))   # H1 与 H4 的摆动点
    level_tol_atr: float = 0.6      # 形态极值"贴近"关键位的容差 = level_tol_atr * ATR

    # ---- "较长一段上涨/下跌"的接近段 ----
    trend_lookback: int = 12        # 形态之前用于度量接近段的K线数
    trend_min_atr: float = 2.0      # 接近段净位移须 >= 该值 * ATR (才算"较长一段")

    # ---- 3K线形态 ----
    c3_body_frac: float = 0.6       # 第3根实体/振幅 >= 此值 (饱满)
    c3_min_atr: float = 0.8         # 第3根振幅 >= 此值 * ATR (有体量)
    c3_break: bool = True           # 第3根收盘须突破前两根基底极值 (动量确认)

    # ---- 首次触及过滤: 刚抵达的关键位反弹质量通常远好于反复触及 ----
    require_fresh: bool = True       # 仅在该关键位"近期未被触及"时进场
    fresh_lookback: int = 30         # 视为"近期"的回看窗口(K线数)

    # ---- 大周期趋势过滤(顺大逆小): 只在 H4 顺势方向做关键位回踩 ----
    trend_filter: bool = True        # True: H4 上行只做多回踩, 下行只做空反抽
    trend_tf: int = 240              # 趋势所用周期(分钟), 默认 H4
    trend_ema: int = 50              # 趋势 EMA 长度(在 trend_tf 周期上)

    # ---- 进场方式 (PA stop 突破单) ----
    # "stop": 在第3根极值外 stop_buf_ticks 个tick挂突破止损单, 仅随后 stop_valid_bars 根有效,
    #         未触发即撤(避免"第3根即山顶/谷底随即反转"的坏单)。回测验证: 提升 3K线期望与胜率。
    # "market": 第3根收盘市价进场(旧行为)。
    entry_type: str = "stop"
    stop_buf_ticks: int = 1
    stop_valid_bars: int = 2        # 仅下一/下两根有效
    tick_size: float = 0.01         # 黄金最小报价变动

    # ---- 指标 / 风控 ----
    atr_period: int = 14
    sl_buffer_atr: float = 0.20     # 止损放在形态极值外侧的缓冲 = 此值 * ATR
    min_stop_atr: float = 0.30      # 止损距离下限(ATR倍数), 太近易被噪声打掉
    max_stop_atr: float = 3.00      # 止损距离上限(ATR倍数)
    target_rr: float = 2.5          # 目标盈亏比 (TP = 入场 ± target_rr * 风险)

    # ---- 回测引擎兼容字段 ----
    min_rr: float = 2.5             # 引擎/加仓校验用; 本策略 TP 固定为 target_rr
    allow_scale_in: bool = False    # 短线一击, 默认不加仓
    max_positions: int = 1

    # 交易时段过滤 (UTC 小时, 空 = 全天)
    sessions_utc: tuple = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)


# ----------------------------------------------------------------------------
# 大周期关键位 -> 因果对齐到工作周期
# ----------------------------------------------------------------------------

def _level_events(df_working: pd.DataFrame, htf_levels) -> list[tuple[int, float]]:
    """
    在每个高周期上找摆动高/低点(关键位), 返回 (确认时的工作K线索引, 关键价) 列表。
    一个位于高周期 i 的摆动点, 在 i+right 根高周期K线收盘后确认; 映射到
    "第一根时间戳 >= 该确认收盘时刻的工作K线"才可见 (无未来函数)。
    """
    work_times = df_working.index.values  # datetime64[ns], 右沿(收盘)时刻, 升序
    n = len(df_working)
    events: list[tuple[int, float]] = []
    for tf_min, left, right in htf_levels:
        r = du.resample(df_working, f"{tf_min}min")
        if len(r) <= left + right:
            continue
        hh = r["high"].to_numpy(dtype=float)
        ll = r["low"].to_numpy(dtype=float)
        htf_times = r.index.values
        ph_idx, pl_idx = find_pivots(hh, ll, left, right)
        for i in ph_idx:
            j = i + right
            if j >= len(r):
                continue
            w = int(np.searchsorted(work_times, htf_times[j], side="left"))
            if w < n:
                events.append((w, float(hh[i])))
        for i in pl_idx:
            j = i + right
            if j >= len(r):
                continue
            w = int(np.searchsorted(work_times, htf_times[j], side="left"))
            if w < n:
                events.append((w, float(ll[i])))
    events.sort(key=lambda e: e[0])
    return events


def _nearest(sorted_levels: list[float], x: float) -> float:
    """在升序 levels 中找与 x 最接近的值; 空则 NaN。"""
    if not sorted_levels:
        return np.nan
    i = bisect.bisect_left(sorted_levels, x)
    best = np.nan
    bd = np.inf
    for k in (i - 1, i):
        if 0 <= k < len(sorted_levels):
            d = abs(sorted_levels[k] - x)
            if d < bd:
                bd, best = d, sorted_levels[k]
    return best


# ----------------------------------------------------------------------------
# 预计算 (与 smc_core.precompute 同形, 供 backtest 引擎调用)
# ----------------------------------------------------------------------------

def precompute(df: pd.DataFrame, p: ThreeCandleParams) -> dict:
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    n = len(c)
    _atr = atr(h, l, c, p.atr_period)

    # 大周期趋势 EMA (顺大逆小过滤用), 真·多周期因果对齐到工作图
    if p.trend_filter and isinstance(df.index, pd.DatetimeIndex):
        base_min = du.infer_tf_minutes(df.index)
        htf_trend = du.mtf_ema_aligned(df, base_min, p.trend_tf, p.trend_ema)
    else:
        htf_trend = np.full(n, np.nan)

    # 关键位: 因果地为每根工作K线计算"与其 low / high 最接近的已知关键价"
    lvl_at_low = np.full(n, np.nan)
    lvl_at_high = np.full(n, np.nan)
    if isinstance(df.index, pd.DatetimeIndex) and p.htf_levels:
        events = _level_events(df, p.htf_levels)
        active: list[float] = []
        ptr = 0
        ne = len(events)
        for t in range(n):
            while ptr < ne and events[ptr][0] <= t:
                bisect.insort(active, events[ptr][1])
                ptr += 1
            if active:
                lvl_at_low[t] = _nearest(active, l[t])
                lvl_at_high[t] = _nearest(active, h[t])

    if isinstance(df.index, pd.DatetimeIndex) and len(p.sessions_utc) > 0:
        hours = df.index.hour.to_numpy()
        in_session = np.isin(hours, np.array(p.sessions_utc))
    else:
        in_session = np.ones(n, dtype=bool)

    return {
        "open": o, "high": h, "low": l, "close": c,
        "atr": _atr, "htf_trend": htf_trend,
        "lvl_at_low": lvl_at_low, "lvl_at_high": lvl_at_high,
        "in_session": in_session,
        "n": n,
    }


# ----------------------------------------------------------------------------
# 进场评估 (第3根K线收盘瞬间)
# ----------------------------------------------------------------------------

def _is_fresh(arr: dict, ref_bar: int, level: float, tol: float, lookback: int) -> bool:
    """关键位是否"近期未被触及": 形态前 lookback 根内没有任何K线触及该 level。
    触及 = 该K线区间 [low,high] 覆盖到 [level-tol, level+tol]。"""
    h, l = arr["high"], arr["low"]
    lo = max(0, ref_bar - lookback)
    for k in range(lo, ref_bar - 2):     # 不含形态本身的 3 根
        if l[k] <= level + tol and h[k] >= level - tol:
            return False
    return True


def evaluate_entry(arr: dict, t: int, p: ThreeCandleParams) -> Optional[Signal]:
    n = arr["n"]
    need = p.trend_lookback + 3
    if t < need or t >= n:
        return None
    if not arr["in_session"][t]:
        return None
    a = arr["atr"][t]
    if not np.isfinite(a) or a <= 0:
        return None

    o, h, l, c = arr["open"], arr["high"], arr["low"], arr["close"]
    c1, c2, c3 = t - 2, t - 1, t
    tol = p.level_tol_atr * a
    htf = arr["htf_trend"][t]
    if p.trend_filter and not np.isfinite(htf):
        return None
    up_ok = (not p.trend_filter) or (c[c3] > htf)     # H4 上行 -> 允许做多
    dn_ok = (not p.trend_filter) or (c[c3] < htf)     # H4 下行 -> 允许做空
    rng3 = h[c3] - l[c3]
    if rng3 <= 0:
        return None
    body3 = abs(c[c3] - o[c3])

    # 第2根收盘须落在第1根区间内 (停顿/收缩) —— 多空通用
    c2_inside = (l[c1] <= c[c2] <= h[c1])
    # 第3根须饱满 + 有体量
    c3_full = (body3 >= p.c3_body_frac * rng3) and (rng3 >= p.c3_min_atr * a)
    if not (c2_inside and c3_full):
        return None

    base_ref = c[t - 2 - p.trend_lookback]  # 接近段起点(形态前 lookback 根的收盘)

    # ---------------- 做多: 下跌进入关键支撑, 第3根饱满阳线反弹 ----------------
    if up_ok and c[c3] > o[c3]:
        dip = c1 + int(np.argmin([l[c1], l[c2], l[c3]]))
        patt_low = l[dip]
        sup = arr["lvl_at_low"][dip]
        approach_down = (base_ref - patt_low) >= p.trend_min_atr * a
        fresh = (not p.require_fresh) or _is_fresh(arr, dip, sup, tol, p.fresh_lookback)
        if (np.isfinite(sup) and abs(patt_low - sup) <= tol and c[c3] > sup
                and approach_down and fresh
                and (not p.c3_break or c[c3] > max(h[c1], h[c2]))):
            sl = patt_low - p.sl_buffer_atr * a
            stop = (p.entry_type == "stop")
            trig = h[c3] + p.stop_buf_ticks * p.tick_size
            entry = trig if stop else c[c3]      # stop: 以触发价为成本基准
            risk = entry - sl
            if p.min_stop_atr * a <= risk <= p.max_stop_atr * a:
                tp = entry + p.target_rr * risk
                return Signal("long", entry, sl, tp, p.target_rr,
                              f"关键支撑{sup:.1f}·3K{'突破' if stop else '反弹'}做多",
                              np.nan, np.nan, sup,
                              entry_type=p.entry_type,
                              trigger=trig if stop else np.nan,
                              valid_bars=p.stop_valid_bars)

    # ---------------- 做空: 上涨进入关键阻力, 第3根饱满阴线回落 ----------------
    if dn_ok and c[c3] < o[c3]:
        top = c1 + int(np.argmax([h[c1], h[c2], h[c3]]))
        patt_high = h[top]
        res = arr["lvl_at_high"][top]
        approach_up = (patt_high - base_ref) >= p.trend_min_atr * a
        fresh = (not p.require_fresh) or _is_fresh(arr, top, res, tol, p.fresh_lookback)
        if (np.isfinite(res) and abs(patt_high - res) <= tol and c[c3] < res
                and approach_up and fresh
                and (not p.c3_break or c[c3] < min(l[c1], l[c2]))):
            sl = patt_high + p.sl_buffer_atr * a
            stop = (p.entry_type == "stop")
            trig = l[c3] - p.stop_buf_ticks * p.tick_size
            entry = trig if stop else c[c3]
            risk = sl - entry
            if p.min_stop_atr * a <= risk <= p.max_stop_atr * a:
                tp = entry - p.target_rr * risk
                return Signal("short", entry, sl, tp, p.target_rr,
                              f"关键阻力{res:.1f}·3K{'突破' if stop else '回落'}做空",
                              np.nan, np.nan, res,
                              entry_type=p.entry_type,
                              trigger=trig if stop else np.nan,
                              valid_bars=p.stop_valid_bars)

    return None


# ----------------------------------------------------------------------------
# 实盘便捷封装
# ----------------------------------------------------------------------------

def generate_signal(df: pd.DataFrame, p: ThreeCandleParams) -> Optional[Signal]:
    """实盘: 传入最近 N 根已收盘K线, 返回最后一根收盘时的信号或 None。
    注意: N 必须足够覆盖 H4 关键位形成(建议 >= 该周期下 240min*几十根)。"""
    if len(df) < (p.trend_lookback + 5):
        return None
    arr = precompute(df, p)
    return evaluate_entry(arr, arr["n"] - 1, p)


# 默认参数 (经 _scan3c 全样本 2012-2024 + 前后半段稳健性筛选)。
# 关键结论(诚实):
#   1) 优势来自四个条件叠加 —— H4 趋势过滤(顺大逆小) + 精确贴住 H4 关键位 +
#      较长接近段(>=3.5ATR) + 首次触及; 缺少 H4 趋势过滤则前半段(2012-17)为负(过拟合近年)。
#   2) 进场用 PA stop 突破单(entry_type="stop"): 第3根极值外1tick挂停损突破单, 仅随后
#      stop_valid_bars 根有效, 未触发即撤。要求"下一根继续突破"过滤掉随即反转的坏单,
#      A/B 实验证明可提升期望与胜率(M5: 28.6%->32.4%, +0.076->+0.084R/信号; M1 提升更大)。
#   3) 关键位来源: H4+H1 是频率↔质量的甜点(频率+35~85%, 期望基本不掉, 总R最高);
#      再往下纳入 M30/M15 则容差(0.4*工作ATR)相对其摆动点太宽, 过滤失去选择性, 期望崩塌甚至转负。
#   4) M1 噪声/成本更高, 期望更薄, 列为可选, 建议优先 M5。
DEFAULT_3C = {
    "M5_3C": ThreeCandleParams(
        name="M5_3C",
        htf_levels=((240, 3, 3), (60, 3, 3)),   # H4+H1 关键位(频率↑且期望基本不掉, 总R最高, 11/13年正)
        level_tol_atr=0.4, trend_lookback=12, trend_min_atr=3.5,
        c3_body_frac=0.6, c3_min_atr=0.8, c3_break=True,
        require_fresh=True, fresh_lookback=20,   # 放宽自50: 频率↑/期望↑/回撤↓且前后半段更均衡(更稳健)
        trend_filter=True, trend_tf=240, trend_ema=50,
        entry_type="stop", stop_buf_ticks=1, stop_valid_bars=2, tick_size=0.01,
        atr_period=14, sl_buffer_atr=0.2, min_stop_atr=0.3, max_stop_atr=3.0,
        target_rr=3.0, min_rr=3.0, allow_scale_in=False, max_positions=1,
    ),
    # M1: 同族参数, 但 1 分钟噪声/成本更高, 期望更薄, 仅作可选(建议优先 M5)。
    # M1 实验最优为 stop 仅下一根有效(stop_valid_bars=1)。
    "M1_3C": ThreeCandleParams(
        name="M1_3C",
        htf_levels=((240, 3, 3), (60, 3, 3)),   # H4+H1 关键位(频率约翻倍, 期望仍高 +0.118R)
        level_tol_atr=0.4, trend_lookback=15, trend_min_atr=3.5,
        c3_body_frac=0.6, c3_min_atr=0.9, c3_break=True,
        require_fresh=True, fresh_lookback=30,   # 放宽自60: 与M5方向一致, 增频且期望基本维持
        trend_filter=True, trend_tf=240, trend_ema=50,
        entry_type="stop", stop_buf_ticks=1, stop_valid_bars=1, tick_size=0.01,
        atr_period=14, sl_buffer_atr=0.2, min_stop_atr=0.3, max_stop_atr=3.0,
        target_rr=3.0, min_rr=3.0, allow_scale_in=False, max_positions=1,
    ),
}


def _run():
    """快速回测: M1(近2年) 与 M5(近6年), 含点差敏感性与逐年。"""
    import three_candle as tc
    from backtest import backtest_tf, ExecConfig

    def yearly(trades, df):
        out = {}
        for tr in trades:
            y = df.index[tr.exit_idx].year
            out[y] = out.get(y, 0.0) + tr.r
        return out

    m1 = du.load_m1()
    m5 = du.resample(m1, "5min")
    defs = [
        ("M5_3C", m5[m5.index >= m5.index.max() - pd.Timedelta(days=int(6 * 365.25))]),
        ("M1_3C", m1[m1.index >= m1.index.max() - pd.Timedelta(days=int(2 * 365.25))]),
    ]
    for key, df in defs:
        p = DEFAULT_3C[key]
        print(f"\n==== {key}  {df.index.min()} ~ {df.index.max()}  ({len(df):,} 根) ====")
        for sp in [0.05, 0.10, 0.20]:
            tr, eq, st = backtest_tf(df, p, ExecConfig(spread=sp), strat=tc)
            if st.get("trades", 0) == 0:
                print(f"  spread={sp:.2f}: 无交易"); continue
            print("  spread=%.2f trades=%4d win=%.1f%% expR=%+.3f PF=%.2f totR=%+.1f maxDD=%.1f"
                  % (sp, st["trades"], st["win_rate"], st["expectancy_R"],
                     st["profit_factor"], st["total_R"], st["max_dd_R"]))
        tr, eq, st = backtest_tf(df, p, ExecConfig(spread=0.10), strat=tc)
        if st.get("trades", 0) > 0:
            yr = yearly(tr, df)
            pos = sum(1 for v in yr.values() if v > 0)
            print("  逐年R(spread0.10): " + " ".join(f"{y}:{v:+.0f}" for y, v in sorted(yr.items())))
            print(f"  正收益年 {pos}/{len(yr)}  多 {st['long_n']}笔/{st['long_total_R']:+.0f}R  "
                  f"空 {st['short_n']}笔/{st['short_total_R']:+.0f}R")


if __name__ == "__main__":
    _run()
