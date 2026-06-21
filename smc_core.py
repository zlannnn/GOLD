"""
smc_core.py
================
黄金 (XAUUSD) 智能资金概念 (Smart Money Concept) / ICT / 价格行为学
交易逻辑的"单一事实来源"(single source of truth)。

设计目标
--------
1. 完全自包含: 只依赖 pandas / numpy, 不依赖 pandas_ta / smartmoneyconcepts / MetaTrader5。
   这样 *分析脚本* 和 *回测脚本* 可以在没有 MT5 的纯 Python 环境直接运行,
   而 *实盘机器人* 复用完全相同的信号函数, 保证回测与实盘逻辑一致。

2. 提供两层 API:
   - 指标 / 结构原语:  atr / ema / rsi / find_pivots / find_fvg ...
   - 信号编排:        precompute(df, p)  -> 把一段 K 线预计算成 numpy 数组字典
                      evaluate_entry(arr, t, p) -> 判断第 t 根(已收盘)K线是否触发进场
   回测对整段历史调用 precompute 一次, 再逐根调用 evaluate_entry;
   实盘每根 K 线收盘后对最近窗口调用 precompute, 再对最后一根调用 evaluate_entry。

交易模型 (ICT / SMC 三要素)
--------------------------
方向(以做多为例, 做空对称):
  1. 流动性猎杀 (Liquidity Sweep): 价格先跌破前一个摆动低点(扫掉止损/卖方流动性);
  2. 性质转变 (CHoCH / Displacement): 随后出现强力上涨, 收回前低并制造 *公允价值缺口 FVG*,
     同时上破最近的次级摆动高点(结构由空转多);
  3. 回踩进场 (FVG / OB Retest): 价格回踩到 FVG/订单块区域并守住, 在当根 K 线收盘确认时进场。

风控(对应用户三原则):
  - 开仓即带止盈止损, 盈亏比 RR >= min_rr (默认 >= 1.5, 永远 >= 1);
  - 止损放在被扫流动性下方(做多)/上方(做空), 止盈取对侧流动性, 不足 RR 时按 min_rr 延伸;
  - 仅在 K 线收盘瞬间评估 (evaluate_entry 只看已收盘 K 线)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# 1. 基础指标 (纯 numpy / pandas 实现, 替代 pandas_ta)
# ----------------------------------------------------------------------------

def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """真实波幅 TR。"""
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    a = high - low
    b = np.abs(high - prev_close)
    c = np.abs(low - prev_close)
    return np.maximum(a, np.maximum(b, c))


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder 平滑 ATR。返回与输入等长的数组(前 period 根用累计均值填充)。"""
    tr = true_range(high, low, close)
    out = np.full_like(tr, np.nan, dtype=np.float64)
    if len(tr) == 0:
        return out
    # Wilder RMA: alpha = 1/period
    alpha = 1.0 / period
    acc = tr[0]
    out[0] = acc
    for i in range(1, len(tr)):
        acc = acc + alpha * (tr[i] - acc)
        out[i] = acc
    return out


def ema(values: np.ndarray, span: int) -> np.ndarray:
    """指数移动平均 (与 pandas ewm(span=...) 一致)。"""
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy()


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """相对强弱指数 RSI。"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()
    roll_down = pd.Series(loss).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()
    rs = np.divide(roll_up, roll_down, out=np.full_like(roll_up, np.inf), where=roll_down != 0)
    return 100.0 - (100.0 / (1.0 + rs))


# ----------------------------------------------------------------------------
# 2. 摆动高低点 (Swing / Pivot) —— 市场结构的基础
# ----------------------------------------------------------------------------

def find_pivots(high: np.ndarray, low: np.ndarray, left: int, right: int):
    """
    分形摆动点检测。
    bar i 是摆动高点: high[i] 是 [i-left, i+right] 区间内的最高;
    bar i 是摆动低点: low[i]  是 [i-left, i+right] 区间内的最低。
    摆动点在 i+right 根 K 线后才被"确认"(causal: 实盘只能用已确认的点)。

    返回:
        ph_idx, pl_idx : 摆动高点/低点的索引列表 (升序)
    """
    n = len(high)
    ph_idx, pl_idx = [], []
    for i in range(left, n - right):
        win_h = high[i - left:i + right + 1]
        win_l = low[i - left:i + right + 1]
        if high[i] == win_h.max() and (win_h == high[i]).sum() == 1:
            ph_idx.append(i)
        elif low[i] == win_l.min() and (win_l == low[i]).sum() == 1:
            pl_idx.append(i)
    return ph_idx, pl_idx


def confirmed_swing_arrays(high, low, left, right):
    """
    生成"截至第 t 根 K 线(已收盘)时, 最近一次已确认的摆动高/低点"前向填充数组。
    一个位于 i 的摆动点, 直到 i+right 根 K 线收盘才可见 (causal)。

    返回字典:
        sh_level[t], sh_idx[t] : 截至 t 已确认的最近摆动高点 价格 / 原始索引
        sl_level[t], sl_idx[t] : 截至 t 已确认的最近摆动低点 价格 / 原始索引
        prev_sh_level[t]       : 上上个摆动高点价格 (用于结构目标)
        prev_sl_level[t]
    """
    n = len(high)
    ph_idx, pl_idx = find_pivots(high, low, left, right)

    sh_level = np.full(n, np.nan)
    sh_idx = np.full(n, -1, dtype=np.int64)
    prev_sh_level = np.full(n, np.nan)
    sl_level = np.full(n, np.nan)
    sl_idx = np.full(n, -1, dtype=np.int64)
    prev_sl_level = np.full(n, np.nan)

    # 摆动高点: 在确认位置 (i+right) 写入, 之后前向填充
    last_lvl = np.nan
    last_idx = -1
    prev_lvl = np.nan
    ph_set = {i + right: i for i in ph_idx}  # confirm_pos -> pivot_idx
    for t in range(n):
        if t in ph_set:
            piv = ph_set[t]
            prev_lvl = last_lvl
            last_lvl = high[piv]
            last_idx = piv
        sh_level[t] = last_lvl
        sh_idx[t] = last_idx
        prev_sh_level[t] = prev_lvl

    last_lvl = np.nan
    last_idx = -1
    prev_lvl = np.nan
    pl_set = {i + right: i for i in pl_idx}
    for t in range(n):
        if t in pl_set:
            piv = pl_set[t]
            prev_lvl = last_lvl
            last_lvl = low[piv]
            last_idx = piv
        sl_level[t] = last_lvl
        sl_idx[t] = last_idx
        prev_sl_level[t] = prev_lvl

    return {
        "sh_level": sh_level, "sh_idx": sh_idx, "prev_sh_level": prev_sh_level,
        "sl_level": sl_level, "sl_idx": sl_idx, "prev_sl_level": prev_sl_level,
    }


# ----------------------------------------------------------------------------
# 3. 公允价值缺口 FVG (Fair Value Gap) —— ICT 进场区
# ----------------------------------------------------------------------------

def find_fvg(open_, high, low, close):
    """
    三K线公允价值缺口。以中间(位移)K线索引 m 记录, 在 m+1 收盘后可见 (causal)。
      看涨 FVG: high[m-1] < low[m+1]   -> 缺口区间 [high[m-1], low[m+1]]
      看跌 FVG: low[m-1]  > high[m+1]  -> 缺口区间 [high[m+1], low[m-1]]
    返回:
        fvg_type[m] : +1 看涨 / -1 看跌 / 0 无
        fvg_top[m], fvg_bot[m] : 缺口上下沿
    """
    n = len(high)
    fvg_type = np.zeros(n, dtype=np.int8)
    fvg_top = np.full(n, np.nan)
    fvg_bot = np.full(n, np.nan)
    for m in range(1, n - 1):
        if high[m - 1] < low[m + 1] and close[m] > open_[m]:
            fvg_type[m] = 1
            fvg_top[m] = low[m + 1]
            fvg_bot[m] = high[m - 1]
        elif low[m - 1] > high[m + 1] and close[m] < open_[m]:
            fvg_type[m] = -1
            fvg_top[m] = low[m - 1]
            fvg_bot[m] = high[m + 1]
    return fvg_type, fvg_top, fvg_bot


# ----------------------------------------------------------------------------
# 4. 策略参数
# ----------------------------------------------------------------------------

@dataclass
class StrategyParams:
    """单周期策略参数。不同周期(1/5/15min)使用不同实例。"""
    name: str = "M15"
    # 进场模型: "reversal" / "breakout" / "pullback" / "ma_bias"(多均线顺大逆小首触)
    entry_model: str = "reversal"
    # 结构
    swing_left: int = 3
    swing_right: int = 3
    # 多周期均线 (ma_bias 模型): 真正的多周期 -> (周期分钟, EMA长度) 列表。
    # 例如 ((5,50),(15,50),(60,50),(240,50)) = M5/M15/H1/H4 各自的 EMA50, 因果对齐到交易图。
    # 最慢周期(240=H4)为大级别方向锚, 次慢(60=H1)做斜率确认; 其余为"逆小"触及线。
    mtf_mas: tuple = ((5, 50), (15, 50), (60, 50), (240, 50))
    ft_lookback: int = 6          # "第一次触及"判定: 触及前该窗口内未触及过该均线
    ma_touch_buf: float = 0.05    # 触及容差 = ma_touch_buf * ATR (略微穿越也算触及)
    ma_require_resume: bool = True  # 是否要求收盘突破上一根极值(动量续势确认)
    # 指标
    atr_period: int = 14
    ema_trend: int = 0             # 趋势过滤 慢 EMA (0 = 关闭)
    ema_fast: int = 50             # 快 EMA (pullback 模型用作动态支撑/趋势判定)
    use_trend_filter: bool = False # True 时只顺 EMA 方向(反转策略一般关闭)
    # FVG / 位移 (作为可选的进场确认/汇合)
    require_fvg: bool = False      # True 时要求反转动作伴随同向 FVG(位移确认)
    fvg_min_atr: float = 0.3       # FVG 位移强度
    zone_valid_bars: int = 24      # 兼容字段 / FVG 有效期
    # 流动性猎杀
    sweep_lookback: int = 25       # 被扫流动性需是该窗口内的极值(确保是真实静置流动性)
    reject_ratio: float = 0.5      # 反转K线收盘需落在K线区间的有利 1-ratio 部分(拒绝插针)
    confirm_bars: int = 1          # 1 = 扫损即收回的同一根进场; 2 = 允许下一根收回确认
    # 风控
    min_rr: float = 1.5            # 最小盈亏比 (>=1, 用户硬性要求)
    sl_buffer_atr: float = 0.15    # 止损放在插针外侧的额外缓冲 = sl_buffer_atr * ATR
    min_stop_atr: float = 0.25     # 止损距离下限 (ATR 倍数), 防止过近被噪声打掉
    max_stop_atr: float = 3.5      # 止损距离上限 (ATR 倍数), 防止异常巨幅
    # 加仓
    allow_scale_in: bool = True
    max_positions: int = 3         # 同方向最多持仓数(含首仓)
    # 交易时段过滤 (UTC 小时, 空 = 全天). 默认伦敦+纽约活跃时段
    sessions_utc: tuple = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)


# ----------------------------------------------------------------------------
# 5. 预计算: 把 K 线 DataFrame 变成 numpy 数组字典
# ----------------------------------------------------------------------------

def precompute(df: pd.DataFrame, p: StrategyParams) -> dict:
    """
    把一段 OHLC(C) DataFrame 预计算为信号评估所需的 numpy 数组。
    df 需要列: open, high, low, close (可选 volume), index 为 DatetimeIndex 或可选。
    """
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)

    _atr = atr(h, l, c, p.atr_period)
    _ema = ema(c, p.ema_trend) if p.ema_trend and p.ema_trend > 0 else np.full(len(c), np.nan)
    _ema_fast = ema(c, p.ema_fast) if p.ema_fast and p.ema_fast > 0 else np.full(len(c), np.nan)

    # 真正的多周期均线带 (ma_bias 模型用): 各周期 EMA 因果对齐到工作周期, 按周期升序排列。
    mtf_cols, mtf_specs = [], []
    if p.entry_model == "ma_bias" and p.mtf_mas and isinstance(df.index, pd.DatetimeIndex):
        import data_utils as _du
        base_min = _du.infer_tf_minutes(df.index)
        for tf_min, per in sorted(p.mtf_mas, key=lambda x: x[0]):
            if tf_min < base_min:      # 比工作周期还细, 无法合成, 跳过
                continue
            mtf_cols.append(_du.mtf_ema_aligned(df, base_min, tf_min, per))
            mtf_specs.append((tf_min, per))
    ribbon = np.column_stack(mtf_cols) if mtf_cols else np.empty((len(c), 0))
    sw = confirmed_swing_arrays(h, l, p.swing_left, p.swing_right)
    fvg_type, fvg_top, fvg_bot = find_fvg(o, h, l, c)

    # 交易时段掩码
    if isinstance(df.index, pd.DatetimeIndex) and len(p.sessions_utc) > 0:
        hours = df.index.hour.to_numpy()
        in_session = np.isin(hours, np.array(p.sessions_utc))
    else:
        in_session = np.ones(len(c), dtype=bool)

    return {
        "open": o, "high": h, "low": l, "close": c,
        "atr": _atr, "ema": _ema, "ema_fast": _ema_fast,
        "ribbon": ribbon, "ribbon_periods": tuple(mtf_specs),
        "fvg_type": fvg_type, "fvg_top": fvg_top, "fvg_bot": fvg_bot,
        "in_session": in_session,
        "n": len(c),
        **sw,
    }


# ----------------------------------------------------------------------------
# 6. 进场评估 (核心) —— 回测与实盘共用
# ----------------------------------------------------------------------------

@dataclass
class Signal:
    side: str          # 'long' / 'short'
    entry: float       # 参考进场价(当根收盘价); entry_type="stop" 时为预估成交价
    sl: float
    tp: float
    rr: float
    reason: str
    poi_top: float = np.nan
    poi_bot: float = np.nan
    swept_level: float = np.nan
    # 进场方式: "market"=信号K线收盘市价进场(默认);
    #          "stop"  =在 trigger 处挂止损突破单, 仅随后 valid_bars 根有效, 未触发即撤。
    entry_type: str = "market"
    trigger: float = np.nan     # stop 单触发价(多头=信号K线高点+缓冲, 空头=低点-缓冲)
    valid_bars: int = 1         # stop 单有效K线数(超时撤单)


def _recent_bull_fvg(arr, t, p):
    """在 [t-zone_valid, t-1] 内寻找最近一个、尚未被填补的看涨 FVG。返回 (m, top, bot) 或 None。"""
    h, l = arr["high"], arr["low"]
    lo = max(1, t - p.zone_valid_bars)
    for m in range(t - 1, lo - 1, -1):
        # FVG 在 m+1 收盘后可见, 因此要求 m+1 <= t
        if arr["fvg_type"][m] == 1 and (m + 1) <= t:
            top, bot = arr["fvg_top"][m], arr["fvg_bot"][m]
            # 检查 m+2 .. t-1 期间是否已被跌破填补(收盘穿过下沿则失效)
            mitigated = False
            for k in range(m + 2, t):
                if arr["close"][k] < bot:
                    mitigated = True
                    break
            if not mitigated:
                return m, top, bot
    return None


def _recent_bear_fvg(arr, t, p):
    """在 [t-zone_valid, t-1] 内寻找最近一个、尚未被填补的看跌 FVG。"""
    h, l = arr["high"], arr["low"]
    lo = max(1, t - p.zone_valid_bars)
    for m in range(t - 1, lo - 1, -1):
        if arr["fvg_type"][m] == -1 and (m + 1) <= t:
            top, bot = arr["fvg_top"][m], arr["fvg_bot"][m]
            mitigated = False
            for k in range(m + 2, t):
                if arr["close"][k] > top:
                    mitigated = True
                    break
            if not mitigated:
                return m, top, bot
    return None


def _has_bull_fvg_near(arr, t, p):
    """t 前后 confirm_bars 内是否出现看涨 FVG(位移确认)。"""
    for m in range(max(1, t - p.confirm_bars - 1), t):
        if arr["fvg_type"][m] == 1 and (m + 1) <= t:
            a = arr["atr"][t]
            if (arr["fvg_top"][m] - arr["fvg_bot"][m]) >= p.fvg_min_atr * a:
                return True
    return False


def _has_bear_fvg_near(arr, t, p):
    for m in range(max(1, t - p.confirm_bars - 1), t):
        if arr["fvg_type"][m] == -1 and (m + 1) <= t:
            a = arr["atr"][t]
            if (arr["fvg_top"][m] - arr["fvg_bot"][m]) >= p.fvg_min_atr * a:
                return True
    return False


def evaluate_entry(arr: dict, t: int, p: StrategyParams) -> Optional[Signal]:
    """
    判断第 t 根(已收盘)K线收盘瞬间是否触发进场。
    返回 Signal 或 None。回测逐根调用, 实盘对最后一根调用。

    模型: ICT 流动性猎杀 + 收回 (Liquidity Sweep + Reclaim / Turtle Soup)
      做多(对称做空):
        1. sl_level[t] 是已确认的最近摆动低点(静置的卖方流动性);
        2. 该低点是最近 sweep_lookback 根内的最低 -> 是真正静置的流动性;
        3. 当根 K 线最低跌破该低点(扫损), 但收盘重新站回其上方(收回);
        4. 收盘落在 K 线区间上半部(拒绝插针, 反转确认);
        5. (可选) 反转动作伴随看涨 FVG(位移确认)。
      进场=当根收盘价; 止损=插针外侧+缓冲(很紧); 止盈=上方对侧流动性, 不足按 min_rr 延伸。
    """
    n = arr["n"]
    need = max(p.swing_left + p.swing_right + 2, p.atr_period + 2, p.sweep_lookback + 2)
    if p.entry_model == "ma_bias":
        need = max(need, p.ft_lookback + 2)  # 高周期均线预热由 ribbon 的 NaN 自动跳过
    if t < need or t >= n:
        return None
    if not arr["in_session"][t]:
        return None

    a = arr["atr"][t]
    if not np.isfinite(a) or a <= 0:
        return None

    o, h, l, c = arr["open"], arr["high"], arr["low"], arr["close"]
    close_t, high_t, low_t = c[t], h[t], l[t]
    bar_rng = high_t - low_t
    if bar_rng <= 0:
        return None

    ema_t = arr["ema"][t]
    trend_up = (not p.use_trend_filter) or (not np.isfinite(ema_t)) or (close_t >= ema_t)
    trend_dn = (not p.use_trend_filter) or (not np.isfinite(ema_t)) or (close_t <= ema_t)

    lo_w = max(0, t - p.sweep_lookback)
    o = arr["open"]

    # ============== 多均线"顺大逆小+首次触及"模型 (ma_bias) ==============
    if p.entry_model == "ma_bias":
        ribbon = arr["ribbon"]            # shape (n, k), 周期由慢到快? 否: 按 ma_ribbon 顺序
        periods = arr["ribbon_periods"]   # 例如 (5,15,60,240)
        k = len(periods)
        if k < 2:
            return None
        anchor = ribbon[t, -1]            # 最慢均线(240) = 大级别方向
        confirm = ribbon[t, -2]           # 次慢(60) = 大级别斜率确认
        if not (np.isfinite(anchor) and np.isfinite(confirm)):
            return None
        tol = p.ma_touch_buf * a
        ft = p.ft_lookback
        if t - ft < 0:
            return None
        # 触及用的"小级别"均线 = 除最慢锚以外的(5,15,60), 由深到浅(慢->快)遍历, 取最深的首触
        touch_order = list(range(k - 2, -1, -1))  # 索引: k-2 .. 0 (即 60,15,5)

        big_up = (close_t > anchor) and (confirm > anchor)
        big_dn = (close_t < anchor) and (confirm < anchor)

        if big_up:
            for j in touch_order:
                ma = ribbon[:, j]
                ma_t = ma[t]
                if not np.isfinite(ma_t):
                    continue
                touched = (low_t <= ma_t + tol) and (close_t >= ma_t)   # 回踩触及且收回上方
                if not touched:
                    continue
                if p.ma_require_resume and not (close_t > o[t] and close_t > h[t - 1]):
                    continue  # 动量续势确认: 阳线收盘突破上一根高点
                # 首次触及: 触及前 ft 根的最低价都在该均线之上(未触及过)
                first = np.all(l[t - ft:t] > ribbon[t - ft:t, j] + tol)
                if not first:
                    continue
                sl = low_t - p.sl_buffer_atr * a
                risk = close_t - sl
                if not (risk > 0 and p.min_stop_atr * a <= risk <= p.max_stop_atr * a):
                    continue
                target = arr["sh_level"][t]
                tp = target if (np.isfinite(target) and (target - close_t) >= p.min_rr * risk) \
                    else close_t + p.min_rr * risk
                rr = (tp - close_t) / risk
                if rr >= p.min_rr:
                    return Signal("long", close_t, sl, tp, rr,
                                  f"大多头·首触{periods[j][0]}min均线回踩做多", np.nan, np.nan, ma_t)
        if big_dn:
            for j in touch_order:
                ma = ribbon[:, j]
                ma_t = ma[t]
                if not np.isfinite(ma_t):
                    continue
                touched = (high_t >= ma_t - tol) and (close_t <= ma_t)
                if not touched:
                    continue
                if p.ma_require_resume and not (close_t < o[t] and close_t < l[t - 1]):
                    continue  # 动量续势确认: 阴线收盘跌破上一根低点
                first = np.all(h[t - ft:t] < ribbon[t - ft:t, j] - tol)
                if not first:
                    continue
                sl = high_t + p.sl_buffer_atr * a
                risk = sl - close_t
                if not (risk > 0 and p.min_stop_atr * a <= risk <= p.max_stop_atr * a):
                    continue
                target = arr["sl_level"][t]
                tp = target if (np.isfinite(target) and (close_t - target) >= p.min_rr * risk) \
                    else close_t - p.min_rr * risk
                rr = (close_t - tp) / risk
                if rr >= p.min_rr:
                    return Signal("short", close_t, sl, tp, rr,
                                  f"大空头·首触{periods[j][0]}min均线反抽做空", np.nan, np.nan, ma_t)
        return None

    # ================= 顺势回踩续势模型 (buy dip / sell rally) =================
    if p.entry_model == "pullback":
        ema_f = arr["ema_fast"][t]
        ema_s = arr["ema"][t]
        if not (np.isfinite(ema_f) and np.isfinite(ema_s)):
            return None
        win_lo = l[lo_w:t + 1]
        win_hi = h[lo_w:t + 1]
        pull_low = np.min(win_lo)
        pull_high = np.max(win_hi)
        # 多: 强势上行(快EMA在慢EMA上方且价在慢EMA上方), 回踩到快EMA附近后, 当根收盘突破前一根高点续涨
        uptrend = ema_f > ema_s and close_t > ema_s
        dipped = pull_low <= ema_f                     # 回踩触及动态支撑
        resume = close_t > o[t] and close_t > h[t - 1]  # 阳线收盘突破上根高点(续势)
        if uptrend and dipped and resume:
            sl = pull_low - p.sl_buffer_atr * a
            risk = close_t - sl
            if risk > 0 and (p.min_stop_atr * a <= risk <= p.max_stop_atr * a):
                target = arr["sh_level"][t]
                tp = target if (np.isfinite(target) and (target - close_t) >= p.min_rr * risk) \
                    else close_t + p.min_rr * risk
                rr = (tp - close_t) / risk
                if rr >= p.min_rr:
                    return Signal("long", close_t, sl, tp, rr, "上升趋势回踩续涨",
                                  np.nan, np.nan, pull_low)
        # 空: 强势下行, 反抽到快EMA附近后, 当根收盘跌破前一根低点续跌
        downtrend = ema_f < ema_s and close_t < ema_s
        rallied = pull_high >= ema_f
        resume_dn = close_t < o[t] and close_t < l[t - 1]
        if downtrend and rallied and resume_dn:
            sl = pull_high + p.sl_buffer_atr * a
            risk = sl - close_t
            if risk > 0 and (p.min_stop_atr * a <= risk <= p.max_stop_atr * a):
                target = arr["sl_level"][t]
                tp = target if (np.isfinite(target) and (close_t - target) >= p.min_rr * risk) \
                    else close_t - p.min_rr * risk
                rr = (close_t - tp) / risk
                if rr >= p.min_rr:
                    return Signal("short", close_t, sl, tp, rr, "下降趋势反抽续跌",
                                  np.nan, np.nan, pull_high)
        return None

    # ================= 突破 / 破结构顺势模型 (Donchian 收盘突破) =================
    if p.entry_model == "breakout":
        resistance = np.max(h[lo_w:t])   # 窗口内最高高点(上方流动性/阻力)
        support = np.min(l[lo_w:t])       # 窗口内最低低点(下方流动性/支撑)
        # 做多: 收盘突破阻力
        if trend_up:
            body_up = close_t > o[t] and (close_t - low_t) / bar_rng >= p.reject_ratio
            ok_fvg = (not p.require_fvg) or _has_bull_fvg_near(arr, t, p)
            if close_t > resistance and body_up and ok_fvg:
                sl = low_t - p.sl_buffer_atr * a
                risk = close_t - sl
                if risk > 0 and (p.min_stop_atr * a <= risk <= p.max_stop_atr * a):
                    tp = close_t + p.min_rr * risk
                    return Signal("long", close_t, sl, tp, p.min_rr,
                                  "破前高顺势做多", np.nan, np.nan, resistance)
        # 做空: 收盘跌破支撑
        if trend_dn:
            body_dn = close_t < o[t] and (high_t - close_t) / bar_rng >= p.reject_ratio
            ok_fvg = (not p.require_fvg) or _has_bear_fvg_near(arr, t, p)
            if close_t < support and body_dn and ok_fvg:
                sl = high_t + p.sl_buffer_atr * a
                risk = sl - close_t
                if risk > 0 and (p.min_stop_atr * a <= risk <= p.max_stop_atr * a):
                    tp = close_t - p.min_rr * risk
                    return Signal("short", close_t, sl, tp, p.min_rr,
                                  "破前低顺势做空", np.nan, np.nan, support)
        return None

    # ---------------- 做多: 扫掉前低后收回 ----------------
    if trend_up:
        level = arr["sl_level"][t]                       # 最近已确认摆动低点
        if np.isfinite(level):
            prior_min_low = np.min(l[lo_w:t])            # 当根之前窗口最低
            swept = low_t < level and close_t > level    # 跌破并收回
            fresh = prior_min_low >= level - 1e-9        # 该低点是窗口内静置流动性
            reclaim_q = (close_t - low_t) / bar_rng >= p.reject_ratio
            ok_fvg = (not p.require_fvg) or _has_bull_fvg_near(arr, t, p)
            if swept and fresh and reclaim_q and ok_fvg:
                sl = low_t - p.sl_buffer_atr * a
                risk = close_t - sl
                if risk > 0 and (p.min_stop_atr * a <= risk <= p.max_stop_atr * a):
                    target = arr["sh_level"][t]
                    tp_struct = target if (np.isfinite(target) and
                                           (target - close_t) >= p.min_rr * risk) else np.nan
                    tp = tp_struct if np.isfinite(tp_struct) else close_t + p.min_rr * risk
                    rr = (tp - close_t) / risk
                    if rr >= p.min_rr:
                        return Signal("long", close_t, sl, tp, rr,
                                      "扫前低后收回(反转做多)", np.nan, np.nan, level)

    # ---------------- 做空: 扫掉前高后收回 ----------------
    if trend_dn:
        level = arr["sh_level"][t]
        if np.isfinite(level):
            prior_max_high = np.max(h[lo_w:t])
            swept = high_t > level and close_t < level
            fresh = prior_max_high <= level + 1e-9
            reclaim_q = (high_t - close_t) / bar_rng >= p.reject_ratio
            ok_fvg = (not p.require_fvg) or _has_bear_fvg_near(arr, t, p)
            if swept and fresh and reclaim_q and ok_fvg:
                sl = high_t + p.sl_buffer_atr * a
                risk = sl - close_t
                if risk > 0 and (p.min_stop_atr * a <= risk <= p.max_stop_atr * a):
                    target = arr["sl_level"][t]
                    tp_struct = target if (np.isfinite(target) and
                                           (close_t - target) >= p.min_rr * risk) else np.nan
                    tp = tp_struct if np.isfinite(tp_struct) else close_t - p.min_rr * risk
                    rr = (close_t - tp) / risk
                    if rr >= p.min_rr:
                        return Signal("short", close_t, sl, tp, rr,
                                      "扫前高后收回(反转做空)", np.nan, np.nan, level)

    return None


# ----------------------------------------------------------------------------
# 7. 实盘便捷封装
# ----------------------------------------------------------------------------

def generate_signal(df: pd.DataFrame, p: StrategyParams) -> Optional[Signal]:
    """
    实盘用: 传入"最近 N 根已收盘 K 线"的 DataFrame(最后一行为刚收盘的 K 线),
    返回最后一根 K 线收盘时的进场信号或 None。
    """
    if len(df) < 60:
        return None
    arr = precompute(df, p)
    return evaluate_entry(arr, arr["n"] - 1, p)


# 各周期默认参数 (经 analyze_history.py + tune.py 在 2012-2024 全样本调校)
# 经验结论(spread=$0.08 raw 账户): pullback 顺势回踩续势模型在黄金上具备正期望,
#   - M15 最稳健 (PF≈1.08, 12年正收益 9/13);
#   - M5  边际正期望 (PF≈1.03), 对点差敏感, 建议 raw/ECN 低点差账户;
#   - M1  接近盈亏平衡, 成本敏感, 列为实验性, 建议优先用 M15。
# 注: pullback 模型不使用 reject_ratio/require_fvg/confirm_bars(那些用于 reversal/breakout)。
SESS = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
DEFAULT_PARAMS = {
    "M1": StrategyParams(
        name="M1", entry_model="pullback",
        swing_left=3, swing_right=3, atr_period=14,
        ema_fast=50, ema_trend=200, sweep_lookback=8,
        min_rr=4.0, sl_buffer_atr=0.10, min_stop_atr=0.25, max_stop_atr=3.5,
        allow_scale_in=True, max_positions=3, sessions_utc=SESS,
    ),
    "M5": StrategyParams(
        name="M5", entry_model="pullback",
        swing_left=3, swing_right=3, atr_period=14,
        ema_fast=50, ema_trend=200, sweep_lookback=8,
        min_rr=4.0, sl_buffer_atr=0.10, min_stop_atr=0.25, max_stop_atr=3.5,
        allow_scale_in=True, max_positions=3, sessions_utc=SESS,
    ),
    "M15": StrategyParams(
        name="M15", entry_model="pullback",
        swing_left=4, swing_right=4, atr_period=14,
        ema_fast=20, ema_trend=50, sweep_lookback=15,
        min_rr=3.0, sl_buffer_atr=0.10, min_stop_atr=0.25, max_stop_atr=3.5,
        allow_scale_in=True, max_positions=3, sessions_utc=SESS,
    ),
    # 真·多周期均线方向偏向 (M15 执行, 用 M15/H1/H4 各自 EMA100 锚定大级别趋势,
    # 只做大级别顺势中"第一次回踩高周期均线"的顺大逆小进场)。
    # 回测(2012-2024, 点差0.10): 单笔期望 +0.162R, PF 1.19, 最大回撤 65R, 9/13 年为正,
    # 显著优于 pullback 基准(+0.050R / PF1.07 / 132R)。频率较低(~82 笔/年), 对点差敏感。
    "M15_MABIAS": StrategyParams(
        name="M15_MABIAS", entry_model="ma_bias",
        swing_left=4, swing_right=4, atr_period=14,
        mtf_mas=((5, 100), (15, 100), (60, 100), (240, 100)),
        ft_lookback=8, ma_touch_buf=0.05, ma_require_resume=False,
        min_rr=3.5, sl_buffer_atr=0.10, min_stop_atr=0.25, max_stop_atr=3.5,
        allow_scale_in=True, max_positions=3, sessions_utc=SESS,
    ),
}
