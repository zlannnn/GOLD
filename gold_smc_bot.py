"""
gold_smc_bot.py
================
黄金 (XAUUSD) SMC/ICT/价格行为 实盘交易机器人 (对接 MetaTrader5)。
仿照 GOLDBOTV0.py 的基底, 但策略逻辑与回测完全一致 (共用 smc_core.evaluate_entry),
并严格遵守用户三原则:

  原则1: 开仓即带止盈止损, 且盈亏比 RR >= min_rr (>=1)。  -> 信号层保证, 下单时一并提交 sl/tp。
  原则2: 允许加仓, 但新加仓止损不得比初始仓位更差。      -> 加仓时把 sl 钳制到不差于初始仓位。
  原则3: 仅在当前周期 K 线收盘的瞬间判断是否交易。        -> 主循环只在"检测到新收盘K线"时评估一次。

可通过环境变量 GOLD_TF 选择运行周期: M1 / M5 / M15。
连接信息通过环境变量提供(见 config.py), 不在代码中硬编码真实账号。

运行:
  set MT5_LOGIN=12345678 & set MT5_PASSWORD=xxx & set MT5_SERVER=Broker-Demo
  set GOLD_TF=M15
  python gold_smc_bot.py
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    raise SystemExit("未安装 MetaTrader5: 请先 `pip install MetaTrader5` (仅 Windows 实盘需要)")

import config as cfg
import smc_core as sc
import three_candle as tc

TF_MAP = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15}
TF_SECONDS = {"M1": 60, "M5": 300, "M15": 900}

# 加仓触发: 浮盈达到该 R 倍数才考虑金字塔加仓(与回测一致)
SCALE_TRIGGER_R = 0.5
BARS_FOR_SIGNAL = 400   # 普通模型每次评估拉取的已收盘K线数量


def select_strategy(preset: str):
    """按预设返回 (策略模块, 参数对象)。3K线短线用 three_candle, 其余用 smc_core。"""
    if preset in tc.DEFAULT_3C:
        return tc, tc.DEFAULT_3C[preset]
    return sc, cfg.PARAMS[preset]


def bars_needed(tf_key: str, p) -> int:
    """估算需要拉取的已收盘K线数量, 保证高周期(均线/关键位)有足够预热历史。"""
    base_min = TF_SECONDS[tf_key] // 60
    # 3K线短线: 需要 H4 趋势EMA 收敛 + 足够 H4 摆动点历史
    if hasattr(p, "htf_levels"):
        warm_ema = (p.trend_tf / base_min) * p.trend_ema * 1.5 if p.trend_filter else 0
        max_htf = max((tf for tf, _, _ in p.htf_levels), default=base_min)
        hist_piv = (max_htf / base_min) * 60          # 约 60 根高周期摆动历史
        return int(max(BARS_FOR_SIGNAL, max(warm_ema, hist_piv) + p.fresh_lookback + 300))
    # ma_bias: 真·多周期均线预热
    if getattr(p, "entry_model", "") == "ma_bias" and p.mtf_mas:
        need = max(int((tf_min / base_min) * period) for tf_min, period in p.mtf_mas)
        return max(BARS_FOR_SIGNAL, need * 4 + 100)
    return BARS_FOR_SIGNAL


# ---------------------------------------------------------------- 基础设施

def init_mt5():
    if not mt5.initialize():
        raise SystemExit(f"MT5 初始化失败: {mt5.last_error()}")
    if cfg.MT5_LOGIN:
        if not mt5.login(cfg.MT5_LOGIN, password=cfg.MT5_PASSWORD, server=cfg.MT5_SERVER):
            raise SystemExit(f"MT5 登录失败: {mt5.last_error()}")
    if not mt5.symbol_select(cfg.SYMBOL, True):
        raise SystemExit(f"无法选择品种 {cfg.SYMBOL}")
    info = mt5.account_info()
    print(f"已连接: 账户 {getattr(info,'login','?')}  余额 {getattr(info,'balance','?')}  "
          f"服务器 {getattr(info,'server','?')}")
    print(f"运行周期: {cfg.ACTIVE_TIMEFRAME}   品种: {cfg.SYMBOL}   magic: {cfg.MAGIC}")


def get_closed_df(tf_key: str, count: int) -> pd.DataFrame | None:
    """
    拉取最近 count 根 *已收盘* K 线 (从 pos=1 开始, pos=0 是正在形成的当前K线)。
    时间换算为 UTC 以匹配交易时段过滤。
    """
    rates = mt5.copy_rates_from_pos(cfg.SYMBOL, TF_MAP[tf_key], 1, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    # 服务器时间(秒) -> UTC
    t = pd.to_datetime(df["time"].to_numpy(), unit="s") - pd.Timedelta(hours=cfg.BROKER_TZ_SHIFT)
    out = pd.DataFrame({
        "open": df["open"].to_numpy(dtype=float),
        "high": df["high"].to_numpy(dtype=float),
        "low": df["low"].to_numpy(dtype=float),
        "close": df["close"].to_numpy(dtype=float),
        "volume": df["tick_volume"].to_numpy(dtype=float),
    }, index=pd.DatetimeIndex(t))
    return out


def my_positions(side: str | None = None):
    """返回本 EA(按 magic) 在该品种上的持仓, 可按方向过滤。"""
    poss = mt5.positions_get(symbol=cfg.SYMBOL) or []
    res = []
    for p in poss:
        if p.magic != cfg.MAGIC:
            continue
        s = "long" if p.type == mt5.ORDER_TYPE_BUY else "short"
        if side is None or s == side:
            res.append(p)
    return res


def my_pending_orders(side: str | None = None):
    """返回本 EA(按 magic) 挂着的未触发挂单(buy_stop/sell_stop)。"""
    orders = mt5.orders_get(symbol=cfg.SYMBOL) or []
    buy_types = (mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_BUY_LIMIT)
    res = []
    for o in orders:
        if o.magic != cfg.MAGIC:
            continue
        s = "long" if o.type in buy_types else "short"
        if side is None or s == side:
            res.append(o)
    return res


def cancel_pending(side: str | None = None):
    """撤销本 EA 的挂单(实现 PA '未触发即撤')。"""
    for o in my_pending_orders(side):
        mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})


def send_stop_order(side: str, lot: float, trigger: float, sl: float, tp: float,
                    expire_ts: int, comment: str):
    """挂 stop 突破单(buy_stop/sell_stop), 到期未触发自动撤销(ORDER_TIME_SPECIFIED)。"""
    digits = mt5.symbol_info(cfg.SYMBOL).digits
    sl, tp = enforce_min_stops(side, trigger, sl, tp)
    req = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": cfg.SYMBOL,
        "volume": normalize_volume(lot),
        "type": mt5.ORDER_TYPE_BUY_STOP if side == "long" else mt5.ORDER_TYPE_SELL_STOP,
        "price": round(trigger, digits),
        "sl": round(sl, digits),
        "tp": round(tp, digits),
        "deviation": cfg.DEVIATION,
        "magic": cfg.MAGIC,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_SPECIFIED,
        "expiration": int(expire_ts),
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res is None:
        print("  挂单返回 None:", mt5.last_error())
    elif res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"  挂单失败 retcode={res.retcode}: {res.comment}")
    else:
        print(f"  ✓ 挂 {side} stop @ {trigger:.2f}  SL {req['sl']:.2f}  TP {req['tp']:.2f} "
              f"(到期 {datetime.fromtimestamp(expire_ts, tz=timezone.utc):%H:%M}Z)")
    return res


def normalize_volume(vol: float) -> float:
    info = mt5.symbol_info(cfg.SYMBOL)
    step = info.volume_step or 0.01
    vol = max(info.volume_min, round(round(vol / step) * step, 2))
    return min(vol, info.volume_max)


def enforce_min_stops(side: str, price: float, sl: float, tp: float):
    """确保 sl/tp 距离 >= 经纪商最小止损位(stops_level), 否则外推。"""
    info = mt5.symbol_info(cfg.SYMBOL)
    point = info.point
    min_dist = (info.trade_stops_level or 0) * point
    if min_dist <= 0:
        return sl, tp
    if side == "long":
        sl = min(sl, price - min_dist)
        tp = max(tp, price + min_dist)
    else:
        sl = max(sl, price + min_dist)
        tp = min(tp, price - min_dist)
    return sl, tp


def send_market(side: str, lot: float, sl: float, tp: float, comment: str):
    tick = mt5.symbol_info_tick(cfg.SYMBOL)
    price = tick.ask if side == "long" else tick.bid
    sl, tp = enforce_min_stops(side, price, sl, tp)
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": cfg.SYMBOL,
        "volume": normalize_volume(lot),
        "type": mt5.ORDER_TYPE_BUY if side == "long" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": round(sl, mt5.symbol_info(cfg.SYMBOL).digits),
        "tp": round(tp, mt5.symbol_info(cfg.SYMBOL).digits),
        "deviation": cfg.DEVIATION,
        "magic": cfg.MAGIC,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res is None:
        print("  order_send 返回 None:", mt5.last_error())
    elif res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"  下单失败 retcode={res.retcode}: {res.comment}")
    else:
        print(f"  ✓ {side} {req['volume']} 手 @ {price:.2f}  SL {req['sl']:.2f}  TP {req['tp']:.2f}")
    return res


# ---------------------------------------------------------------- 交易决策

def on_bar_close(p, strat=sc):
    """每根 K 线收盘时调用一次, 判断并执行进场/加仓。strat: 信号来源模块(smc_core 或 three_candle)。"""
    df = get_closed_df(cfg.ACTIVE_TIMEFRAME, bars_needed(cfg.ACTIVE_TIMEFRAME, p))
    if df is None or len(df) < 80:
        print("  K线数据不足, 跳过")
        return

    sig = strat.generate_signal(df, p)
    if sig is None:
        return

    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z] 信号: {sig.side} "
          f"entry≈{sig.entry:.2f} SL {sig.sl:.2f} TP {sig.tp:.2f} RR {sig.rr:.2f} ({sig.reason})")

    opp = "short" if sig.side == "long" else "long"
    if my_positions(opp):
        print("  已有反向持仓, 不对冲, 跳过")
        return

    same = my_positions(sig.side)

    # ---- PA stop 突破单: 在第3根极值外挂 stop, 仅随后 valid_bars 根有效, 未触发即撤 ----
    if getattr(sig, "entry_type", "market") == "stop":
        # 新K线收盘 -> 上一根的旧挂单已失效, 先撤销, 再(若无持仓)按本信号重新挂单
        cancel_pending()
        if same:
            return   # 已有同向持仓(短线 max_positions=1), 不再挂单
        sec = TF_SECONDS[cfg.ACTIVE_TIMEFRAME] * max(1, int(getattr(sig, "valid_bars", 1)))
        expire_ts = int(datetime.now(timezone.utc).timestamp()) + sec
        send_stop_order(sig.side, cfg.BASE_LOT, sig.trigger, sig.sl, sig.tp,
                        expire_ts, f"3C-{p.name}-stop")
        return

    if not same:
        # 首仓: 直接用信号的 sl/tp(已保证 RR>=min_rr)
        send_market(sig.side, cfg.BASE_LOT, sig.sl, sig.tp, f"SMC-{p.name}-entry")
        return

    # ---- 加仓(金字塔) ----
    if not p.allow_scale_in or len(same) >= p.max_positions:
        return
    tick = mt5.symbol_info_tick(cfg.SYMBOL)
    cur = tick.ask if sig.side == "long" else tick.bid
    first = min(same, key=lambda x: x.time)          # 最早的仓位 = 初始仓
    init_risk = abs(first.price_open - first.sl) if first.sl else abs(sig.entry - sig.sl)
    favor = (cur - first.price_open) if sig.side == "long" else (first.price_open - cur)
    if init_risk <= 0 or favor < SCALE_TRIGGER_R * init_risk:
        return  # 浮盈不足, 不加仓

    # 原则2: 钳制新仓止损, 不得差于初始仓位止损
    worst_sl = (min(x.sl for x in same if x.sl) if any(x.sl for x in same) else sig.sl)
    if sig.side == "long":
        new_sl = max(sig.sl, worst_sl)
    else:
        new_sl = min(sig.sl, worst_sl) if worst_sl else sig.sl
    # 原则1: 钳制后仍需满足 RR>=min_rr, 否则放弃加仓
    risk = abs(cur - new_sl)
    rr = (sig.tp - cur) / risk if sig.side == "long" else (cur - sig.tp) / risk
    if risk <= 0 or rr < p.min_rr:
        print(f"  加仓被否决(钳制止损后 RR={rr:.2f} < {p.min_rr})")
        return
    print(f"  加仓 #{len(same)+1} (浮盈 {favor:.2f} ≥ {SCALE_TRIGGER_R}R)")
    send_market(sig.side, cfg.SCALE_LOT, new_sl, sig.tp, f"SMC-{p.name}-scale")


# ---------------------------------------------------------------- 主循环

def main():
    init_mt5()
    tf_key = cfg.ACTIVE_TIMEFRAME
    strat, p = select_strategy(cfg.ACTIVE_PRESET)
    model = getattr(p, "entry_model", "three_candle")
    print(f"策略: 预设={cfg.ACTIVE_PRESET}  模型={model}  周期={tf_key}  "
          f"min_RR={p.min_rr}  时段UTC={p.sessions_utc}")
    print("等待 K 线收盘 ... (仅在收盘瞬间评估)\n")

    last_closed_time = None
    try:
        while True:
            rates = mt5.copy_rates_from_pos(cfg.SYMBOL, TF_MAP[tf_key], 1, 1)
            if rates is None or len(rates) == 0:
                time.sleep(1)
                continue
            closed_time = int(rates[-1]["time"])
            if last_closed_time is None:
                last_closed_time = closed_time          # 启动时不立即交易, 等下一根收盘
            elif closed_time != last_closed_time:
                last_closed_time = closed_time
                bar_dt = datetime.fromtimestamp(closed_time, tz=timezone.utc)
                print(f"--- 新K线收盘 (服务器时间戳 {bar_dt:%Y-%m-%d %H:%M}) ---")
                try:
                    on_bar_close(p, strat)
                except Exception as e:
                    print("  决策异常:", e)
            # 收盘附近高频检查, 其余时间低频
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n手动停止。")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
