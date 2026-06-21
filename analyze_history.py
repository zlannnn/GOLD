"""
analyze_history.py
================
对黄金历史数据 (gold2012_2024.csv, 1分钟) 做统计分析, 为 1/5/15min 三个周期的
SMC/ICT 交易逻辑提供"数据依据"。

输出 (E:\\fun\\GOLD\\reports\\):
  - analysis_report.md            文字统计报告
  - atr_by_hour_<TF>.png          各 UTC 小时平均波幅(决定最佳交易时段)
  - sweep_reversal_<TF>.png       流动性扫损后反转的统计(策略核心假设的验证)

运行:
  python analyze_history.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_utils as du
import smc_core as sc

REPORT_DIR = r"E:\fun\GOLD\reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# 分析时为控制运行时间, 对超大样本的"逐摆动点"分析取最近 MAX_BARS 根
MAX_BARS_PIVOT = 900_000


def basic_stats(df: pd.DataFrame) -> dict:
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    rng = h - l
    body = np.abs(c - o)
    a = sc.atr(h, l, c, 14)
    return {
        "bars": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "avg_range": float(np.nanmean(rng)),
        "median_range": float(np.nanmedian(rng)),
        "avg_body": float(np.nanmean(body)),
        "avg_atr14": float(np.nanmean(a)),
        "p90_range": float(np.nanpercentile(rng, 90)),
    }


def hourly_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """按 UTC 小时统计平均波幅与平均绝对收益。"""
    rng = (df["high"] - df["low"]).to_numpy()
    body = (df["close"] - df["open"]).abs().to_numpy()
    g = pd.DataFrame({"hour": df.index.hour, "range": rng, "body": body})
    return g.groupby("hour").agg(avg_range=("range", "mean"),
                                 avg_body=("body", "mean"),
                                 count=("range", "size"))


def fvg_fill_rate(df: pd.DataFrame, valid_bars: int) -> dict:
    """统计 FVG 在形成后 valid_bars 根内被回补(价格回到缺口)的比例。"""
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    ftype, ftop, fbot = sc.find_fvg(o, h, l, c)
    n = len(c)
    idxs = np.where(ftype != 0)[0]
    bull = bear = bull_filled = bear_filled = 0
    for m in idxs:
        end = min(n, m + 1 + valid_bars)
        if end <= m + 2:
            continue
        if ftype[m] == 1:
            bull += 1
            # 回补: 后续最低价触及缺口上沿
            if np.min(l[m + 2:end]) <= ftop[m]:
                bull_filled += 1
        else:
            bear += 1
            # 回补: 后续最高价触及缺口下沿
            if np.max(h[m + 2:end]) >= fbot[m]:
                bear_filled += 1
    return {
        "bull_fvg": bull, "bull_fill_rate": (bull_filled / bull * 100) if bull else 0.0,
        "bear_fvg": bear, "bear_fill_rate": (bear_filled / bear * 100) if bear else 0.0,
        "fvg_per_1000bars": (len(idxs) / n * 1000) if n else 0.0,
    }


def swing_leg_stats(df: pd.DataFrame, left: int, right: int) -> dict:
    """摆动腿(相邻摆动点间价格差)的分布, 用于设定止损 ATR 倍数范围。"""
    h = df["high"].to_numpy(); l = df["low"].to_numpy()
    ph, pl = sc.find_pivots(h, l, left, right)
    pts = sorted([(i, h[i], 1) for i in ph] + [(i, l[i], -1) for i in pl])
    legs = []
    for k in range(1, len(pts)):
        legs.append(abs(pts[k][1] - pts[k - 1][1]))
    legs = np.array(legs) if legs else np.array([np.nan])
    a = sc.atr(h, l, df["close"].to_numpy(), 14)
    return {
        "num_pivots": len(pts),
        "median_leg": float(np.nanmedian(legs)),
        "avg_leg": float(np.nanmean(legs)),
        "median_leg_in_atr": float(np.nanmedian(legs) / np.nanmean(a)) if np.nanmean(a) else np.nan,
    }


def sweep_reversal_edge(df: pd.DataFrame, left: int, right: int,
                        horizon: int = 20) -> dict:
    """
    验证策略核心假设: 摆动低点被扫破后, 价格在 horizon 根内向上反转的概率与幅度
    (做空对称)。这是 SMC 流动性猎杀逻辑是否在黄金上成立的关键证据。
    """
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    a = sc.atr(h, l, c, 14)
    ph, pl = sc.find_pivots(h, l, left, right)
    n = len(c)

    # 卖方流动性(低点)被扫 -> 看是否上反转
    up_rev = 0; low_sweeps = 0; up_mfe = []
    for piv in pl:
        level = l[piv]
        conf = piv + right            # 该低点确认位置
        # 在确认后向前找第一根扫破(低于 level)的 K 线
        j = None
        for k in range(conf + 1, min(n, conf + 1 + horizon)):
            if l[k] < level:
                j = k
                break
        if j is None:
            continue
        low_sweeps += 1
        atr_j = a[j] if np.isfinite(a[j]) and a[j] > 0 else np.nanmean(a)
        sweep_low = l[j]
        end = min(n, j + 1 + horizon)
        mfe_up = (np.max(h[j:end]) - sweep_low) / atr_j   # 反转上行最大幅度(ATR)
        up_mfe.append(mfe_up)
        # 定义"反转成功": 扫破后 horizon 根内向上走 >= 1 个 ATR
        if mfe_up >= 1.0:
            up_rev += 1

    # 买方流动性(高点)被扫 -> 看是否下反转
    dn_rev = 0; high_sweeps = 0; dn_mfe = []
    for piv in ph:
        level = h[piv]
        conf = piv + right
        j = None
        for k in range(conf + 1, min(n, conf + 1 + horizon)):
            if h[k] > level:
                j = k
                break
        if j is None:
            continue
        high_sweeps += 1
        atr_j = a[j] if np.isfinite(a[j]) and a[j] > 0 else np.nanmean(a)
        sweep_high = h[j]
        end = min(n, j + 1 + horizon)
        mfe_dn = (sweep_high - np.min(l[j:end])) / atr_j
        dn_mfe.append(mfe_dn)
        if mfe_dn >= 1.0:
            dn_rev += 1

    return {
        "low_sweeps": low_sweeps,
        "up_reversal_rate": (up_rev / low_sweeps * 100) if low_sweeps else 0.0,
        "up_median_mfe_atr": float(np.nanmedian(up_mfe)) if up_mfe else np.nan,
        "high_sweeps": high_sweeps,
        "dn_reversal_rate": (dn_rev / high_sweeps * 100) if high_sweeps else 0.0,
        "dn_median_mfe_atr": float(np.nanmedian(dn_mfe)) if dn_mfe else np.nan,
        "up_mfe": up_mfe, "dn_mfe": dn_mfe,
    }


def plot_hourly(hv: pd.DataFrame, tf: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hv.index, hv["avg_range"], color="#c9a227")
    ax.set_title(f"{tf} 平均K线波幅(美元) - 按UTC小时")
    ax.set_xlabel("UTC 小时"); ax.set_ylabel("平均 high-low ($)")
    ax.set_xticks(range(0, 24))
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, f"atr_by_hour_{tf}.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def plot_sweep(edge: dict, tf: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if edge["up_mfe"]:
        axes[0].hist(np.clip(edge["up_mfe"], 0, 6), bins=30, color="#2e7d32")
    axes[0].set_title(f"{tf} 低点被扫后向上MFE (ATR)")
    axes[0].axvline(1.0, color="red", ls="--")
    if edge["dn_mfe"]:
        axes[1].hist(np.clip(edge["dn_mfe"], 0, 6), bins=30, color="#c62828")
    axes[1].set_title(f"{tf} 高点被扫后向下MFE (ATR)")
    axes[1].axvline(1.0, color="red", ls="--")
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, f"sweep_reversal_{tf}.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    print("加载 1 分钟历史数据 ... (约 420 万行, 请稍候)")
    m1 = du.load_m1()
    print(f"  完成: {m1.index.min()} -> {m1.index.max()}, 共 {len(m1):,} 根\n")

    lines = ["# 黄金历史数据分析报告 (SMC/ICT 策略依据)\n",
             f"- 数据: gold2012_2024.csv  1分钟  {m1.index.min()} ~ {m1.index.max()}",
             f"- 总根数(1min): {len(m1):,}\n"]

    tfs = [("M1", m1), ("M5", du.resample(m1, "5min")), ("M15", du.resample(m1, "15min"))]

    for tf, df in tfs:
        p = sc.DEFAULT_PARAMS[tf]
        print(f"==== 分析 {tf} ({len(df):,} 根) ====")
        bs = basic_stats(df)
        hv = hourly_volatility(df)
        fr = fvg_fill_rate(df, p.zone_valid_bars)

        df_piv = df if len(df) <= MAX_BARS_PIVOT else df.iloc[-MAX_BARS_PIVOT:]
        note = "" if len(df) <= MAX_BARS_PIVOT else f"(摆动/扫损统计基于最近 {MAX_BARS_PIVOT:,} 根)"
        legs = swing_leg_stats(df_piv, p.swing_left, p.swing_right)
        edge = sweep_reversal_edge(df_piv, p.swing_left, p.swing_right, horizon=p.zone_valid_bars)

        p_hour = plot_hourly(hv, tf)
        p_sweep = plot_sweep(edge, tf)

        best_hours = hv.sort_values("avg_range", ascending=False).head(6).index.tolist()

        lines += [
            f"\n## {tf} 周期 {note}",
            f"- K线数: {bs['bars']:,}  范围: {bs['start']} ~ {bs['end']}",
            f"- 平均波幅 high-low: **${bs['avg_range']:.2f}**  中位数 ${bs['median_range']:.2f}  "
            f"90分位 ${bs['p90_range']:.2f}",
            f"- 平均实体: ${bs['avg_body']:.2f}   平均 ATR(14): **${bs['avg_atr14']:.2f}**",
            f"- 波动最大的 6 个 UTC 小时: {best_hours}  (-> 交易时段建议)",
            f"- FVG: 看涨 {fr['bull_fvg']:,} 个(回补率 {fr['bull_fill_rate']:.1f}%), "
            f"看跌 {fr['bear_fvg']:,} 个(回补率 {fr['bear_fill_rate']:.1f}%), "
            f"密度 {fr['fvg_per_1000bars']:.1f}/千根",
            f"- 摆动腿中位数: ${legs['median_leg']:.2f} (≈ {legs['median_leg_in_atr']:.2f} ATR), "
            f"摆动点数 {legs['num_pivots']:,}",
            f"- **流动性扫损反转验证**:",
            f"    - 低点被扫 {edge['low_sweeps']:,} 次, 之后 {p.zone_valid_bars} 根内上行≥1ATR 概率 "
            f"**{edge['up_reversal_rate']:.1f}%**, 上行MFE中位 {edge['up_median_mfe_atr']:.2f} ATR",
            f"    - 高点被扫 {edge['high_sweeps']:,} 次, 之后下行≥1ATR 概率 "
            f"**{edge['dn_reversal_rate']:.1f}%**, 下行MFE中位 {edge['dn_median_mfe_atr']:.2f} ATR",
            f"- 图: {os.path.basename(p_hour)} , {os.path.basename(p_sweep)}",
        ]
        print("  完成。\n")

    report_path = os.path.join(REPORT_DIR, "analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"报告已写入: {report_path}")


if __name__ == "__main__":
    main()
