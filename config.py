"""
config.py
================
集中管理: MT5 连接信息、交易品种、各周期策略参数、资金/风控。

安全提示
--------
不要把真实账户的账号密码硬编码进仓库。优先用环境变量:
    set MT5_LOGIN=12345678
    set MT5_PASSWORD=xxxxxx
    set MT5_SERVER=Broker-Demo
若未设置环境变量, 则使用下方 DEMO 占位值(需自行填写, 默认指向模拟盘)。
"""

import os

from smc_core import StrategyParams, DEFAULT_PARAMS

# ---------------- MT5 连接 ----------------
MT5_LOGIN = int(os.environ.get("MT5_LOGIN", "0") or 0)
MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "")
MT5_SERVER = os.environ.get("MT5_SERVER", "")   # 例: "Tickmill-Demo" / "ICMarketsSC-Demo"

SYMBOL = os.environ.get("MT5_SYMBOL", "XAUUSD")

# 经纪商服务器时间相对 UTC 的偏移(小时)。多数 MT5 服务器是 GMT+2/+3。
# 该值用于把 K 线时间换算成 UTC 以做交易时段过滤。常见: 夏令时 +3, 冬令时 +2。
BROKER_TZ_SHIFT = int(os.environ.get("MT5_TZ_SHIFT", "3"))

# ---------------- 资金 / 风控 ----------------
BASE_LOT = float(os.environ.get("GOLD_BASE_LOT", "0.01"))   # 首仓手数
SCALE_LOT = float(os.environ.get("GOLD_SCALE_LOT", "0.01")) # 每次加仓手数
MAGIC = 246001                                              # EA 魔术号
DEVIATION = 30                                              # 允许滑点(点)

# 选择实盘运行的"预设": "M1" / "M5" / "M15" / "M15_MABIAS"(真·多周期均线方向偏向)
ACTIVE_PRESET = os.environ.get("GOLD_TF", "M15")

# 各周期参数(来自 smc_core.DEFAULT_PARAMS, 可在此覆写)
PARAMS: dict[str, StrategyParams] = DEFAULT_PARAMS

# 预设 -> MT5 实际拉取的基础周期(预设名可能与周期不同, 如 M15_MABIAS 仍在 M15 上执行)
# 短线 3K线策略: M5_3C 在 M5 上执行, M1_3C 在 M1 上执行(关键位均来自 H4)。
PRESET_BASE_TF = {
    "M1": "M1", "M5": "M5", "M15": "M15", "M15_MABIAS": "M15",
    "M5_3C": "M5", "M1_3C": "M1",
}
ACTIVE_TIMEFRAME = PRESET_BASE_TF.get(ACTIVE_PRESET, ACTIVE_PRESET)
