#!/usr/bin/env python3
"""alpha2026 · Step 5: 策略构建复杂度实验

在**完全相同**的 LightGBM 每周再训练信号上(与 03 的 weekly 变体同源), 对比 6 种组合构建策略:
  strat_base       Top-K 等权(现状基准, 应≈03 weekly)
  strat_dropout    TopK-Dropout: 持仓跌出前 2K 才卖出, 空位从头部补 —— 降换手省成本(Qlib 官方做法)
  strat_riskparity 风险平价: Top-K 内按 60 日波动率倒数加权(单票上限 35%)
  strat_voltarget  波动率目标: base 权重 + 按 base 策略已实现波动率动态调整总仓位(目标年化 20%)
  strat_timing     趋势择时: 市场等权指数收于 30 周均线上方才满仓, 否则空仓持币
  strat_combo      全叠: Dropout 选股 + 风险平价加权 + min(波动率目标, 趋势) 仓位

检验问题: 组合构建层面的复杂度, 相比"Top-K 等权一把梭"到底值多少?

用法: python 05_strategy_enrichment.py --market gold
"""
import argparse
import importlib.util
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "configs" / "markets.yaml").read_text())

# 复用 03 的训练/预测/指标工具
_spec = importlib.util.spec_from_file_location("exp03", ROOT / "scripts" / "03_run_experiments.py")
exp03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp03)

TARGET_VOL = 0.20
W_CAP = 0.35
VOL_WINDOW = 60
SMA_WEEKS = 30
DROP_BUF = 2  # 持仓跌出 top (DROP_BUF*K) 才卖


def log(m):
    print(f"[{pd.Timestamp.now():%H:%M:%S}] {m}", flush=True)


# ------------------------------------------------------------------ 信号获取
def get_scores(market):
    """优先读取 03 落盘的周度信号(新鲜); 否则现场重训一遍 weekly LightGBM"""
    fp = ROOT / "data" / "scores" / f"{market}_lgbm_weekly.pkl"
    if fp.exists():
        obj = pd.read_pickle(fp)
        saved = pd.Timestamp(obj["saved_at"])
        if pd.Timestamp.now() - saved < pd.Timedelta(days=14):
            log(f"复用 03 落盘信号({saved:%m-%d %H:%M})")
            return obj["scores"]
    log("信号缺失/过期, 现场重训 weekly LightGBM ...")
    D = exp03.init_qlib(market)
    cal = [pd.Timestamp(d) for d in D.calendar(freq="day")]
    bt_start = pd.Timestamp(CFG["backtest_start"])
    cal_bt = [d for d in cal if d >= bt_start]
    from qlib.data.dataset.handler import DataHandlerLP

    ds = exp03.build_dataset(market, D)
    ds.segments = {"probe": (CFG["processor_fit_start"], CFG["processor_fit_end"])}
    probe = ds.prepare("probe", data_key=DataHandlerLP.DK_L)
    try:
        n_label = ds.prepare("probe", col_set="label", data_key=DataHandlerLP.DK_L).shape[1]
    except Exception:  # noqa: BLE001
        n_label = 1
    exp03.N_FEATURES[0] = max(len(probe.columns) - n_label, 1)

    rebals = exp03.rebalance_dates(cal)
    scores = {}
    for ri, T in enumerate(rebals):
        pos = cal.index(T)
        labels_end = cal[pos - 2]
        model, _ = exp03.fit_at(ds, cal, labels_end, "lgbm", cal_bt_hint=cal_bt[0])
        win = [d for d in cal_bt if d >= T]
        if ri + 1 < len(rebals):
            win = [d for d in win if d < rebals[ri + 1]]
        pred = exp03.predict_window(model, ds, win)
        for dt_, grp in pred.groupby("datetime"):
            scores[pd.Timestamp(dt_)] = grp.set_index("instrument")["score"]
        log(f"  refit@{T:%Y-%m-%d} ({ri + 1}/{len(rebals)})")
    sdir = ROOT / "data" / "scores"
    sdir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"scores": scores, "saved_at": pd.Timestamp.now()}, sdir / f"{market}_lgbm_weekly.pkl")
    return scores


# ------------------------------------------------------------------ 模拟器
def simulate(cal_bt, prices, plan, exposures, cost_buy, cost_sell, force_after=5):
    """plan: {date: {inst: weight}}; exposures: {date: float 0..1}; 权重按 (equity×exposure) 分配"""
    insts = list(prices.columns)
    last_px, missing = {}, {c: 10**6 for c in insts}
    shares, cash = {}, 1.0
    nav_hist = []

    for t in cal_bt:
        for c in insts:
            px = prices.at[t, c]
            if px == px:
                last_px[c] = float(px)
                missing[c] = 0
            else:
                missing[c] = missing.get(c, 0) + 1
        for c in list(shares):  # 长期停牌强平
            if missing.get(c, 0) >= force_after:
                cash += shares[c] * last_px[c] * (1 - cost_sell)
                shares.pop(c)
        if t in plan:
            w = plan[t] or {}
            w = {c: x for c, x in w.items() if c in last_px and last_px[c] > 0}
            equity = cash + sum(s * last_px[c] for c, s in shares.items())
            exp_ = float(exposures.get(t, 1.0))
            budget = equity * exp_
            new_shares = {c: budget * x / last_px[c] for c, x in w.items()}
            allc = set(new_shares) | set(shares)
            buy_v = sum(max(new_shares.get(c, 0) - shares.get(c, 0), 0) * last_px.get(c, 0) for c in allc)
            sell_v = sum(max(shares.get(c, 0) - new_shares.get(c, 0), 0) * last_px.get(c, 0) for c in allc)
            cost = buy_v * cost_buy + sell_v * cost_sell
            cash = equity - sum(s * last_px[c] for c, s in new_shares.items()) - cost
            shares = new_shares
        nav_hist.append((t, cash + sum(s * last_px.get(c, 0.0) for c, s in shares.items())))
    nav = pd.Series(dict(nav_hist))
    nav.index = pd.to_datetime(nav.index)
    return nav.sort_index()


# ------------------------------------------------------------------ 策略构建
def w_equal(picks):
    return {c: 1 / len(picks) for c in picks}


def topk(score, k, insts):
    s = score.dropna()
    s = s[s.index.isin(insts)]
    return list(s.nlargest(k).index) if len(s) >= k else list(s.index)


def build_plans(market, scores, rebals, prices_full, mcfg):
    """返回 {strategy: (plan, exposures)}; 波动率/均线基于全历史(2017起), 无前视"""
    K = mcfg["topk"]
    prices = prices_full.loc[cal_bt_all[market]]
    insts = list(prices.columns)
    rets = prices_full.pct_change()
    vol = rets.rolling(VOL_WINDOW).std() * np.sqrt(mcfg["ann_days"])  # 年化波动(全历史)
    idx = (1 + rets.mean(axis=1).fillna(0)).cumprod()  # 市场等权指数(全历史)
    sma = idx.rolling(SMA_WEEKS * 5).mean()

    plans = {"strat_base": {}, "strat_dropout": {}, "strat_riskparity": {},
             "strat_voltarget": {}, "strat_timing": {}, "strat_combo": {}}
    exps = {k: {} for k in plans}
    held = []  # dropout 状态

    for T in rebals:
        if T not in scores:
            continue
        score = scores[T]
        picks = topk(score, K, insts)
        # dropout: 持仓在全市场排名跌出 2K 才卖, 空位从头部补
        if held:
            rank_all = score.rank(ascending=False)
            alive = last_valid(prices, T, held)
            keep = [c for c in held if c in alive and c in rank_all and rank_all[c] <= DROP_BUF * K]
            for c in [x for x in topk(score, K, insts) if x not in keep]:
                if len(keep) >= K:
                    break
                keep.append(c)
            held = keep
        else:
            held = picks[:]
        # 风险平价权重(基于 T-1 及以前的波动率, 无前视)
        vhist = vol.shift(1).loc[:T]
        v = vhist.iloc[-1] if len(vhist) else pd.Series(dtype=float)
        def invvol(cs):
            vv = np.maximum(v.reindex(cs).fillna(np.nanmedian(v.values) if v.notna().any() else 0.2), 0.01)
            w = (1 / vv) / (1 / vv).sum()
            w = np.minimum(w, W_CAP)
            return (w / w.sum()).to_dict()
        plans["strat_base"][T] = w_equal(picks) if picks else {}
        plans["strat_dropout"][T] = w_equal(held) if held else {}
        plans["strat_riskparity"][T] = invvol(picks) if picks else {}
        plans["strat_combo"][T] = invvol(held) if held else {}
        plans["strat_voltarget"][T] = plans["strat_base"][T]  # base 权重 + 仓位层控制
        plans["strat_timing"][T] = plans["strat_base"][T]
        # 择时: 指数在 30 周均线上方满仓; 均线数据不足时默认满仓(而非空仓)
        sma_hist = sma.loc[:T].dropna()
        exps["strat_timing"][T] = 1.0 if (len(sma_hist) == 0 or idx.loc[T] > sma_hist.iloc[-1]) else 0.0
        exps["strat_combo"][T] = exps["strat_timing"][T]

    # 波动率目标: 先模拟 base 拿到其日收益, 再算因果的实现波动率
    nav_base = simulate(cal_bt_all[market], prices, plans["strat_base"], {T: 1.0 for T in plans["strat_base"]},
                        mcfg["cost_buy"], mcfg["cost_sell"])
    r = nav_base.pct_change()
    for T in plans["strat_voltarget"]:
        hist = r.loc[:T].iloc[:-1].dropna()
        if len(hist) >= 40 and hist.std() > 0:
            exps["strat_voltarget"][T] = float(min(1.0, TARGET_VOL / (hist.std() * np.sqrt(mcfg["ann_days"]))))
        else:
            exps["strat_voltarget"][T] = 1.0
    # combo 仓位 = min(波动率目标, 趋势)
    for T in plans["strat_combo"]:
        exps["strat_combo"][T] = min(exps["strat_combo"][T], exps["strat_voltarget"].get(T, 1.0))
    return plans, exps, nav_base


def last_valid(prices, T, cols):
    return [c for c in cols if prices.loc[:T, c].iloc[-1] == prices.loc[:T, c].iloc[-1]]


# ------------------------------------------------------------------ 主流程
cal_bt_all = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, choices=["cn", "us", "crypto", "gold"])
    args = ap.parse_args()
    mcfg = CFG["markets"][args.market]
    t0 = time.time()

    D = exp03.init_qlib(args.market)
    cal = [pd.Timestamp(d) for d in D.calendar(freq="day")]
    cal_bt = [d for d in cal if d >= pd.Timestamp(CFG["backtest_start"])]
    cal_bt_all[args.market] = cal_bt
    rebals = exp03.rebalance_dates(cal)

    uni = json.loads((ROOT / "data" / "universe.json").read_text())[args.market]
    insts = list(dict.fromkeys(u["inst"] for u in uni))
    px = D.features(D.list_instruments(D.instruments("all"), as_list=True), ["$close"], start_time=CFG["data_start"])
    prices_full = px["$close"].unstack("instrument")
    prices_full.index = pd.to_datetime(prices_full.index)
    prices_full = prices_full.reindex(cal)[insts]
    cal_bt_all[args.market] = cal_bt

    scores = get_scores(args.market)
    log(f"{args.market}: 信号 {len(scores)} 天, 回测 {cal_bt[0]:%Y-%m-%d}~{cal_bt[-1]:%Y-%m-%d}")

    plans, exps, _ = build_plans(args.market, scores, rebals, prices_full, mcfg)

    base_dir = ROOT / "results" / args.market
    prices_bt = prices_full.loc[cal_bt]
    for name, plan in plans.items():
        nav = simulate(cal_bt, prices_bt, plan, exps[name], mcfg["cost_buy"], mcfg["cost_sell"])
        d = base_dir / name
        d.mkdir(parents=True, exist_ok=True)
        nav.rename("nav").to_csv(d / "equity.csv", index_label="date")
        m = exp03.metrics(nav, mcfg["ann_days"])
        (d / "metrics.json").write_text(json.dumps({
            "market": args.market, "strategy": name, "alpha": "lgbm_weekly",
            "config": mcfg, "backtest": [str(cal_bt[0])[:10], str(cal_bt[-1])[:10]], **m,
        }, ensure_ascii=False, indent=1, default=str))
        log(f"  {name:18s} ret={m['total_return']:+.4f} sharpe={m.get('sharpe')} dd={m['max_drawdown']:+.3f}")
    log(f"{args.market} 策略实验完成 ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
