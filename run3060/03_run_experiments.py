#!/usr/bin/env python3
"""alpha2026 · Step 3: 回测实验(单市场单次调用)

复现 Microsoft Qlib 官方 Alpha158 + LightGBM 基准方法, 在自建四市场数据上做:
  - buyhold : 等权买入持有基准
  - static  : 年初用 2025-12-31 前全部数据训练一次, 全年不更新
  - monthly : 每月首个调仓日重新训练
  - weekly  : 每周首个调仓日重新训练(核心实验: 每周调整模型能否提高收益)
  - (可选) gru_static : Qlib GRU 深度模型对照(GPU)

交易约定(与 Qlib 默认回测一致, 略理想化但前后一致):
  - T 日收盘计算特征/打分, 以 T 日收盘价成交, 标签为 T+1→T+2 收益
  - 每周首个交易日调仓, Top-K 等权, 双边成本见 configs/markets.yaml
  - 连续 5 个交易日无行情的持仓按最后已知价强制平仓

用法: python 03_run_experiments.py --market cn --model lgbm --variants buyhold,static,monthly,weekly
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # qlib LGBModel 经 mlflow 记录指标
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "configs" / "markets.yaml").read_text())
RESULTS = ROOT / "results"

LGB_PARAMS = dict(  # Qlib 官方 Alpha158+LightGBM benchmark 调参(examples/benchmarks/LightGBM)
    loss="mse",
    colsample_bytree=0.8879,
    learning_rate=0.0421,
    subsample=0.8789,
    lambda_l1=205.6999,
    lambda_l2=580.9768,
    max_depth=8,
    num_leaves=210,
    num_threads=16,
    seed=42,
)


def log(m):
    print(f"[{pd.Timestamp.now():%H:%M:%S}] {m}", flush=True)


# ------------------------------------------------------------------ qlib 初始化
def init_qlib(market):
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(ROOT / "data" / "qlib" / market), region="cn")
    import logging

    for name in ["qlib", "qlib.data", "qlib.contrib"]:
        logging.getLogger(name).setLevel(logging.ERROR)
    return D


def build_dataset(market, D, for_gru=False):
    """Alpha158 handler + DatasetH; 归一化统计在 processor_fit 区间冻结, 防泄漏"""
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.processor import CSRankNorm, DropnaLabel, Fillna, RobustZScoreNorm

    proc_infer = [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "feature"}},
    ]
    hcfg = dict(
        start_time=CFG["data_start"],
        end_time=pd.Timestamp.now().strftime("%Y-%m-%d"),
        fit_start_time=CFG["processor_fit_start"],
        fit_end_time=CFG["processor_fit_end"],
        instruments="all",
        infer_processors=proc_infer,
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
        label=["Ref($close, -2) / Ref($close, -1) - 1"],
    )
    t0 = time.time()
    handler = Alpha158(**hcfg)
    ds = DatasetH(handler=handler, segments={"train": ("2018-01-01", "2018-12-31")})
    log(f"handler built ({time.time()-t0:.0f}s)")
    return ds


# ------------------------------------------------------------------ 调度工具
def rebalance_dates(cal):
    """回测区间内每个 ISO 周的第一个交易日"""
    bt = [d for d in cal if d >= pd.Timestamp(CFG["backtest_start"])]
    s = pd.Series(bt, index=bt)
    iso = s.groupby([s.index.isocalendar().year, s.index.isocalendar().week]).first()
    return [pd.Timestamp(d) for d in iso.values]


def make_model(kind, n_feat):
    if kind == "lgbm":
        from qlib.contrib.model.gbdt import LGBModel

        return LGBModel(**LGB_PARAMS)
    elif kind == "gru":
        from qlib.contrib.model.pytorch_gru import GRU

        return GRU(d_feat=n_feat, hidden_size=64, num_layers=2, dropout=0.1,
                   n_epochs=60, lr=2e-3, batch_size=2000, early_stop=8, seed=42, GPU=0)
    raise ValueError(kind)


def fit_at(ds, cal, labels_end, kind):
    """在 (data_start, labels_end] 上重训: 末 15% 作验证(早停)"""
    idx = [d for d in cal if d <= labels_end]
    n_valid = max(60, int(len(idx) * 0.15))
    n_valid = min(n_valid, len(idx) - 60)
    train_end, valid_start = idx[len(idx) - n_valid - 1], idx[len(idx) - n_valid]
    ds.segments = {
        "train": (pd.Timestamp(CFG["data_start"]), train_end),
        "valid": (valid_start, labels_end),
    }
    model = make_model(kind, N_FEATURES[0])
    t0 = time.time()
    if kind == "lgbm":
        model.fit(ds, num_boost_round=1000, early_stopping_rounds=50, verbose_eval=-1)
    else:
        evals = {}
        model.fit(ds, evals_result=evals)
    dt = time.time() - t0
    return model, dict(train_end=str(train_end)[:10], labels_end=str(labels_end)[:10], fit_sec=round(dt, 1))


def predict_window(model, ds, dates):
    """预测 [dates[0], dates[-1]] 窗口内所有交易日的得分"""
    if not dates:
        return pd.DataFrame()
    ds.segments = dict(ds.segments) | {"test": (dates[0], dates[-1])}
    pred = model.predict(ds, segment="test")
    if isinstance(pred, np.ndarray):
        from qlib.data.dataset.handler import DataHandlerLP

        feats = ds.prepare("test", data_key=DataHandlerLP.DK_I)
        pred = pd.Series(pred, index=feats.index)
    pred = pred.reset_index()
    pred.columns = ["datetime", "instrument", "score"]
    pred["datetime"] = pd.to_datetime(pred["datetime"])
    return pred


# ------------------------------------------------------------------ 交易模拟器
def simulate(cal_bt, prices, rebal_map, cost_buy, cost_sell, force_after=5):
    """share 级模拟: rebal_map = {date: [target instruments]}"""
    insts = list(prices.columns)
    last_px = {}
    missing = {c: 10**6 for c in insts}
    shares = {}
    cash = 1.0
    nav_hist, trades = [], []

    for t in cal_bt:
        for c in insts:
            px = prices.at[t, c] if c in prices.columns else np.nan
            if px == px:  # not NaN
                last_px[c] = float(px)
                missing[c] = 0
            else:
                missing[c] += 1
        # 长期无行情 → 按最后已知价强平
        for c in list(shares):
            if missing.get(c, 0) >= force_after:
                cash += shares[c] * last_px[c] * (1 - cost_sell)
                trades.append({"date": str(t)[:10], "type": "force_sell", "inst": c, "value": shares[c] * last_px[c]})
                shares.pop(c)
        equity = cash + sum(s * last_px.get(c, 0.0) for c, s in shares.items())
        if t in rebal_map and equity > 0:
            targets = [c for c in rebal_map[t] if c in last_px]
            if targets:
                tv = equity / len(targets)
                new_shares = {c: tv / last_px[c] for c in targets}
                all_c = set(targets) | set(shares)
                buy_v = sum(max(new_shares.get(c, 0) - shares.get(c, 0), 0) * last_px.get(c, 0) for c in all_c)
                sell_v = sum(max(shares.get(c, 0) - new_shares.get(c, 0), 0) * last_px.get(c, 0) for c in all_c)
                cost = buy_v * cost_buy + sell_v * cost_sell
                bought = [c for c in targets if new_shares[c] - shares.get(c, 0) > tv * 0.01]
                sold = [c for c in shares if c not in new_shares]
                shares = new_shares
                cash = equity - sum(s * last_px[c] for c, s in new_shares.items()) - cost
                trades.append({
                    "date": str(t)[:10], "type": "rebalance",
                    "bought": bought, "sold": sold,
                    "turnover": round((buy_v + sell_v) / 2 / equity, 4),
                    "n_hold": len(targets),
                })
        nav = cash + sum(s * last_px.get(c, 0.0) for c, s in shares.items())
        nav_hist.append((t, nav))
    nav = pd.Series(dict(nav_hist), name="nav")
    nav.index = pd.to_datetime(nav.index)
    return nav.sort_index(), trades


def metrics(nav, ann_days):
    ret = nav.pct_change().dropna()
    if len(ret) < 5:
        return {}
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    years = len(ret) / ann_days
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    vol = float(ret.std() * np.sqrt(ann_days))
    sharpe = float(ret.mean() / ret.std() * np.sqrt(ann_days)) if ret.std() > 0 else np.nan
    dd = float((nav / nav.cummax() - 1).min())
    wk = nav.resample("W-FRI").last().dropna().pct_change().dropna()
    return {
        "total_return": round(total, 4),
        "annual_return": round(float(ann), 4) if ann == ann else None,
        "annual_vol": round(vol, 4),
        "sharpe": round(sharpe, 3) if sharpe == sharpe else None,
        "max_drawdown": round(dd, 4),
        "calmar": round(float(ann / abs(dd)), 3) if dd < 0 and ann == ann else None,
        "weekly_win_rate": round(float((wk > 0).mean()), 4) if len(wk) else None,
    }


# ------------------------------------------------------------------ 主流程
N_FEATURES = [158]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, choices=["cn", "us", "crypto", "gold"])
    ap.add_argument("--model", default="lgbm", choices=["lgbm", "gru"])
    ap.add_argument("--variants", default="buyhold,static,monthly,weekly")
    args = ap.parse_args()

    mcfg = CFG["markets"][args.market]
    D = init_qlib(args.market)

    cal = [pd.Timestamp(d) for d in D.calendar(freq="day")]
    bt_start = pd.Timestamp(CFG["backtest_start"])
    cal_bt = [d for d in cal if d >= bt_start]
    log(f"{args.market}: 回测交易日 {cal_bt[0]:%Y-%m-%d} ~ {cal_bt[-1]:%Y-%m-%d} 共{len(cal_bt)}天")

    # 全量收盘价面板(用于模拟/IC)
    uni = json.loads((ROOT / "data" / "universe.json").read_text())[args.market]
    insts = [u["inst"] for u in uni]
    px_df = D.features(D.list_instruments(D.instruments("all"), as_list=True), ["$close"],
                       start_time=CFG["data_start"])
    px_df = px_df["$close"].unstack("instrument")
    px_df.index = pd.to_datetime(px_df.index)
    prices_bt = px_df.reindex(cal_bt)[insts]
    label_panel = (px_df.shift(-2) / px_df.shift(-1) - 1).reindex(cal_bt)  # 与训练标签同口径

    rebals = rebalance_dates(cal)
    # 各再训练频率的 refit 调仓日
    refit_plan = {
        "static": [rebals[0]],
        "monthly": [d for i, d in enumerate(rebals) if i == 0 or d.month != rebals[i - 1].month],
        "weekly": rebals,
    }

    outdir_base = RESULTS / args.market
    outdir_base.mkdir(parents=True, exist_ok=True)
    run_meta = {"market": args.market, "model": args.model, "config": mcfg,
                "backtest": [str(cal_bt[0])[:10], str(cal_bt[-1])[:10]],
                "n_calendar_days": len(cal_bt), "n_universe": len(insts)}

    variants = args.variants.split(",")

    # ---- buyhold
    if "buyhold" in variants:
        targets0 = [c for c in insts if prices_bt.iloc[0][c] == prices_bt.iloc[0][c]]
        nav, trades = simulate(cal_bt, prices_bt, {cal_bt[0]: targets0}, mcfg["cost_buy"], mcfg["cost_sell"])
        save(outdir_base / "buyhold", run_meta, nav, trades, None, None)
        log(f"buyhold done: {metrics(nav, mcfg['ann_days'])}")

    # ---- 模型变体
    need_model = [v for v in variants if v != "buyhold"]
    if need_model:
        from qlib.data.dataset.handler import DataHandlerLP

        ds = build_dataset(args.market, D)
        # 特征数(供 GRU d_feat): 总列数 − label 列数(label 列名为表达式字符串而非 LABEL0)
        ds.segments = {"probe": (CFG["processor_fit_start"], CFG["processor_fit_end"])}
        probe = ds.prepare("probe", data_key=DataHandlerLP.DK_L)
        try:
            n_label = ds.prepare("probe", col_set="label", data_key=DataHandlerLP.DK_L).shape[1]
        except Exception:  # noqa: BLE001
            n_label = 1
        N_FEATURES[0] = max(len(probe.columns) - n_label, 1)
        log(f"features: {N_FEATURES[0]}")

        for variant in need_model:
            t_start = time.time()
            refits = refit_plan[variant]
            scores_by_date = {}
            refit_log = []
            for ri, T in enumerate(refits):
                pos = cal.index(T)
                labels_end = cal[pos - 2]
                model, fmeta = fit_at(ds, cal, labels_end, args.model)
                refit_log.append(fmeta | {"refit_date": str(T)[:10]})
                win_dates = [d for d in cal_bt if d >= T]
                if ri + 1 < len(refits):
                    win_dates = [d for d in win_dates if d < refits[ri + 1]]
                pred = predict_window(model, ds, win_dates)
                for dt_, grp in pred.groupby("datetime"):
                    scores_by_date[pd.Timestamp(dt_)] = grp.set_index("instrument")["score"]
                log(f"  [{variant}] refit@{T:%Y-%m-%d} train≤{fmeta['labels_end']} {fmeta['fit_sec']}s "
                    f"({ri+1}/{len(refits)}) pred_days={len(win_dates)}")
                del model
            # 周度调仓映射
            rebal_map = {}
            for T in rebals:
                if T in scores_by_date:
                    s = scores_by_date[T].dropna()
                    s = s[s.index.isin(insts)]
                    if len(s) >= mcfg["topk"]:
                        rebal_map[T] = list(s.nlargest(mcfg["topk"]).index)
            nav, trades = simulate(cal_bt, prices_bt, rebal_map, mcfg["cost_buy"], mcfg["cost_sell"])
            # IC(逐日)
            ic_rows = {}
            for t, s in scores_by_date.items():
                if t in label_panel.index:
                    lab = label_panel.loc[t].reindex(s.index)
                    pair = pd.concat([s.rename("s"), lab.rename("l")], axis=1).dropna()
                    if len(pair) >= 10:
                        ic_rows[t] = pair["s"].corr(pair["l"], method="spearman")
            ic = pd.Series(ic_rows).sort_index()
            ic_summary = {
                "mean_ic": round(float(ic.mean()), 4) if len(ic) else None,
                "ic_ir": round(float(ic.mean() / ic.std()), 3) if len(ic) and ic.std() > 0 else None,
                "ic_positive_rate": round(float((ic > 0).mean()), 4) if len(ic) else None,
                "n_days": len(ic),
            }
            vdir = outdir_base / (variant if args.model == "lgbm" else f"{args.model}_{variant}")
            save(vdir, run_meta | {"variant": variant}, nav, trades, ic_summary, ic, refit_log)
            log(f"[{vdir.name}] {metrics(nav, mcfg['ann_days'])} IC={ic_summary} "
                f"elapsed={time.time()-t_start:.0f}s")


def save(vdir, meta, nav, trades, ic_summary, ic, refit_log=None):
    vdir.mkdir(parents=True, exist_ok=True)
    nav.rename("nav").to_csv(vdir / "equity.csv", index_label="date")
    m = metrics(nav, meta["config"]["ann_days"]) if "config" in meta else metrics(nav, 252)
    (vdir / "metrics.json").write_text(json.dumps({**meta, **m, "ic": ic_summary}, ensure_ascii=False, indent=1, default=str))
    (vdir / "trades.json").write_text(json.dumps(trades, ensure_ascii=False, default=str))
    if refit_log:
        (vdir / "refit_log.json").write_text(json.dumps(refit_log, ensure_ascii=False, indent=1))
    if ic is not None and len(ic):
        ic.rename("ic").to_csv(vdir / "ic.csv", index_label="date")


if __name__ == "__main__":
    main()
