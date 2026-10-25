#!/usr/bin/env python3
"""alpha2026 · Step 4: 聚合回测结果 → docs/data/*.json(前端) + docs/assets/*.png

组合口径: 四市场子策略等权、日度再平衡、跨市场零成本(子策略内部成本已计入各自净值)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "configs" / "markets.yaml").read_text())
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

VARIANT_LABELS = {
    "buyhold": "等权买入持有",
    "static": "静态模型(年初训练一次)",
    "monthly": "每月再训练",
    "weekly": "每周再训练",
    "gru_static": "GRU静态(GPU)",
    "gru_weekly": "GRU每周再训练",
}
ORDER = ["buyhold", "static", "monthly", "gru_static", "weekly"]


def load_variant(market, vdir):
    d = RESULTS / market / vdir
    if not (d / "metrics.json").exists():
        return None
    m = json.loads((d / "metrics.json").read_text())
    nav = pd.read_csv(d / "equity.csv", parse_dates=["date"]).set_index("date")["nav"]
    ic = pd.read_csv(d / "ic.csv", parse_dates=["date"]).set_index("date")["ic"] if (d / "ic.csv").exists() else None
    trades = json.loads((d / "trades.json").read_text())
    out = {
        "dir": vdir,
        "label": VARIANT_LABELS.get(vdir, vdir),
        "metrics": m,
        "nav": [[t.strftime("%Y-%m-%d"), round(float(v), 6)] for t, v in nav.items()],
        "n_trades": len(trades),
        "recent_trades": [
            {"date": x["date"], "bought": x.get("bought", [])[:8], "sold": x.get("sold", [])[:8]}
            for x in trades[-25:] if x.get("type") == "rebalance"
        ][-12:],
    }
    if ic is not None and len(ic):
        out["ic_daily"] = [[t.strftime("%Y-%m-%d"), round(float(v), 4)] for t, v in ic.items()]
        out["ic_4w_ma"] = [[t.strftime("%Y-%m-%d"), round(float(v), 4)]
                           for t, v in ic.rolling(20).mean().dropna().items()]
    return out, nav


def portfolio_navs(nav_dict_by_variant):
    """nav_dict_by_variant: {variant: {market: Series}} → 等权日度再平衡组合净值"""
    out = {}
    for variant, per_mkt in nav_dict_by_variant.items():
        if len(per_mkt) < 4:
            continue
        df = pd.DataFrame(per_mkt)
        idx = df.index.union(sum([s.index for s in per_mkt.values()], pd.DatetimeIndex([]))).sort_values()
        df = df.reindex(idx).ffill().bfill()
        rets = df.pct_change().fillna(0.0)
        port_ret = rets.mean(axis=1)
        out[variant] = (1 + port_ret).cumprod()
    return out


def metr(nav, ann_days=252):
    ret = nav.pct_change().dropna()
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    years = len(ret) / ann_days
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    dd = float((nav / nav.cummax() - 1).min())
    return dict(total_return=round(total, 4), annual_return=round(ann, 4) if ann == ann else None,
                sharpe=round(float(ret.mean() / ret.std() * np.sqrt(ann_days)), 3) if ret.std() > 0 else None,
                max_drawdown=round(dd, 4))


def main():
    universe = json.loads((ROOT / "data" / "universe.json").read_text())
    meta = json.loads((ROOT / "data" / "meta.json").read_text())
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    (DOCS / "assets").mkdir(parents=True, exist_ok=True)

    all_variant_navs = {}  # variant -> {market: Series}
    markets_json, summary_markets = {}, {}
    bt_start, bt_end = None, None

    for market, mcfg in CFG["markets"].items():
        mdir = RESULTS / market
        if not mdir.exists():
            continue
        variants, navs = {}, {}
        for vdir in sorted([p.name for p in mdir.iterdir() if p.is_dir()],
                           key=lambda v: ORDER.index(v) if v in ORDER else 99):
            r = load_variant(market, vdir)
            if r:
                variants[vdir] = r[0]
                navs[vdir] = r[1]
                d0, d1 = r[1].index[0], r[1].index[-1]
                bt_start = d0 if bt_start is None else min(bt_start, d0)
                bt_end = d1 if bt_end is None else max(bt_end, d1)
                all_variant_navs.setdefault(vdir, {})[market] = r[1]
        if not variants:
            continue
        markets_json[market] = {
            "key": market, "name": mcfg["name"], "topk": mcfg["topk"],
            "n_universe": len(universe.get(market, [])),
            "instruments": [{"inst": u["inst"], "name": u["name"]} for u in universe.get(market, [])][:400],
            "variants": variants,
            "backtest": variants[list(variants)[0]]["metrics"]["backtest"],
        }
        summary_markets[market] = {
            "name": mcfg["name"], "n_universe": len(universe.get(market, [])),
            "variants": {k: {kk: v["metrics"].get(kk) for kk in
                             ["total_return", "annual_return", "sharpe", "max_drawdown", "weekly_win_rate", "ic"]}
                         for k, v in variants.items()},
        }

    port = portfolio_navs(all_variant_navs)
    portfolio_json = {
        "name": "四市场等权组合",
        "variants": {},
        "metrics": {},
    }
    for v, nav in port.items():
        portfolio_json["variants"][v] = {"label": VARIANT_LABELS.get(v, v),
                                         "nav": [[t.strftime("%Y-%m-%d"), round(float(x), 6)] for t, x in nav.items()]}
        portfolio_json["metrics"][v] = metr(nav)

    summary = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "backtest_range": [str(bt_start)[:10], str(bt_end)[:10]] if bt_start else None,
        "data_fetched_at": meta.get("fetched_at"),
        "markets": summary_markets,
        "portfolio": {k: v for k, v in portfolio_json["metrics"].items()},
        "engine": "Microsoft Qlib Alpha158 + LightGBM/GRU · walk-forward",
    }

    # 周度再训练增益
    gains = {}
    for market, mv in summary_markets.items():
        w, s = mv["variants"].get("weekly"), mv["variants"].get("static")
        if w and s:
            gains[market] = {
                "name": mv["name"],
                "return_delta": round((w.get("total_return") or 0) - (s.get("total_return") or 0), 4),
                "sharpe_delta": round((w.get("sharpe") or 0) - (s.get("sharpe") or 0), 3),
                "ic_delta": round((w.get("ic") or {}).get("mean_ic", 0) - (s.get("ic") or {}).get("mean_ic", 0), 4),
            }
    pw, ps = portfolio_json["metrics"].get("weekly"), portfolio_json["metrics"].get("static")
    if pw and ps:
        gains["portfolio"] = {"name": "四市场组合",
                              "return_delta": round(pw["total_return"] - ps["total_return"], 4),
                              "sharpe_delta": round(pw["sharpe"] - ps["sharpe"], 3)}
    summary["weekly_vs_static_gain"] = gains

    (DOCS / "data" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False))
    (DOCS / "data" / "portfolio.json").write_text(json.dumps(portfolio_json, ensure_ascii=False))
    for k, v in markets_json.items():
        (DOCS / f"data/{k}.json").write_text(json.dumps(v, ensure_ascii=False))
    print("docs/data 写入:", [p.name for p in (DOCS / "data").iterdir()])

    # ---------------- PNG 图表(英文标签, 避免远端缺中文字体)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
                             "axes.spines.top": False, "axes.spines.right": False})
        colors = {"buyhold": "#8a8f98", "static": "#4c8dff", "monthly": "#ffb02e", "weekly": "#00d68f",
                  "gru_static": "#b47aff"}

        # 1. 组合净值
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for v in ["buyhold", "static", "monthly", "gru_static", "weekly"]:
            if v in port:
                nav = port[v]
                ax.plot(nav.index, nav.values, label=VARIANT_LABELS.get(v, v) or v,
                        color=colors.get(v), lw=1.8 if v == "weekly" else 1.2)
        ax.set_title("Portfolio (Equal-Weight 4 Markets) 2026 YTD")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(DOCS / "assets" / "portfolio.png")
        plt.close(fig)

        # 2. 各市场 weekly vs static vs buyhold
        mkts = [m for m in markets_json if len(markets_json[m]["variants"]) >= 2]
        if mkts:
            fig, axes = plt.subplots(2, 2, figsize=(11, 7))
            for ax, m in zip(axes.flat, mkts):
                for v, vd in markets_json[m]["variants"].items():
                    s = pd.Series(dict(vd["nav"]), dtype=float)
                    ax.plot(s.index, s.values, label=vd["label"] or v, color=colors.get(v),
                            lw=1.8 if v == "weekly" else 1.1)
                ax.set_title(f"{m}: {markets_json[m]['name']}", fontsize=10)
                ax.legend(fontsize=7)
                ax.tick_params(labelsize=7)
            for ax in axes.flat[len(mkts):]:
                ax.set_visible(False)
            fig.suptitle("Per-Market NAV 2026 YTD (weekly-retrain vs static vs buy&hold)", fontsize=11)
            fig.tight_layout()
            fig.savefig(DOCS / "assets" / "markets.png")
            plt.close(fig)

        # 3. 增益柱状图
        if len(gains) > 1:
            fig, ax = plt.subplots(figsize=(8, 3.6))
            names = list(gains)
            vals = [gains[n]["return_delta"] * 100 for n in names]
            ax.bar(names, vals, color=["#00d68f" if x > 0 else "#ff5c5c" for x in vals])
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title("Weekly-Retrain vs Static: Total Return Delta (pp, 2026 YTD)")
            ax.tick_params(labelsize=8)
            fig.tight_layout()
            fig.savefig(DOCS / "assets" / "gain.png")
            plt.close(fig)
        print("docs/assets 写入: portfolio.png markets.png gain.png")
    except Exception as e:  # noqa: BLE001
        print("PNG 生成失败(不影响站点):", repr(e))


if __name__ == "__main__":
    main()
