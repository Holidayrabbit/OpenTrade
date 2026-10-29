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
    "gru_static": "GRU神经网络(GPU)",
    "gru_weekly": "GRU每周再训练",
    "linear_static": "线性回归(基准)",
    "linear_weekly": "线性回归·每周再训练",
    "alstm_static": "ALSTM注意力网络(GPU)",
    "ridge_static": "Ridge回归(L2正则)",
    "mlp_static": "MLP前馈网络(GPU)",
    "strat_base": "策略基线:TopK等权",
    "strat_dropout": "TopK-Dropout(降换手)",
    "strat_riskparity": "风险平价加权",
    "strat_voltarget": "波动率目标20%",
    "strat_timing": "趋势择时(30周均线)",
    "strat_combo": "全策略叠加",
}
ORDER = ["buyhold", "static", "linear_static", "ridge_static", "monthly", "gru_static", "alstm_static", "mlp_static",
         "weekly", "linear_weekly", "strat_base", "strat_dropout", "strat_riskparity",
         "strat_voltarget", "strat_timing", "strat_combo"]


def load_variant(market, vdir):
    d = RESULTS / market / vdir
    if not (d / "metrics.json").exists():
        return None
    m = json.loads((d / "metrics.json").read_text())
    nav = pd.read_csv(d / "equity.csv", parse_dates=["date"]).set_index("date")["nav"]
    ic = pd.read_csv(d / "ic.csv", parse_dates=["date"]).set_index("date")["ic"] if (d / "ic.csv").exists() else None
    try:
        trades = json.loads((d / "trades.json").read_text())
    except Exception:  # noqa: BLE001
        trades = []
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
        idx = None
        for s in per_mkt.values():
            idx = s.index if idx is None else idx.union(s.index)
        idx = idx.sort_values()
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

    # ---------------- PNG 图表(注册 CJK 字体渲染中文图例; 缺字体自动回退英文标签)
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import font_manager
        import matplotlib.pyplot as plt

        cjk_ok = False
        for fp in [ROOT / "configs" / "fonts" / "NotoSansCJKsc-Regular.otf",
                   ROOT / "data" / "fonts" / "NotoSansCJKsc-Regular.otf",
                   Path.home() / "fonts" / "NotoSansCJKsc-Regular.otf"]:
            if fp.exists():
                try:
                    font_manager.fontManager.addfont(str(fp))
                    cjk_ok = True
                    break
                except Exception:  # noqa: BLE001
                    pass
        print(f"PNG 中文字体: {'Noto Sans CJK SC 已注册' if cjk_ok else '未找到, 图例回退英文'}", flush=True)
        EN_LABELS = {
            "buyhold": "Buy&Hold", "static": "Static (train once)", "monthly": "Monthly retrain",
            "weekly": "Weekly retrain", "gru_static": "GRU (GPU)", "linear_static": "Linear OLS",
            "linear_weekly": "Linear weekly", "ridge_static": "Ridge (L2)", "mlp_static": "MLP (GPU)",
            "alstm_static": "ALSTM (GPU)", "strat_base": "Strat: TopK equal", "strat_dropout": "TopK-Dropout",
            "strat_riskparity": "Risk parity", "strat_voltarget": "Vol target 20%",
            "strat_timing": "Trend timing", "strat_combo": "Combo",
        }
        MKT_EN = {"cn": "CN CSI300", "us": "US SP500", "crypto": "Crypto BinanceTop15", "gold": "Gold basket"}
        if cjk_ok:
            plt.rcParams.update({"font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
                                 "axes.unicode_minus": False})
            label = lambda v: VARIANT_LABELS.get(v, v)
            mkt_name = lambda m: CFG["markets"][m]["name"]
        else:
            plt.rcParams.update({"font.family": ["DejaVu Sans"], "axes.unicode_minus": False})
            label = lambda v: EN_LABELS.get(v, v)
            mkt_name = lambda m: MKT_EN.get(m, m)

        plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
                             "axes.spines.top": False, "axes.spines.right": False})
        colors = {"buyhold": "#8a8f98", "static": "#4c8dff", "monthly": "#ffb02e", "weekly": "#00d68f",
                  "gru_static": "#b47aff", "linear_static": "#7fd3ff", "linear_weekly": "#00b4d8",
                  "ridge_static": "#ffe08a", "mlp_static": "#c9a2ff", "alstm_static": "#ff8fd0",
                  "strat_base": "#9aa7ff", "strat_dropout": "#6f8dff", "strat_riskparity": "#4fd1c5",
                  "strat_voltarget": "#f6ad55", "strat_timing": "#fc8181", "strat_combo": "#ed64a6"}

        # 1. 组合净值
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for v in ["buyhold", "static", "monthly", "gru_static", "weekly"]:
            if v in port:
                nav = port[v]
                ax.plot(nav.index, nav.values, label=label(v),
                        color=colors.get(v), lw=1.8 if v == "weekly" else 1.2)
        ax.set_title("Portfolio (Equal-Weight 4 Markets) 2026 YTD")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(DOCS / "assets" / "portfolio.png")
        plt.close(fig)

        # 2. 各市场全部变体净值
        mkts = [m for m in markets_json if len(markets_json[m]["variants"]) >= 2]
        if mkts:
            fig, axes = plt.subplots(2, 2, figsize=(11, 7))
            for ax, m in zip(axes.flat, mkts):
                for v, vd in markets_json[m]["variants"].items():
                    s = pd.Series(dict(vd["nav"]), dtype=float)
                    ax.plot(s.index, s.values, label=label(v), color=colors.get(v),
                            lw=1.8 if v == "weekly" else 1.1)
                ax.set_title(f"{m}: {mkt_name(m)}", fontsize=10)
                ax.legend(fontsize=6, ncol=2)
                ax.tick_params(labelsize=7)
            for ax in axes.flat[len(mkts):]:
                ax.set_visible(False)
            fig.suptitle("Per-Market NAV 2026 YTD (all variants)", fontsize=11)
            fig.tight_layout()
            fig.savefig(DOCS / "assets" / "markets.png")
            plt.close(fig)

        # 3. 增益柱状图
        if len(gains) > 1:
            fig, ax = plt.subplots(figsize=(8, 3.6))
            names = [gains[n]["name"] if cjk_ok and "name" in gains[n] else n for n in gains]
            vals = [gains[n]["return_delta"] * 100 for n in gains]
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
