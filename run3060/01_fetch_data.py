#!/usr/bin/env python3
"""alpha2026 · Step 1: 抓取四市场日线数据(A股/美股/加密/黄金)

数据源(全部免费公开接口, 国内网络可达):
  - A股/ETF:  东方财富 push2his K线接口(前复权)
  - 美股:     东方财富(通过搜索API解析 secid)
  - 加密:     data-api.binance.vision(币安公开行情镜像)
  - 黄金:     新浪伦敦金现 / 东财COMEX / 上金所Au99.99(akshare) / 黄金ETF/矿股(东财)

输出:
  data/qlib_csv/{market}/all.csv   # 长表: symbol,date,open,high,low,close,volume,amount
  data/universe.json               # 每市场标的池(名称/来源/起始日期)
  data/meta.json                   # 抓取元信息(时间/数量/失败清单)
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CFG = __import__("yaml").safe_load((ROOT / "configs" / "markets.yaml").read_text())

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
RAW_DIR = ROOT / "data" / "qlib_csv"
BINANCE = "https://data-api.binance.vision"
TODAY = dt.date.today().strftime("%Y-%m-%d")

os.makedirs(RAW_DIR, exist_ok=True)


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def http_get(url, params=None, headers=None, retries=4, timeout=20):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers={**UA, **(headers or {})}, timeout=timeout)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:100]}")
            if r.status_code not in (429, 500, 501, 502, 503, 504):
                raise last  # 4xx(限流封禁等)不重试
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(min(5 * (i + 1), 20))  # 5,10,15,20s 指数退避
    raise last


# ---------------------------------------------------------------- 东财 K 线
EM_KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


def em_kline(secid, beg, end, fqt=1):
    """东财日线, 返回 DataFrame[date,open,high,low,close,volume,amount]; 按5年分块规避单次行数上限"""
    frames = []
    b = int(beg.replace("-", ""))
    e = int(end.replace("-", ""))
    for cb, ce in _year_chunks(b, e, 5):
        params = dict(
            secid=secid, fields1="f1,f2,f3,f4,f5,f6", fields2="f51,f52,f53,f54,f55,f56,f57",
            klt=101, fqt=fqt, beg=cb, end=ce,
        )
        js = http_get(EM_KLINE, params).json()
        data = (js.get("data") or {})
        klines = data.get("klines") or []
        if not klines:
            continue
        rows = [k.split(",") for k in klines]
        frames.append(pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume", "amount"]))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _year_chunks(b, e, years):
    out = []
    y0 = int(str(b)[:4])
    y1 = int(str(e)[:4])
    while y0 <= y1:
        y2 = min(y0 + years - 1, y1)
        out.append((f"{y0}0101", f"{y2}1231"))
        y0 = y2 + 1
    return out


def cn_secid(code):
    # 沪市: 股票6xx / 基金5xx → 1. ; 深市: 股票0/3xx / 基金1xx → 0.
    return ("1." if code.startswith(("6", "5")) else "0.") + code


# ---------------------------------------------------------------- 美股 secid 映射
_search_cache = {}


def us_secid(ticker):
    """东财搜索接口解析美股 QuoteID(如 105.AAPL), 带本地缓存"""
    if ticker in _search_cache:
        return _search_cache[ticker]
    try:
        js = http_get(
            "https://searchapi.eastmoney.com/api/suggest/get",
            dict(input=ticker, type="14", token="D43BF722C8E33BDC906FB84D85E326E8", count="8"),
        ).json()
        for item in ((js.get("QuotationCodeTable") or {}).get("Data") or []):
            if item.get("Classify") == "UsStock" and item.get("Code", "").upper() == ticker.upper():
                secid = item.get("MktNum") + "." + item.get("Code")
                _search_cache[ticker] = secid
                return secid
    except Exception as e:  # noqa: BLE001
        log(f"  search fail {ticker}: {e!r}")
    _search_cache[ticker] = None
    return None


# ---------------------------------------------------------------- 腾讯行情(主源)
TX_KLINE = {
    "cn": "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
    "us": "https://web.ifzq.gtimg.cn/appstock/app/usfqkline/get",
}


def _date_chunks(start, end, days):
    out = []
    cur = dt.date.fromisoformat(start)
    stop = dt.date.fromisoformat(end)
    while cur <= stop:
        nxt = min(cur + dt.timedelta(days=days), stop)
        out.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt + dt.timedelta(days=1)
    return out


def tx_kline(tx_symbol, kind):
    """腾讯日线(前复权). kind: cn→fqkline / us→usfqkline; 分块抓取规避单次条数上限
    行格式: [date, open, close, high, low, volume, ...]"""
    frames = []
    for b, e in _date_chunks(CFG["data_start"], TODAY, 900):
        params = f"param={tx_symbol},day,{b},{e},640,qfq"
        r = http_get(TX_KLINE[kind], params=params)
        js = r.json()
        data = (js.get("data") or {}).get(tx_symbol) or {}
        rows = data.get("qfqday") or data.get("day") or []
        if not rows:
            continue
        frames.append(pd.DataFrame([x[:6] for x in rows], columns=["date", "open", "close", "high", "low", "volume"]))
        time.sleep(0.4)  # 腾讯限流防护: 控制单标的分块请求节奏
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["amount"] = math.nan
    return df[["date", "open", "high", "low", "close", "volume", "amount"]]


def cn_tx_symbol(code):
    return ("sh" if code.startswith(("6", "5")) else "sz") + code


_tx_map_path = ROOT / "data" / "raw" / "_tx_us_map.json"
_tx_map = json.loads(_tx_map_path.read_text()) if _tx_map_path.exists() else {}


def tx_us_symbol(ticker):
    """smartbox 搜索: BRK.B → usBRK.B.N (后缀 .OQ纳斯达克/.N纽交所)"""
    key = ticker.upper()
    if key in _tx_map:
        return _tx_map[key]
    sym = None
    try:
        r = http_get(f"https://smartbox.gtimg.cn/s3/?v=2&q={ticker}&t=us")
        m = re.search(r"us~([a-z0-9.\-]+)~", r.text)
        if m and len(m.group(1)) > 2:
            sym = "us" + m.group(1).upper()
    except Exception as e:  # noqa: BLE001
        log(f"  smartbox fail {ticker}: {e!r}")
    _tx_map[key] = sym
    _tx_map_path.parent.mkdir(parents=True, exist_ok=True)
    _tx_map_path.write_text(json.dumps(_tx_map))
    return sym


# ---------------------------------------------------------------- 新浪伦敦金现
def sina_xau():
    r = http_get(
        "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20t=/GlobalFuturesService.getGlobalFuturesDailyKLine",
        params={"symbol": "XAU"},
        headers={"Referer": "https://finance.sina.com.cn"},
        timeout=25,
    )
    arr = json.loads(re.search(r"\[.*\]", r.text, re.S).group(0))
    df = pd.DataFrame(arr)[["date", "open", "high", "low", "close"]]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = math.nan
    df["amount"] = math.nan
    return df[["date", "open", "high", "low", "close", "volume", "amount"]]


# ---------------------------------------------------------------- 上金所
def sge_au99(symbol="Au99.99"):
    import akshare as ak
    df = ak.spot_hist_sge(symbol=symbol)
    df = df.rename(columns={"date": "date", "open": "open", "close": "close", "high": "high", "low": "low"})
    df["volume"] = math.nan
    df["amount"] = math.nan
    df["date"] = df["date"].astype(str)
    return df[["date", "open", "high", "low", "close", "volume", "amount"]]


# ---------------------------------------------------------------- 币安
def binance_klines(symbol, start_ms, end_ms):
    out = []
    cur = start_ms
    while cur < end_ms:
        js = http_get(
            f"{BINANCE}/api/v3/klines",
            dict(symbol=symbol, interval="1d", startTime=cur, endTime=end_ms, limit=1000),
            timeout=25,
        ).json()
        if not isinstance(js, list) or not js:
            break
        out.extend(js)
        cur = js[-1][6] + 1  # closeTime + 1ms
        if len(js) < 1000:
            break
        time.sleep(0.15)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=["openT", "open", "high", "low", "close", "vol", "closeT", "quoteVol", "n", "tb", "tq", "ig"])
    df["date"] = pd.to_datetime(df["openT"], unit="ms").dt.strftime("%Y-%m-%d")
    for c in ["open", "high", "low", "close", "vol", "quoteVol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", "open", "high", "low", "close"]].assign(volume=df["vol"], amount=df["quoteVol"])


# ---------------------------------------------------------------- 标的池构建
def universe_cn():
    import akshare as ak
    for fn, arg in [("index_stock_cons_csindex", "000300"), ("index_stock_cons", "000300")]:
        try:
            df = getattr(ak, fn)(symbol=arg)
            codes = df["成分券代码"].astype(str).str.zfill(6)
            names = df["成分券名称"].astype(str)
            return [{"inst": ("SH" if c.startswith("6") else "SZ") + c, "code": c, "name": n} for c, n in zip(codes, names)]
        except Exception as e:  # noqa: BLE001
            log(f"cn universe via {fn} failed: {e!r}, fallback")
    raise RuntimeError("无法获取沪深300成分股")


def universe_us():
    df = pd.read_csv(ROOT / "configs" / "sp500.csv")
    return [{"inst": re.sub(r"[^A-Za-z0-9]", "", s).upper(), "code": s, "name": n} for s, n in zip(df["Symbol"], df["Security"])]


def universe_crypto(cfg):
    log("  binance 24h tickers ...")
    tk = http_get(f"{BINANCE}/api/v3/ticker/24hr", timeout=30).json()
    excl = CFG["crypto_exclude_substrings"]
    cands = [t for t in tk if t["symbol"].endswith("USDT")]
    cands = [t for t in cands if not any(x in t["symbol"] for x in excl)]
    cands.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
    start_ms = int(pd.Timestamp(CFG["data_start"]).timestamp() * 1000)
    min_list = pd.Timestamp(CFG["crypto_min_list_date"])
    picked = []
    for t in cands:
        if len(picked) >= CFG["crypto_top_n"]:
            break
        sym = t["symbol"]
        try:
            k = binance_klines(sym, start_ms, int(time.time() * 1000))
        except Exception:  # noqa: BLE001
            continue
        if k.empty or pd.Timestamp(k["date"].iloc[0]) > min_list:
            continue
        picked.append({"inst": sym, "code": sym, "name": sym.replace("USDT", "/USDT"), "_klines": k})
        log(f"  + {sym} (first={k['date'].iloc[0]}, last close={k['close'].iloc[-1]:.2f})")
    return picked


def universe_gold():
    return [{"inst": g["inst"], "code": g["inst"], "name": g["name"], "_source": g["source"]} for g in CFG["gold_universe"]]


# ---------------------------------------------------------------- 抓取调度
def fetch_one(inst_meta, market, force=False):
    """抓取单标的并写入 data/raw/{market}/{inst}.csv; 已有且新鲜的缓存直接跳过"""
    raw_dir = ROOT / "data" / "raw" / market
    raw_dir.mkdir(parents=True, exist_ok=True)
    inst = inst_meta["inst"]
    fp = raw_dir / f"{inst}.csv"
    stale_before = (dt.date.today() - dt.timedelta(days=5)).isoformat()
    if not force and fp.exists():
        try:
            old = pd.read_csv(fp)
            if len(old) and str(old["date"].max()) >= stale_before:
                return inst_meta, old
        except Exception:  # noqa: BLE001
            pass

    last_err = None
    for attempt in range(3):
        try:
            df = _fetch_one_inner(inst_meta, market)
            if df is None or df.empty:
                raise RuntimeError("empty")
            df = df[df["date"] >= CFG["data_start"]].reset_index(drop=True)
            if not len(df):
                raise RuntimeError("no rows in range")
            df.insert(0, "symbol", inst)
            df.to_csv(fp, index=False)
            time.sleep(0.2 + 0.3 * attempt)
            return inst_meta, df
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise last_err


def _fetch_one_inner(inst_meta, market):
    inst = inst_meta["inst"]
    if market == "gold":
        src = inst_meta["_source"]
        if src == "sina_xau":
            return sina_xau()
        if src == "sge":
            return sge_au99()
        if src == "em_global_futures":
            return em_kline("101.GC00Y", CFG["data_start"], TODAY)
        if src == "em_us":
            txsym = tx_us_symbol(inst_meta["code"].replace("US", ""))
            return tx_kline(txsym, "us") if txsym else pd.DataFrame()
        return tx_kline(cn_tx_symbol(inst[2:]), "cn")
    if market == "cn":
        return tx_kline(cn_tx_symbol(inst[2:]), "cn")
    if market == "crypto":
        return inst_meta.pop("_klines")
    # us
    txsym = tx_us_symbol(inst_meta["code"])
    return tx_kline(txsym, "us") if txsym else pd.DataFrame()


def run(market):
    log(f"=== {market} 构建标的池 ===")
    if market == "cn":
        uni = universe_cn()
    elif market == "us":
        uni = universe_us()
    elif market == "crypto":
        uni = universe_crypto(CFG)
    else:
        uni = universe_gold()

    ok, fail = [], []
    frames = []
    pending = list(uni)  # 待抓清单; 每轮失败的进入下一轮(带冷却), 最多 3 轮

    def worker(m):
        try:
            return fetch_one(m, market)
        except Exception as e:  # noqa: BLE001
            return (m, repr(e)[:80])

    for round_no in range(3):
        if not pending:
            break
        if round_no:
            cooldown = 60 * round_no
            log(f"  {market}: 第{round_no + 1}轮重试 {len(pending)} 个标的, 冷却 {cooldown}s ...")
            time.sleep(cooldown)
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(worker, m): m["inst"] for m in pending}
            done = 0
            next_pending = []
            for fu in as_completed(futs):
                done += 1
                meta, r = fu.result()
                if isinstance(r, pd.DataFrame):
                    frames.append(r)
                    ok.append(meta | {"first_date": r["date"].iloc[0], "last_date": r["date"].iloc[-1], "rows": len(r)})
                else:
                    fail.append({"inst": meta["inst"], "err": r})
                    next_pending.append(meta)
                if done % 50 == 0:
                    log(f"  {market}: 本轮 {done}/{len(pending)} done, fail={len(next_pending)}")
            pending = next_pending

    # 用已有 raw 缓存补齐本轮失败但历史抓取成功的标的
    raw_dir = ROOT / "data" / "raw" / market
    failed_insts = {f["inst"] for f in fail}
    for f in list(fail):
        fp = raw_dir / f"{f['inst']}.csv"
        if fp.exists():
            try:
                old = pd.read_csv(fp)
                if len(old):
                    frames.append(old)
                    ok.append({"inst": f["inst"], "name": f["inst"], "first_date": old["date"].iloc[0],
                               "last_date": old["date"].max(), "rows": len(old), "from_cache": True})
                    fail.remove(f)
            except Exception:  # noqa: BLE001
                pass

    if frames:
        all_df = pd.concat(frames).drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
        out = RAW_DIR / market
        out.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(out / "all.csv", index=False)
    log(f"=== {market}: 成功 {len(ok)} / 失败 {len(fail)} ===")
    for f in fail:
        log(f"  {market} FAIL {f['inst']}: {f['err']}")
    return ok, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="cn,us,crypto,gold")
    args = ap.parse_args()

    uni_path = ROOT / "data" / "universe.json"
    meta_path = ROOT / "data" / "meta.json"
    uni_all = json.loads(uni_path.read_text()) if uni_path.exists() else {}
    meta_all = json.loads(meta_path.read_text()) if meta_path.exists() else {"fail": {}}

    for m in args.markets.split(","):
        ok, fail = run(m.strip())
        uni_all[m] = ok
        meta_all.setdefault("fail", {})[m] = fail

    uni_all = {k: v for k, v in sorted(uni_all.items())}
    uni_path.write_text(json.dumps(uni_all, ensure_ascii=False, indent=1))
    meta_all["fetched_at"] = dt.datetime.now().isoformat(timespec="seconds")
    meta_all["data_start"] = CFG["data_start"]
    meta_path.write_text(json.dumps(meta_all, ensure_ascii=False, indent=1))
    log("universe.json / meta.json 写入完成")


if __name__ == "__main__":
    main()
