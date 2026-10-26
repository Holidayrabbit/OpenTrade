#!/usr/bin/env python3
"""alpha2026 · Step 2: 将 CSV 落地为 Qlib 二进制数据(每市场独立数据目录/交易日历)

调用官方 scripts/dump_bin.py(vendored), 产出 data/qlib/{market}/{calendars,instruments,features}
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DUMP = ROOT / "scripts" / "vendor" / "dump_bin.py"


def dump_market(market: str):
    csv_dir = ROOT / "data" / "qlib_csv" / market
    assert (csv_dir / "all.csv").exists(), f"缺少 {csv_dir}/all.csv, 请先运行 01_fetch_data.py"

    # dump_bin 约定: 一个 csv 文件 = 一个标的(文件名即代码), 故先拆分
    import pandas as pd

    split_dir = csv_dir / "split"
    split_dir.mkdir(exist_ok=True)
    df = pd.read_csv(csv_dir / "all.csv", dtype={"symbol": str})
    for sym, g in df.groupby("symbol"):
        g.drop(columns=["symbol"]).to_csv(split_dir / f"{sym.lower()}.csv", index=False)

    qlib_dir = ROOT / "data" / "qlib" / market
    if qlib_dir.exists():
        import shutil

        shutil.rmtree(qlib_dir)
    cmd = [
        sys.executable, str(DUMP), "dump_all",
        "--data_path", str(split_dir),
        "--qlib_dir", str(qlib_dir),
        "--include_fields", "open,high,low,close,volume,amount",
        "--date_field_name", "date",
        "--max_workers", "16",
    ]
    print(f"[dump] {market}: {len(df.groupby('symbol'))} instruments", flush=True)
    subprocess.run(cmd, check=True)

    # 校验
    import warnings

    warnings.filterwarnings("ignore")
    import qlib  # noqa: E402
    from qlib.data import D  # noqa: E402

    qlib.init(provider_uri=str(qlib_dir), region="cn")
    cal = D.calendar(freq="day")
    insts = D.list_instruments(D.instruments("all"), as_list=True)
    df = D.features(insts, ["$close", "$volume"], start_time=cal[-5], end_time=cal[-1])
    n_inst = df.index.get_level_values("instrument").nunique()
    # 完整性硬校验: 落地标的数必须 >= universe 的 90%, 防止残缺宇宙悄悄进入回测
    uni = json.loads((ROOT / "data" / "universe.json").read_text()).get(market) or []
    uni_n = len({u["inst"] for u in uni})
    print(f"[check] {market}: {len(cal)} 个交易日({str(cal[0])[:10]} ~ {str(cal[-1])[:10]}), "
          f"标的 {len(insts)} / universe {uni_n}, 近5日有行情 {n_inst}", flush=True)
    assert uni_n == 0 or len(insts) >= uni_n * 0.9, f"{market} 标的覆盖不足({len(insts)}/{uni_n}), 拒绝落地"
    assert len(insts) > 10 and n_inst > 10, "标的覆盖异常"
    return True


if __name__ == "__main__":
    markets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["cn", "us", "crypto", "gold"]
    for m in markets:
        dump_market(m)
    print("dump done")
