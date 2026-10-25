#!/usr/bin/env bash
# alpha2026 · 一键全流程(在部署机上执行, 建议 nohup)
#   nohup bash run_all.sh > run_all.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
export MLFLOW_ALLOW_FILE_STORE=true

# 默认 miniconda 环境(部署机), 可用 ALPHA_ENV 覆盖
ENV_NAME="${ALPHA_ENV:-quant}"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
fi
echo "python: $(which python)"

step1_skip=0
if [ -f data/qlib_csv/gold/all.csv ] && [ -f data/qlib_csv/cn/all.csv ] && [ -f data/qlib_csv/us/all.csv ] && [ -f data/qlib_csv/crypto/all.csv ]; then
  echo "[skip] 数据已存在, 跳过抓取(删除 data/qlib_csv 可强制重抓)"
  step1_skip=1
fi
if [ "$step1_skip" = "0" ]; then
  python scripts/01_fetch_data.py --markets cn,us,crypto,gold
fi

python scripts/02_dump_qlib.py cn,us,crypto,gold

# 四市场 × buyhold/static/monthly/weekly(LightGBM)
for m in gold crypto cn us; do
  python scripts/03_run_experiments.py --market "$m" --model lgbm --variants buyhold,static,monthly,weekly
done

# GRU 深度模型对照(GPU, A股静态)
python scripts/03_run_experiments.py --market cn --model gru --variants static || echo "[warn] GRU 失败, 不影响主线"

python scripts/04_build_site.py
echo "ALL_DONE $(date)"
