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

# 逐市场: 抓取(增量缓存) + qlib 落地, 失败冷却后重试最多 3 次
# (02 的标的数校验保证不会用残缺数据出结果)
for m in gold crypto cn us; do
  ok=0
  for attempt in 1 2 3; do
    if python scripts/01_fetch_data.py --markets "$m" && python scripts/02_dump_qlib.py "$m"; then
      ok=1; break
    fi
    echo "[retry] $m 第${attempt}次失败, 冷却 120s ..."
    sleep 120
  done
  if [ "$ok" != "1" ]; then
    echo "[FATAL] 市场 $m 数据无法就绪, 终止(已产出的其他市场结果保留)"
    exit 1
  fi
done

# 四市场 × buyhold/static/monthly/weekly(LightGBM)
for m in gold crypto cn us; do
  python scripts/03_run_experiments.py --market "$m" --model lgbm --variants buyhold,static,monthly,weekly
done

# GRU 深度模型对照(GPU, A股静态)
python scripts/03_run_experiments.py --market cn --model gru --variants static || echo "[warn] GRU 失败, 不影响主线"

python scripts/04_build_site.py
echo "ALL_DONE $(date)"
