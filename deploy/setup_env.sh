#!/usr/bin/env bash
# 部署机环境初始化(已在 win3060wsl/WSL2 + RTX3060 实测)
# 用法: 在目标机上 bash deploy/setup_env.sh
set -euo pipefail

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "未找到 miniconda3, 请先安装: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

conda env list | grep -q "^quant " || conda create -n quant python=3.10 -y
conda activate quant

# 国内网络使用清华 PyPI 镜像; 海外机器可去掉 -i
PIP="pip install -i https://pypi.tuna.tsinghua.edu.cn/simple"
$PIP --upgrade pip
$PIP "numpy==1.26.4" "pandas==2.2.3" pyqlib lightgbm akshare loguru matplotlib pyarrow scipy scikit-learn tqdm requests pyyaml fire
$PIP torch --no-cache-dir   # GPU 深度模型对照(CPU 环境可跳过)

python - << 'EOF'
import qlib, lightgbm, akshare, pandas
print("qlib", qlib.__version__, "| lgbm", lightgbm.__version__, "| akshare", akshare.__version__)
EOF
echo "环境就绪: conda activate quant && bash run_all.sh"
