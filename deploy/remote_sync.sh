#!/usr/bin/env bash
# 从 Mac 同步项目到 win3060wsl 并启动全流程(后台), 结果回收到本地
#   bash deploy/remote_sync.sh run     # 同步 + 远程跑 run_all.sh
#   bash deploy/remote_sync.sh status  # 查看远程进度
#   bash deploy/remote_sync.sh fetch   # 回收 results + docs 到本地
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${ALPHA_HOST:-win3060wsl}"
REMOTE_DIR="${REMOTE_DIR:-~/alpha2026}"

case "${1:-help}" in
run)
  rsync -az --exclude data --exclude results --exclude docs/data --exclude .git ./ "$HOST:$REMOTE_DIR/"
  ssh "$HOST" "nohup bash $REMOTE_DIR/run_all.sh > $REMOTE_DIR/run_all.log 2>&1 & echo LAUNCHED"
  ;;
status)
  ssh "$HOST" "tail -20 $REMOTE_DIR/run_all.log 2>/dev/null; pgrep -fa 03_run_experiments | head -2"
  ;;
fetch)
  rsync -az "$HOST:$REMOTE_DIR/results/" results/
  rsync -az "$HOST:$REMOTE_DIR/docs/" docs/
  rsync -az "$HOST:$REMOTE_DIR/data/universe.json" "$HOST:$REMOTE_DIR/data/meta.json" data/ 2>/dev/null || true
  echo "results/ 与 docs/ 已回收"
  ;;
*)
  echo "usage: run | status | fetch";;
esac
