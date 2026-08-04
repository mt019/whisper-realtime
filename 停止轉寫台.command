#!/bin/zsh
# 停止「啟動轉寫台」開起來的那一個進程。只殺 pid 檔裡記的那一個，不用 pattern 掃。

cd "$(dirname "$0")" || exit 1
PIDFILE=".streamlit/workbench.pid"

if [[ ! -f "$PIDFILE" ]]; then
  echo "沒有找到 $PIDFILE，表示不是用「啟動轉寫台」開的。"
  echo "自己在終端機開的那個，回到那個視窗按 Ctrl-C。"
  exit 1
fi

PID=$(cat "$PIDFILE")
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID" && echo "已停止（PID $PID）。"
else
  echo "PID $PID 已經不在了，可能早就關掉。"
fi
rm -f "$PIDFILE"
