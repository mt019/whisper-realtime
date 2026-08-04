#!/bin/zsh
# 雙擊即用。已經在跑就直接開瀏覽器，不會重複啟動。
# 停止：關掉這個終端機視窗，或執行同目錄的「停止轉寫台.command」。

cd "$(dirname "$0")" || exit 1
PORT=8501
URL="http://localhost:$PORT"
PIDFILE=".streamlit/workbench.pid"

open_browser() {
  [[ -n "$NO_OPEN" ]] || open "$URL"
}

# 已經有一個在跑就重用它
if curl -s -o /dev/null --max-time 2 "$URL"; then
  echo "轉寫台已經在 $URL 跑著，直接開瀏覽器。"
  open_browser
  exit 0
fi

if [[ ! -x .venv311/bin/streamlit ]]; then
  echo "找不到 .venv311/bin/streamlit。先建虛擬環境："
  echo "  python3 -m venv .venv311 && .venv311/bin/pip install -r requirements.txt"
  exit 1
fi

echo "啟動轉寫台…（模型 large-v3-turbo-q5_0，VAD 已接上）"
mkdir -p .streamlit
.venv311/bin/streamlit run whispertc_workbench.py \
  --server.port "$PORT" --server.headless true --browser.gatherUsageStats false &
APP_PID=$!
echo "$APP_PID" > "$PIDFILE"

# 等它真的起來再開瀏覽器，最多等 60 秒
for i in {1..60}; do
  if curl -s -o /dev/null --max-time 2 "$URL"; then
    echo "好了：$URL"
    open_browser
    break
  fi
  sleep 1
done

echo
echo "關掉這個視窗就會停止轉寫台。"
wait "$APP_PID"
