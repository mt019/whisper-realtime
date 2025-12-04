# WhisperTC Workbench

WhisperTC Workbench 是一個針對繁體中文調校的 Whisper 轉寫工作台。你可以在一個 Streamlit 介面裡，同時完成模型設定、批次轉寫、後處理與字幕輸出，整個流程都在本機完成，不需要把錄音上傳到雲端。

這個專案最早是從 `fw_streamlit.py` 拆出來，一路邊實際上課/整理逐字稿邊改。現在的版本比較聚焦、好維護，也比較適合分享給有類似需求的人。

## 功能亮點
- 即時調整模型與推理參數：在側邊欄選擇模型大小、計算精度、束搜尋大小、VAD 靈敏度，邊聽邊調。
- 繁體中文最佳化：內建簡轉繁 (`OpenCC`)，搭配停頓式標點、自適應門檻，以及可選的「文本式標點融合」。
- 領域知識提示：可上傳 PDF/MD/TXT，萃取關鍵詞當作提示，用來拉高專業術語的命中率。
- 預設領域資料夾：可以在 Streamlit secrets 設定常用路徑，啟動時自動掛載（檔名含「稿」的會被自動略過）。
- 常見勘誤自動更正：轉寫結果會套用自訂錯別字對照表，減少你在 TXT/SRT 裡逐字修正的時間。
- 大檔案切段策略：可以選擇是否自動把長錄音切成數段並行處理；預設不切，優先降低潛在錯誤風險。
- 友善輸出：轉寫完成後可以下載 TXT / SRT，也可以在畫面上整理段落後再做最後調整。

## 目錄結構
```
whispertc-workbench/
├─ whispertc_workbench.py   # Streamlit 主程式
├─ requirements.txt         # 主要 Python 依賴
├─ README.md                # 快速開始說明
└─ to-do.md                 # 功能開發備忘
```

## 環境需求
- Python 3.10+
- macOS (Apple Silicon) 或具備 AVX 的 CPU 環境
- 建議安裝 `ffmpeg` 以支援更多音訊格式

### 建立虛擬環境與安裝依賴
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows 改用 .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 主要涵蓋：
- `streamlit`、`faster-whisper`、`opencc-python-reimplemented`
- `streamlit-webrtc`、`av`（音訊/視訊元件）
- `numpy`, `numba`, `llvmlite`

## 使用方式
```bash
streamlit run whispertc_workbench.py
```

啟動後瀏覽器會自動開啟 `http://localhost:8501`。上傳音訊、選好模型和參數，必要時加上領域提示，再按「開始轉寫」即可。

### 預設領域知識來源
- 所有預設路徑與勘誤項目集中在 `config/domain_defaults.json`。你可以直接改這個檔，也可以用環境變數 `WHISPERTC_DEFAULTS` 或 Streamlit `defaults_config_path` 指到你自己的設定。
- 在 `.streamlit/secrets.toml` 裡設定 `domain_kb_paths`，啟動時就會自動掛載對應資料夾。例如：

  ```toml
  domain_kb_paths = [
    "/Users/iw/Documents/NTU/1141/1141_Tax_Ko/_Material/法源",
    "/Users/iw/Documents/NTU/1141/1141_Tax_Ko/mkdocs/My_Notes"
  ]
  ```

- 若沒設定，程式會回退到上面示範中的預設路徑。系統會自動忽略檔名或路徑中含有「稿」的檔案，並只載入 `.pdf` / `.md` / `.txt`。
- `My_Notes` 目錄下的文件會被加一點權重，讓你平常筆記裡的用語比較容易被模型「看到」。
- 介面上你還是可以再拖曳其他檔案進來，預設與手動來源會一起變成領域提示。

### 常見勘誤列表
- 預設內建少量勘誤對照，定義在 `config/domain_defaults.json`（例如「稅券→稅捐」、「未接→位階」、「激增→稽徵」）。
- 你可以在 `.streamlit/secrets.toml` 用 `correction_paths` 指向額外的勘誤檔案，或在任一領域資料夾裡放一個 `common_corrections.txt`。
- 勘誤檔案格式：
  - JSON：以 `{ "正確字": ["常見錯字1", "常見錯字2"] }` 表示，多個錯誤可對應同一正確字。
  - 純文字：每行 `錯誤：正確`（可使用 `:`, `：`, `->`, `=>` 等分隔符），支援 `#` 或 `//` 作為註解行。
- 只要格式對，這些條目就會自動套用到最後輸出的 TXT / SRT。

### 快捷鍵建議
Streamlit 內建快捷鍵與互動設計可搭配使用：
- `s`：聚焦 Sidebar，搭配 `Enter` 可快速切換模型或參數。
- `r`：重新執行程式，清空狀態。
- `Shift+Enter`：在文字輸入框送出內容。
- `Ctrl+K` / `Cmd+K`：開啟指令面板，快速重啟或跳轉元件。
- `?`：顯示 Streamlit 官方快捷鍵總覽。

將「開始轉寫」按鈕放置於側邊欄頂部，可透過 `Tab` / `Shift+Tab` 切換並以 `Space` 觸發，形成全鍵盤操作流程。

## 推薦工作流程
1. 啟動 app 後先設定模型、精度與 VAD 參數。
2. 若有領域知識檔，於側邊欄上傳並檢視預覽詞彙。
3. 上傳音訊檔或輸入初始提示，按下「開始轉寫」。
4. 觀察進度列與 ETA，完成後檢視整合結果與 SRT。
5. 下載 TXT / SRT，必要時於外部工具做細部編修。

## GitHub 發佈建議
1. `git init`
2. `git add .`
3. `git commit -m "Initial commit for WhisperTC Workbench"`
4. `git branch -M main`
5. `git remote add origin git@github.com:<your-account>/<repo-name>.git`
6. `git push -u origin main`

> 請先於 GitHub 建立新的空白 repository，名稱可與專案一致（如 `whispertc-workbench`）。

## 後續規劃
- 繼續於 `to-do.md` 跟進需求與優化項目。
- 若未來需擴增 API 或 Docker 部署方案，可於新分支維護，保持 Streamlit 版本純粹。
