# Engineering Log

## 2026-04-21 - whisper.cpp / Apple Silicon Optimization

### Context
- Target machine: MacBook Air M3, 16 GB RAM.
- Problem: `faster-whisper` with `medium` on long lecture recordings caused high CPU load, memory compression, swap pressure, and slow end-to-end transcription.
- Operational constraint: long recordings should usually run as a single file because naive time-based chunking can cut across context and trigger repeated hallucinated output.

### ASR Backend Changes
- Added selectable STT backend:
  - `faster-whisper` remains available as a fallback.
  - `whisper.cpp` is now the preferred local backend when available.
- Installed and built official `whisper.cpp` under:
  - CLI: `/Users/iw/Documents/whisper.cpp/build/bin/whisper-cli`
  - Full model: `/Users/iw/Documents/whisper.cpp/models/ggml-medium.bin`
  - Quantized model: `/Users/iw/Documents/whisper.cpp/models/ggml-medium-q5_0.bin`
- Verified the build uses Apple Silicon acceleration:
  - `use gpu = 1`
  - `flash attn = 1`
  - `using MTL0 backend`
  - `found device: Apple M3`
- The app auto-detects common `whisper.cpp` paths and prefers `ggml-medium-q5_0.bin`.
- Added an optimal-configuration guard for `whisper.cpp`: backend must be `whisper.cpp`, model should be medium/q5-style preferred model, and `beam=1`.

### Runtime Defaults
- Default model for the legacy `faster-whisper` path was lowered from `medium` to `small`.
- Default beam size was lowered from `2` to `1`.
- Temporary WAV export now uses `16 kHz`, mono, 16-bit WAV instead of larger 44.1 kHz WAV.
- Long recording auto-chunking remains off by default.

### Progress And Live Display
- `whisper.cpp` CLI output is streamed with `subprocess.Popen`.
- Timestamped CLI lines such as `[00:00:00.000 --> 00:00:11.000] text` are parsed live.
- Live text display now updates segment-by-segment while `whisper.cpp` is running.
- Progress bar updates are based on the latest segment end timestamp:
  - processed audio duration
  - total audio duration
  - elapsed time
  - estimated remaining time
  - estimated finish time
- Added startup progress states for model loading, transcription start, and waiting for the first segment.

### UI Changes
- Removed unused empty column layout that could create blank vertical space.
- Removed the `beforeunload` iframe injection because it created a large blank region in the Streamlit page.
- Moved rarely changed ASR internals into an advanced settings section so the normal transcription flow is not cluttered by backend, model path, precision, thread, VAD, punctuation threshold, or chunking controls.
- Simplified single-file mode wording:
  - uses "整段轉寫" instead of "分段 1"
  - uses "開始轉換時間點" instead of "分段 1 開始轉換時間點"
- Progress bar styling changed to a bright pink UI:
  - fill: `#ff8fb3`
  - track: `#ffe8ef`
  - text: `#8f4d63`

### Punctuation Logic
- Added punctuation helpers:
  - `has_any_punctuation`
  - `has_terminal_punctuation`
- `SmartPunctuator` now avoids inserting punctuation after text that already ends with punctuation.
- For `whisper.cpp` output, if source segments already contain punctuation, final post-processing preserves it instead of re-punctuating by pause rules.
- Final sentence-ending fallback only applies when the final text lacks terminal punctuation.

### Known Tradeoffs
- `whisper.cpp` CLI live output is segment-level, not word-level.
- `faster-whisper` still has a more convenient Python generator API for fine-grained streaming and word timestamps.
- `beforeunload` protection was removed to avoid layout breakage; if needed, it should be reintroduced without a visible/space-occupying Streamlit component.

### Verification
- `python3 -m py_compile whispertc_workbench.py` passed after the changes.
- `whisper.cpp` sample run confirmed Metal backend on Apple M3 with `ggml-medium-q5_0.bin`.
