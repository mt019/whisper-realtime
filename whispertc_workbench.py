# fw_streamlit.py
import streamlit as st
from faster_whisper import WhisperModel
from opencc import OpenCC
from pydub import AudioSegment
import io, os, time, wave, math, json, hashlib, subprocess, shutil
import re
from typing import List, Tuple, Dict, Any, Pattern
from dataclasses import dataclass
import platform

from statistics import median
# --- multiprocessing context and traceback
from multiprocessing import get_context

from pathlib import Path

# Detect worker mode for child processes
IS_WORKER = os.environ.get("ASR_WORKER") == "1"

# --- Domain knowledge defaults ---
CONFIG_DEFAULT_PATH = Path(__file__).resolve().parent / "config" / "domain_defaults.json"


@st.cache_data(show_spinner=False)
def _load_defaults_config() -> Dict[str, Any]:
    candidates: List[Path] = []
    try:
        cfg_path = st.secrets.get("defaults_config_path")
        if cfg_path:
            if isinstance(cfg_path, (list, tuple)):
                for p in cfg_path:
                    candidates.append(Path(str(p)).expanduser())
            else:
                candidates.append(Path(str(cfg_path)).expanduser())
    except Exception:
        pass
    env_path = os.environ.get("WHISPERTC_DEFAULTS")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(CONFIG_DEFAULT_PATH)

    for cand in candidates:
        try:
            if not cand.exists():
                continue
            data = json.loads(cand.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            continue
    return {}


DEFAULTS_CONFIG = _load_defaults_config()

WHISPERCPP_DEFAULT_CLI = str(DEFAULTS_CONFIG.get("whispercpp_cli", "") or os.environ.get("WHISPERCPP_CLI", ""))
WHISPERCPP_DEFAULT_MODEL = str(DEFAULTS_CONFIG.get("whispercpp_model", "") or os.environ.get("WHISPERCPP_MODEL", ""))
WHISPERCPP_COMMON_DIRS = [
    Path.home() / "Documents" / "whisper.cpp",
    Path.home() / "whisper.cpp",
    Path("/opt/homebrew/opt/whisper-cpp"),
]
WHISPERCPP_OPTIMAL_MODEL_NAMES = (
    "ggml-medium-q5_0.bin",
    "ggml-medium-q8_0.bin",
    "ggml-medium.bin",
    "ggml-small-q8_0.bin",
    "ggml-small.bin",
)

DOMAIN_SECRET_KEY = DEFAULTS_CONFIG.get("domain_secret_key", "domain_kb_paths")
DEFAULT_DOMAIN_PATHS = [str(Path(p).expanduser()) for p in DEFAULTS_CONFIG.get("domain_paths", [])]
DOMAIN_ALLOWED_SUFFIXES = set(DEFAULTS_CONFIG.get("allowed_suffixes", [".pdf", ".md", ".txt"]))
DOMAIN_IGNORE_KEYWORD = DEFAULTS_CONFIG.get("ignore_keyword", "稿")
DOMAIN_PRIORITY_HINTS = [str(h) for h in DEFAULTS_CONFIG.get("priority_hints", [])]

CORRECTION_SECRET_KEY = DEFAULTS_CONFIG.get("correction_secret_key", "correction_paths")
DEFAULT_CORRECTION_FILES: List[str] = [str(Path(p).expanduser()) for p in DEFAULTS_CONFIG.get("correction_files", [])]


def _extract_correction_map(cfg: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    # 1) support new correction_map format: {"correct": ["wrong1", ...]}
    corr_map = cfg.get("correction_map", {})
    if isinstance(corr_map, dict):
        for correct, wrongs in corr_map.items():
            if not correct:
                continue
            if isinstance(wrongs, (list, tuple, set)):
                for wrong in wrongs:
                    wrong = (wrong or "").strip()
                    if wrong:
                        mapping[wrong] = str(correct).strip()
            elif wrongs:
                mapping[str(wrongs).strip()] = str(correct).strip()
    # 2) backward compatibility for correction_pairs [[wrong, correct], ...]
    corr_pairs = cfg.get("correction_pairs", [])
    if isinstance(corr_pairs, (list, tuple)):
        for pair in corr_pairs:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                wrong, correct = pair
                wrong = (wrong or "").strip()
                correct = (correct or "").strip()
                if wrong and correct:
                    mapping[wrong] = correct
    return mapping


def _extract_regex_correction_map(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    mapping: List[Tuple[str, str]] = []
    raw_map = cfg.get("regex_correction_map", {})
    if isinstance(raw_map, dict):
        # Prefer consistent schema with correction_map: {"correct": ["regex1", ...]}
        if all(isinstance(v, (list, tuple, set)) for v in raw_map.values()):
            for correct, patterns in raw_map.items():
                replacement = (str(correct) if correct is not None else "").strip()
                if not replacement:
                    continue
                for pattern in patterns:
                    p = (str(pattern) if pattern is not None else "").strip()
                    if p:
                        mapping.append((p, replacement))
        else:
            # Backward compatibility: {"pattern": "replacement"}
            for pattern, replacement in raw_map.items():
                p = (str(pattern) if pattern is not None else "").strip()
                r = (str(replacement) if replacement is not None else "").strip()
                if p and r:
                    mapping.append((p, r))
    return mapping


DEFAULT_CORRECTION_MAP = _extract_correction_map(DEFAULTS_CONFIG)
DEFAULT_REGEX_CORRECTION_MAP = _extract_regex_correction_map(DEFAULTS_CONFIG)


@dataclass(frozen=True)
class PunctuationSettings:
    """Operator-tunable punctuation thresholds stored in milliseconds."""
    comma_ms: int
    period_ms: int
    adaptive: bool = True

    @property
    def comma_threshold(self) -> float:
        return max(0.01, self.comma_ms / 1000.0)

    @property
    def period_threshold(self) -> float:
        return max(0.01, self.period_ms / 1000.0)


def _load_default_punctuation_settings(cfg: Dict[str, Any]) -> PunctuationSettings:
    """Load punctuation thresholds from configuration with safe fallbacks."""
    base = cfg.get("punctuation_defaults", {}) if isinstance(cfg, dict) else {}

    def _int_field(key: str, default: int) -> int:
        try:
            return max(10, int(base.get(key, default)))
        except Exception:
            return default

    comma_ms = _int_field("comma_ms", 150)
    period_ms = _int_field("period_ms", 350)
    adaptive = bool(base.get("adaptive", True))
    return PunctuationSettings(comma_ms=comma_ms, period_ms=period_ms, adaptive=adaptive)


DEFAULT_PUNCTUATION_SETTINGS = _load_default_punctuation_settings(DEFAULTS_CONFIG)
CURRENT_PUNCT_SETTINGS: PunctuationSettings = DEFAULT_PUNCTUATION_SETTINGS


class StaticDomainFile:
    """Wrap local bytes payload to mimic Streamlit UploadedFile interface."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def read(self) -> bytes:
        return self._data

    def seek(self, pos: int):
        # For API compatibility with UploadedFile
        return None


def _coerce_secret_paths(raw) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple, set)):
        return [str(x) for x in raw if x]
    return []


@st.cache_data(show_spinner=False)
def _load_domain_bytes_from_paths(paths: Tuple[str, ...]) -> List[Tuple[str, bytes]]:
    collected: List[Tuple[str, bytes]] = []
    seen_digests: set[Tuple[str, str]] = set()
    for base in paths:
        base_path = Path(base).expanduser()
        if not base_path.exists():
            continue
        for file_path in base_path.rglob("*"):
            if not file_path.is_file():
                continue
            if DOMAIN_IGNORE_KEYWORD and DOMAIN_IGNORE_KEYWORD in file_path.as_posix():
                continue
            if file_path.suffix.lower() not in DOMAIN_ALLOWED_SUFFIXES:
                continue
            try:
                data = file_path.read_bytes()
                digest = hashlib.md5(data).hexdigest()
                # Avoid re-processing identical files that share the same name and content
                sig = (file_path.name.lower(), digest)
                if sig in seen_digests:
                    continue
                seen_digests.add(sig)
                collected.append((str(file_path), data))
            except Exception:
                continue
    return collected


def _resolve_default_domain_files() -> Tuple[List[StaticDomainFile], List[str]]:
    try:
        raw_paths = st.secrets.get(DOMAIN_SECRET_KEY)
    except Exception:
        raw_paths = None
    candidate_paths = _coerce_secret_paths(raw_paths)
    if not candidate_paths:
        candidate_paths = DEFAULT_DOMAIN_PATHS
    candidate_paths = [str(Path(p).expanduser()) for p in candidate_paths if p]
    if not candidate_paths:
        return [], []
    payload = _load_domain_bytes_from_paths(tuple(candidate_paths))
    files = [StaticDomainFile(name=path, data=data) for path, data in payload]
    return files, candidate_paths


preset_domain_files: List[StaticDomainFile] = []
preset_domain_paths: List[str] = []
combined_domain_files: List = []
CORRECTION_MAP: Dict[str, str] = {}
REGEX_CORRECTION_RULES: List[Tuple[Pattern[str], str]] = []


def _detect_whispercpp_cli() -> str:
    for name in ("whisper-cli", "main"):
        found = shutil.which(name)
        if found:
            return found
    for base in WHISPERCPP_COMMON_DIRS:
        for rel in ("build/bin/whisper-cli", "build/bin/main", "main"):
            cand = base / rel
            if cand.exists():
                return str(cand)
    return ""


def _detect_whispercpp_model() -> str:
    for base in WHISPERCPP_COMMON_DIRS:
        model_dir = base / "models"
        for name in WHISPERCPP_OPTIMAL_MODEL_NAMES:
            cand = model_dir / name
            if cand.exists():
                return str(cand)
        if model_dir.exists():
            candidates = sorted(
                list(model_dir.glob("ggml-*.bin")) + list(model_dir.glob("*.gguf")),
                key=lambda p: p.stat().st_size if p.exists() else 0,
            )
            if candidates:
                return str(candidates[0])
    return ""


def _is_whispercpp_optimal(cli_path: str, model_path: str, backend: str, beam: int) -> bool:
    if backend != "whisper.cpp":
        return False
    if not cli_path or not Path(str(cli_path)).expanduser().exists():
        return False
    if not model_path or not Path(str(model_path)).expanduser().exists():
        return False
    model_name = Path(str(model_path)).name
    return model_name in WHISPERCPP_OPTIMAL_MODEL_NAMES and int(beam or 0) == 1


def _normalize_paths(paths: List[str]) -> Tuple[str, ...]:
    normalized = []
    for p in paths:
        if not p:
            continue
        normalized.append(str(Path(p).expanduser()))
    # 去重並保持順序
    seen = set()
    uniq = []
    for p in normalized:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return tuple(uniq)


@st.cache_data(show_spinner=False)
def _load_corrections_from_paths(
    paths: Tuple[str, ...],
    base_mapping: Dict[str, str],
    base_regex_mapping: Tuple[Tuple[str, str], ...],
) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    corrections: Dict[str, str] = dict(base_mapping)
    regex_corrections: List[Tuple[str, str]] = list(base_regex_mapping)
    for p in paths:
        file_path = Path(p)
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            if file_path.suffix.lower() == ".json":
                data = json.loads(file_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    raw_regex_map = data.get("regex_correction_map")
                    if isinstance(raw_regex_map, dict):
                        if all(isinstance(v, (list, tuple, set)) for v in raw_regex_map.values()):
                            for correct, patterns in raw_regex_map.items():
                                replacement = (str(correct) if correct is not None else "").strip()
                                if not replacement:
                                    continue
                                for pattern in patterns:
                                    pat = (str(pattern) if pattern is not None else "").strip()
                                    if pat:
                                        regex_corrections.append((pat, replacement))
                        else:
                            for pattern, replacement in raw_regex_map.items():
                                pat = (str(pattern) if pattern is not None else "").strip()
                                rep = (str(replacement) if replacement is not None else "").strip()
                                if pat and rep:
                                    regex_corrections.append((pat, rep))
                    if "correction_map" in data and isinstance(data.get("correction_map"), dict):
                        data = data.get("correction_map", {})
                    # Determine format: correct -> [wrongs] or wrong -> correct
                    if all(isinstance(v, (list, tuple, set)) for v in data.values()):
                        for correct, wrongs in data.items():
                            for wrong in wrongs:
                                w = (str(wrong) if wrong is not None else "").strip()
                                if w:
                                    corrections[w] = str(correct).strip()
                    else:
                        for wrong, correct in data.items():
                            w = (str(wrong) if wrong is not None else "").strip()
                            c = (str(correct) if correct is not None else "").strip()
                            if w and c:
                                corrections[w] = c
                elif isinstance(data, list):
                    for entry in data:
                        if isinstance(entry, dict):
                            wrong = (str(entry.get("wrong", "")).strip())
                            correct = (str(entry.get("correct", "")).strip())
                            if wrong and correct:
                                corrections[wrong] = correct
                            # support mapping style
                            for key, value in entry.items():
                                if key == "correct":
                                    continue
                                if key == "wrong":
                                    continue
                                if isinstance(value, (list, tuple, set)):
                                    for wrong in value:
                                        w = (str(wrong).strip())
                                        if w and entry.get("correct"):
                                            corrections[w] = str(entry.get("correct")).strip()
                        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                            wrong, correct = entry
                            w = (str(wrong).strip())
                            c = (str(correct).strip())
                            if w and c:
                                corrections[w] = c
                continue
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            separator = None
            for cand in ("：", ":", "->", "=>", ",", "，"):
                if cand in line:
                    separator = cand
                    break
            if not separator:
                continue
            parts = line.split(separator, 1)
            if len(parts) != 2:
                continue
            wrong, right = parts[0].strip(), parts[1].strip()
            if wrong and right:
                corrections[wrong] = right
    return corrections, regex_corrections


def _compile_regex_correction_rules(rules: List[Tuple[str, str]]) -> List[Tuple[Pattern[str], str]]:
    compiled: List[Tuple[Pattern[str], str]] = []
    for pattern, replacement in rules:
        try:
            compiled.append((re.compile(pattern), replacement))
        except re.error:
            continue
    return compiled


def _resolve_corrections(domain_dirs: Tuple[str, ...]) -> Tuple[Dict[str, str], List[Tuple[Pattern[str], str]]]:
    try:
        raw_paths = st.secrets.get(CORRECTION_SECRET_KEY)
    except Exception:
        raw_paths = None
    candidate_paths = _coerce_secret_paths(raw_paths)
    # fallback to hardcoded + domain-local suggestion file
    fallback = list(DEFAULT_CORRECTION_FILES)
    for d in domain_dirs:
        fallback.append(str(Path(d) / "common_corrections.txt"))
    candidate_paths.extend(fallback)
    normalized = _normalize_paths(candidate_paths)
    corrections, regex_corrections = _load_corrections_from_paths(
        normalized,
        DEFAULT_CORRECTION_MAP,
        tuple(DEFAULT_REGEX_CORRECTION_MAP),
    )
    return corrections, _compile_regex_correction_rules(regex_corrections)


def apply_common_corrections(text: str) -> str:
    if not text or (not CORRECTION_MAP and not REGEX_CORRECTION_RULES):
        return text
    output = text
    for wrong, right in CORRECTION_MAP.items():
        output = output.replace(wrong, right)
    for pattern, replacement in REGEX_CORRECTION_RULES:
        output = pattern.sub(replacement, output)
    return output


def _trigger_rerun():
    """Trigger a Streamlit rerun, compatible across versions."""
    rerun_fn = getattr(st, "experimental_rerun", None)
    if rerun_fn:
        rerun_fn()
        return
    rerun_fn = getattr(st, "rerun", None)
    if rerun_fn:
        rerun_fn()
        return
    try:
        from streamlit.runtime.scriptrunner import RerunException, RerunData
    except Exception as exc:
        raise RuntimeError("Streamlit rerun is not available in this environment.") from exc
    raise RerunException(RerunData())


def _on_start_transcribe():
    """Mark transcription as started and queue the main workflow."""
    if st.session_state.get("transcribing"):
        return
    st.session_state["transcribe_success_message"] = ""
    st.session_state["transcribing"] = True
    st.session_state["start_transcribe_pending"] = True
    st.session_state["transcribe_status_message"] = "🚀 已收到轉錄請求，正在初始化…"


if not IS_WORKER:
    st.set_page_config(page_title="稅法 STT", layout="wide")
    st.title("momo STT｜語音轉文字")

if not IS_WORKER:
    preset_domain_files, preset_domain_paths = _resolve_default_domain_files()
    CORRECTION_MAP, REGEX_CORRECTION_RULES = _resolve_corrections(tuple(preset_domain_paths))
    sidebar = st.sidebar
    sidebar.markdown(
        """
        <style>
        [data-testid="stSidebar"] .run-btn-wrapper button {
            font-size: 1.4rem;
            font-weight: 600;
            padding: 1.4rem 0.5rem;
        }
        [data-testid="stSidebar"] .run-btn-wrapper {
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
        </style>
        <style>
        [data-testid="stFileUploaderDropzone"][aria-label*="上傳音檔"] {
            min-height: 30rem !important;  /* default is ~6rem, double it */
        }
        [data-testid="stFileUploaderDropzone"][aria-label*="上傳音檔"] section {
            padding: 2rem !important;      /* increase padding inside dropzone */
        }
        div[data-testid="stProgress"] {
            padding-top: 0.15rem;
            padding-bottom: 0.2rem;
        }
        div[data-testid="stProgress"] > div {
            height: 0.72rem !important;
            background-color: #ffe8ef !important;
            border-radius: 999px !important;
            overflow: hidden !important;
        }
        div[data-testid="stProgress"] > div > div {
            height: 0.72rem !important;
            background-color: #ff8fb3 !important;
            border-radius: 999px !important;
        }
        div[data-testid="stProgress"] > div > div > div {
            background-color: #ff8fb3 !important;
            border-radius: 999px !important;
        }
        div[data-testid="stProgress"] p {
            color: #8f4d63 !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with sidebar.container():
        sidebar.markdown('<div class="run-btn-wrapper">', unsafe_allow_html=True)
        if "transcribing" not in st.session_state:
            st.session_state["transcribing"] = False
        if "start_transcribe_pending" not in st.session_state:
            st.session_state["start_transcribe_pending"] = False
        if "transcribe_status_message" not in st.session_state:
            st.session_state["transcribe_status_message"] = ""
        if "transcribe_success_message" not in st.session_state:
            st.session_state["transcribe_success_message"] = ""
        sidebar.button(
            "開始轉寫",
            use_container_width=True,
            key="run_button_primary",
            on_click=_on_start_transcribe,
            disabled=st.session_state["transcribing"],
        )
        sidebar.markdown('</div>', unsafe_allow_html=True)






    sidebar.markdown("---")
    detected_whispercpp_cli = _detect_whispercpp_cli()
    detected_whispercpp_model = _detect_whispercpp_model()
    default_backend_idx = 1 if (WHISPERCPP_DEFAULT_CLI or detected_whispercpp_cli) else 0
    asr_backend_options = ["faster-whisper", "whisper.cpp"]
    asr_backend = asr_backend_options[default_backend_idx]
    model_name = "small"
    compute = "int8"
    beam_size = 1
    whispercpp_cli = WHISPERCPP_DEFAULT_CLI or detected_whispercpp_cli
    whispercpp_model = WHISPERCPP_DEFAULT_MODEL or detected_whispercpp_model
    whispercpp_threads = max(1, min(6, (os.cpu_count() or 4) - 2))
    vad = True
    vad_threshold = 0.50
    vad_min_silence = 250
    vad_min_speech = 150
    vad_pad = 200
    punc_rule = True
    default_punc = DEFAULT_PUNCTUATION_SETTINGS
    comma_ms = default_punc.comma_ms
    period_ms = default_punc.period_ms
    adaptive_check = default_punc.adaptive
    reflow_enable = False
    reflow_width = 60
    auto_chunk_audio = False
    with sidebar.expander("進階設定", expanded=False):
        asr_backend = st.selectbox("STT 後端", asr_backend_options, index=default_backend_idx)
        if asr_backend == "faster-whisper":
            model_name = st.selectbox("模型", ["base","small","medium","distil-large-v3","large-v3"], index=1)
            compute = st.selectbox("精度", ["int8","float16","float32"], index=0)
        beam_size = st.selectbox("束搜尋大小", [1,2,4,5,8], index=0)
        if asr_backend == "whisper.cpp":
            whispercpp_cli = st.text_input("CLI 路徑", whispercpp_cli)
            whispercpp_model = st.text_input("模型檔路徑", whispercpp_model)
            whispercpp_threads = st.slider("CPU threads", 1, max(1, os.cpu_count() or 8), min(whispercpp_threads, max(1, os.cpu_count() or 8)), 1)
            st.caption("本機最佳性能：使用 Metal 編譯的 whisper.cpp；M3 Air 16GB 建議 medium-q5_0，beam=1。")
            if _is_whispercpp_optimal(whispercpp_cli, whispercpp_model, asr_backend, beam_size):
                st.success("最佳配置已啟用")
            else:
                st.warning("目前不是最佳配置；請使用 whisper.cpp、medium-q5_0、beam=1。")
        vad = st.checkbox("啟用 VAD 過濾", vad)
        with st.expander("VAD 靈敏度", expanded=False):
            vad_threshold = st.slider("threshold", 0.05, 0.95, vad_threshold, 0.01)
            vad_min_silence = st.slider("min_silence_duration_ms", 50, 1200, vad_min_silence, 10)
            vad_min_speech = st.slider("min_speech_duration_ms", 50, 800, vad_min_speech, 10)
            vad_pad = st.slider("speech_pad_ms", 0, 500, vad_pad, 10)
        punc_rule = st.checkbox("依停頓自動補標點", punc_rule)
        with st.expander("標點靈敏度", expanded=False):
            comma_ms = st.slider("逗號門檻 (ms)", 80, 400, comma_ms, 10)
            period_ms = st.slider("句號門檻 (ms)", 120, 800, period_ms, 10)
            adaptive_check = st.checkbox("自適應門檻（依語速）", adaptive_check)
        with st.expander("段落輸出", expanded=False):
            reflow_enable = st.checkbox("依語境分段顯示", reflow_enable)
            reflow_width = st.slider("每段最長字數", 20, 120, reflow_width, 5)
        auto_chunk_audio = st.checkbox("長錄音自動分段轉寫", auto_chunk_audio)
    CURRENT_PUNCT_SETTINGS = PunctuationSettings(
        comma_ms=int(comma_ms),
        period_ms=int(period_ms),
        adaptive=bool(adaptive_check),
    )
    st.session_state["punctuation_settings"] = CURRENT_PUNCT_SETTINGS

    sidebar.markdown("---")
    sidebar.caption("可選：載入 PDF/MD/TXT 作為領域詞彙，增強辨識")
    if preset_domain_files:
        sidebar.caption(f"預設已掛載 {len(preset_domain_files)} 份領域文本（{len(preset_domain_paths)} 個資料夾）")
        with sidebar.expander("自動載入來源", expanded=False):
            st.write("目錄：")
            for p in preset_domain_paths:
                st.code(p)
            sample_names = [Path(f.name).name for f in preset_domain_files[:5]]
            if sample_names:
                st.write("示例檔案：")
                st.code("\n".join(sample_names))
    domain_files = sidebar.file_uploader("領域知識檔 (PDF/MD/TXT)", type=["pdf","md","txt"], accept_multiple_files=True)
    terms_topk = sidebar.slider("最多帶入詞彙數", 10, 2000, 1500, 10)
    auto_use_domain = sidebar.checkbox("在轉寫時自動帶入領域詞彙", True)

    init_prompt = sidebar.text_input("初始提示（可留空）", "請使用繁體中文輸出。")
    combined_domain_files = list(preset_domain_files)
    if domain_files:
        combined_domain_files.extend(domain_files)

cc = OpenCC("s2t")

# The following parameters must be available in both UI and worker mode
if not IS_WORKER:
    vad_params = {
        "threshold": vad_threshold,
        "min_silence_duration_ms": vad_min_silence,
        "min_speech_duration_ms": vad_min_speech,
        "speech_pad_ms": vad_pad,
    }
else:
    # Dummy values for workers (actual values passed as arguments)
    vad_params = {}
    CORRECTION_MAP = dict(DEFAULT_CORRECTION_MAP)
    REGEX_CORRECTION_RULES = _compile_regex_correction_rules(DEFAULT_REGEX_CORRECTION_MAP)
    CURRENT_PUNCT_SETTINGS = DEFAULT_PUNCTUATION_SETTINGS



@st.cache_resource(show_spinner=False)
def load_model(name, compute_type):
    # 嘗試使用 Metal，不支援時自動回退到 CPU，並記錄實際裝置
    for device in ("metal", "cpu"):
        try:
            cpu_threads = 1 if device == "metal" else 0
            m = WhisperModel(name, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
            st.session_state["device_used"] = device
            return m
        except Exception:
            continue
    # 最後防禦：強制 CPU
    m = WhisperModel(name, device="cpu", compute_type=compute_type, cpu_threads=0)
    st.session_state["device_used"] = "cpu"
    return m

def fmt_dur(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


CJK_PUNCS = "，。、？！：；,.!?"
SENTENCE_END_PUNCS = "。！？…!?．."
ANY_END_PUNCS = "。！？，、；：…,.!?;:．"

ZH_PUNC_MAP = str.maketrans({
    ",": "，",
    ".": "。",
    "?": "？",
    "!": "！",
    ":": "：",
    ";": "；",
})

def normalize_zh_punc(s: str) -> str:
    if not s:
        return s
    # 替換半形英式標點為全形中文標點
    s = s.translate(ZH_PUNC_MAP)
    # 移除標點前多餘空白
    for p in "，。？！：；":
        s = s.replace(" " + p, p)
    return s


def has_any_punctuation(s: str) -> bool:
    return any(ch in CJK_PUNCS or ch in "、；：;:…" for ch in (s or ""))


def has_terminal_punctuation(s: str) -> bool:
    stripped = (s or "").rstrip()
    return bool(stripped) and stripped[-1] in ANY_END_PUNCS

# --- 簡易文本抽取與詞彙抽取 ---
def _read_pdf_bytes(name: str, data: bytes) -> str:
    try:
        # 輕依賴：pypdf，如環境未安裝，回退空字串
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(io.BytesIO(data))
        texts = []
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            if t:
                texts.append(t)
        return "\n".join(texts)
    except Exception:
        return ""

_MD_CODE_FENCE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_MD_INLINE_CODE = re.compile(r"`[^`]+`")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^\)]+\)")
_HTML_TAG = re.compile(r"<[^>]+>")

def _md_to_text(s: str) -> str:
    s = _MD_CODE_FENCE.sub(" ", s)
    s = _MD_IMAGE.sub(" ", s)
    s = _MD_LINK.sub(r"\1", s)
    s = _MD_INLINE_CODE.sub(" ", s)
    # 去除標題/清單符號
    s = re.sub(r"^[#>*\-+\s]+", "", s, flags=re.MULTILINE)
    s = _HTML_TAG.sub(" ", s)
    s = re.sub(r"\|", " ", s)  # 表格分隔
    s = re.sub(r"\s+", " ", s)
    return s.strip()

_RE_CJK = re.compile(r"[\u4e00-\u9fff]{2,10}")
_RE_EN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")

def extract_terms(text: str, top_k: int = 60) -> List[str]:
    if not text:
        return []
    counts: dict[str, int] = {}
    # CJK 片語（連續漢字 2~10）
    for m in _RE_CJK.finditer(text):
        tok = m.group(0)
        counts[tok] = counts.get(tok, 0) + 1
    # 英文/代號（保留大小寫的代表形式，統計以小寫）
    seen_case: dict[str, str] = {}
    for m in _RE_EN.finditer(text):
        raw = m.group(0)
        key = raw.lower()
        if key not in seen_case:
            seen_case[key] = raw
        counts[key] = counts.get(key, 0) + 1
    # 排序：頻率、長度
    items = []
    for k, v in counts.items():
        shown = seen_case.get(k, k)
        items.append((v, len(shown), shown))
    items.sort(key=lambda x: (-x[0], -x[1], x[2]))
    terms = []
    for _, _, w in items:
        if w not in terms:
            terms.append(w)
        if len(terms) >= max(10, top_k):
            break
    return terms

def build_domain_prompt(files, top_k: int = 60) -> tuple[str, List[str]]:
    texts = []
    for f in files or []:
        name = getattr(f, "name", "") or ""
        data = f.read()
        if hasattr(f, "seek"):
            try:
                f.seek(0)
            except Exception:
                pass
        try:
            if name.lower().endswith(".pdf"):
                t = _read_pdf_bytes(name, data)
            elif name.lower().endswith(".md"):
                t = _md_to_text(data.decode("utf-8", errors="ignore"))
            else:
                t = data.decode("utf-8", errors="ignore")
        except Exception:
            t = ""
        if t:
            weight = 1
            for hint in DOMAIN_PRIORITY_HINTS:
                if hint and hint in name:
                    # Boost repeated text so priority files surface more terms
                    weight = 5
                    break
            texts.extend([t] * weight)
    big = "\n".join(texts)
    terms = extract_terms(big, top_k=top_k)
    # 以全形逗號分隔，控制長度（避免過長 prompt）
    prompt = "，".join(terms)
    if len(prompt) > 800:
        prompt = prompt[:800]
    return prompt, terms

def count_term_hits(text: str, terms: List[str]) -> tuple[int, int]:
    if not text or not terms:
        return 0, 0
    uniq = 0
    total = 0
    t_low = text.lower()
    for t in terms:
        if not t:
            continue
        # 英文用不分大小寫；CJK 直接比對
        if re.search(r"[A-Za-z]", t):
            c = t_low.count(t.lower())
        else:            
            c = text.count(t)
        if c > 0:
            uniq += 1
            total += c
    return uniq, total

# 段落重排
def reflow_text(s: str, max_len: int = 60) -> str:
    if not s:
        return s
    # 先在句末標點後加段落
    out = s
    for p in ("。", "！", "？"):
        out = out.replace(p, p + "\n\n")
    # 基於長度的簡單換行：每段超過 max_len 且末尾為逗號時換行
    lines = []
    for para in out.split("\n\n"):
        buf = ""
        for ch in para:
            buf += ch
            if len(buf) >= max_len and buf.endswith("，"):
                lines.append(buf.strip())
                buf = ""
        if buf:
            lines.append(buf.strip())
    return "\n\n".join(lines)





def typical_gap_from_words(words) -> float:
    gaps = []
    prev_e = None
    for w in words:
        if prev_e is not None and w.start is not None and prev_e is not None:
            g = (w.start or 0.0) - (prev_e or 0.0)
            if g > 0.03:  # 忽略極短數值抖動
                gaps.append(g)
        prev_e = w.end or prev_e
    return median(gaps) if gaps else 0.0

def save_wav(bytes_data, path):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
        w.writeframes(bytes_data)

def audio_info_from_bytes(b: bytes, filename: str|None=None) -> Tuple[AudioSegment, float]:
    buf = io.BytesIO(b)
    fmt = None
    if filename and "." in filename:
        fmt = filename.rsplit(".",1)[-1].lower()
    seg = AudioSegment.from_file(buf, format=fmt)  # 交給 pydub/ffmpeg 判斷
    dur = seg.duration_seconds
    return seg, dur

def export_seg_to_wav(seg: AudioSegment, path: str):
    seg.set_channels(1).set_frame_rate(16000).set_sample_width(2).export(path, format="wav")

def start_timecode(t: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(max(t, 0))) + f",{int((t%1)*1000):03d}"

def to_seconds(ts: str) -> float:
    """將 SRT 時間碼 HH:MM:SS,mmm 轉為秒數"""
    try:
        h, m, s_ms = ts.split(":")
        s, ms = s_ms.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    except Exception:
        return 0.0
    

@dataclass(frozen=True)
class LexicalRules:
    sentence_starters: frozenset[str]
    clause_starters: frozenset[str]
    clause_pause_after: frozenset[str]
    continuation_words: frozenset[str]
    sentence_break_after: frozenset[str]
    question_endings: str
    enum_token_re: Pattern[str]
    no_break_suffixes: tuple[str, ...]
    sentence_break_tails: tuple[str, ...]
    clause_break_before: tuple[str, ...]


def _normalize_token_list(raw: Any) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        raw = raw.split(",")
    if isinstance(raw, (list, tuple, set)):
        out = []
        for item in raw:
            token = str(item).strip()
            if token:
                out.append(token)
        return out
    return []


DEFAULT_LEXICAL_RULES = LexicalRules(
    sentence_starters=frozenset(
        {
            "因此", "所以", "然而", "但是", "不過", "可是", "結果",
            "最後", "最後一個", "最後面", "接下來", "然後", "再來", "另外",
            "此外", "除此之外", "同時", "總之", "總而言之",
        }
    ),
    clause_starters=frozenset(
        {
            "因為", "如果", "當", "當然", "尤其", "尤其是", "特別是",
            "例如", "比如", "比如說", "舉例來說", "換句話說", "也就是說",
            "或者", "或者是", "以及", "還有", "另外", "再加上", "同時",
            "包含", "包括",
        }
    ),
    clause_pause_after=frozenset(
        {
            "例如", "比如", "比如說", "舉例來說", "換句話說", "也就是說",
            "總之", "總而言之", "所以", "因此", "因為", "結果",
        }
    ),
    continuation_words=frozenset(
        {
            "和", "跟", "與", "及", "並且", "而且", "以及", "還有", "還要", "還會", "還會再", "再", "又",
            "同時", "或者", "或是", "而", "而且在", "並且在", "而在", "還在",
        }
    ),
    sentence_break_after=frozenset(
        {
            "因此", "所以", "總之", "總而言之", "結果", "換句話說", "也就是說", "因而", "因此在",
            "因此就", "所以就", "因此才",
        }
    ),
    question_endings="嗎呢？?",
    enum_token_re=re.compile(r"^(第[一二三四五六七八九十百千]+|[一二三四五六七八九十]+)[、．.]?$"),
    no_break_suffixes=(
        "的", "得", "地", "著", "着", "了", "呢", "嗎", "嘛", "啊", "呀", "啦", "吧", "喔", "哦", "噢",
    ),
    sentence_break_tails=(
        "的時候", "的階段", "的情況下", "的結果", "的狀況下", "之下", "之前", "之後", "之時", "之際",
    ),
    clause_break_before=(
        "如果", "假如", "當", "當你", "當我們", "當然", "尤其", "尤其是", "特別是",
        "另外", "此外", "還有", "以及", "再加上", "同時", "甚至", "甚至於", "或者", "或是",
        "例如", "比如", "比如說", "舉例來說", "換句話說", "也就是說",
    ),
)


def _load_lexical_rules(base: LexicalRules, overrides: Dict[str, Any] | None) -> LexicalRules:
    if not overrides or not isinstance(overrides, dict):
        return base

    def merge_set(key: str, current: frozenset[str]) -> frozenset[str]:
        tokens = _normalize_token_list(overrides.get(key))
        return current if not tokens else frozenset(tokens)

    def merge_tuple(key: str, current: tuple[str, ...]) -> tuple[str, ...]:
        tokens = _normalize_token_list(overrides.get(key))
        return current if not tokens else tuple(tokens)

    qe_raw = overrides.get("question_endings")
    question_endings = base.question_endings
    if isinstance(qe_raw, str) and qe_raw.strip():
        question_endings = qe_raw.strip()

    enum_pattern_raw = overrides.get("enum_token_pattern")
    enum_token_re = base.enum_token_re
    if isinstance(enum_pattern_raw, str) and enum_pattern_raw.strip():
        try:
            enum_token_re = re.compile(enum_pattern_raw.strip())
        except re.error:
            pass

    return LexicalRules(
        sentence_starters=merge_set("sentence_starters", base.sentence_starters),
        clause_starters=merge_set("clause_starters", base.clause_starters),
        clause_pause_after=merge_set("clause_pause_after", base.clause_pause_after),
        continuation_words=merge_set("continuation_words", base.continuation_words),
        sentence_break_after=merge_set("sentence_break_after", base.sentence_break_after),
        question_endings=question_endings,
        enum_token_re=enum_token_re,
        no_break_suffixes=merge_tuple("no_break_suffixes", base.no_break_suffixes),
        sentence_break_tails=merge_tuple("sentence_break_tails", base.sentence_break_tails),
        clause_break_before=merge_tuple("clause_break_before", base.clause_break_before),
    )


LEXICAL_RULES = _load_lexical_rules(
    DEFAULT_LEXICAL_RULES,
    DEFAULTS_CONFIG.get("punctuation_rules"),
)


class SmartPunctuator:
    """Rule-based punctuation controller combining timing gaps and lexical heuristics."""

    def __init__(self, rules: LexicalRules | None = None) -> None:
        self.rules = rules or LEXICAL_RULES
        self.tokens: List[str] = []
        self.prev_end: float | None = None
        self.sentence_chars: int = 0
        self.clause_chars: int = 0
        self.prev_word: str = ""

    def _char_len(self, text: str) -> int:
        return sum(1 for ch in text if ch not in CJK_PUNCS and not ch.isspace())

    def _after_punc(self, punc: str) -> None:
        if punc in ("。", "！", "？"):
            self.sentence_chars = 0
            self.clause_chars = 0
        elif punc in ("，", "、", "；", "："):
            self.clause_chars = 0

    def _append_to_last(self, punc: str) -> None:
        if not self.tokens:
            return
        last = self.tokens[-1].rstrip()
        if not last:
            return
        if has_terminal_punctuation(last):
            return
        last_tail = last[-1]
        if last_tail in "。！？" and punc in "。！？":
            return
        if last_tail in "，、；：" and punc in "，、；：":
            return
        if last_tail in "，、；：" and punc in "。！？":
            last = last.rstrip("，、；：")
        self.tokens[-1] = last + punc
        self._after_punc(punc)
        self.prev_word = self.tokens[-1]

    def _push_word(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.tokens.append(text)
        char_len = self._char_len(text)
        self.sentence_chars += char_len
        self.clause_chars += char_len
        tail = text[-1]
        if tail in "。！？，、；：":
            self._after_punc(tail)
        self.prev_word = text

    def _choose_sentence_end(self) -> str:
        last = self.prev_word.strip()
        if last and last[-1] in self.rules.question_endings:
            return "？"
        return "。"

    def _decide_punc(self, gap: float, upcoming: str, comma_th: float, period_th: float, coarse: bool) -> str:
        if not self.tokens:
            return ""
        gap = max(0.0, gap)
        sentence_score = 0.0
        comma_score = 0.0
        upcoming = upcoming.strip()
        prev = self.prev_word.strip()

        # 時間間隔權重
        if gap >= period_th:
            sentence_score += 4.5
        elif gap >= period_th * 0.9:
            sentence_score += 3.2
        elif gap >= max(period_th * 0.7, comma_th * 1.3):
            sentence_score += 1.4

        if gap >= comma_th:
            comma_score += 3.0
        elif gap >= comma_th * 0.85:
            comma_score += 2.0
        elif gap >= comma_th * 0.7:
            comma_score += 0.8

        # 句長/子句長權重
        if self.sentence_chars >= 52:
            sentence_score += 4.0
        elif self.sentence_chars >= 40:
            sentence_score += 2.4
        elif self.sentence_chars >= 28 and self.clause_chars >= 16:
            sentence_score += 1.2

        if self.clause_chars >= 28:
            comma_score += 2.6
        elif self.clause_chars >= 20:
            comma_score += 1.8
        elif self.clause_chars >= 14:
            comma_score += 1.0

        lexical_force = ""

        # 詞彙 heuristics
        if upcoming:
            if upcoming in self.rules.sentence_starters and self.sentence_chars >= 12:
                sentence_score += 2.6
            if upcoming in self.rules.clause_starters and self.clause_chars >= 6:
                comma_score += 2.4
            if upcoming in self.rules.clause_pause_after and self.clause_chars >= 6:
                comma_score += 1.4
            if self.rules.enum_token_re.match(upcoming) and self.sentence_chars >= 8:
                sentence_score += 2.2
            if any(upcoming.startswith(pat) for pat in self.rules.clause_break_before):
                lexical_force = "comma"
                comma_score += 2.0

        if prev:
            trimmed_prev = prev.rstrip("，、；：")
            if trimmed_prev[-1:] in self.rules.question_endings:
                sentence_score += 1.0
            if trimmed_prev in self.rules.clause_pause_after:
                comma_score += 1.6
            if trimmed_prev in self.rules.sentence_starters:
                sentence_score += 1.2
            if trimmed_prev.endswith(("但是", "然而", "可是")):
                sentence_score += 1.4
            if any(trimmed_prev.endswith(pat) for pat in self.rules.sentence_break_tails):
                sentence_score += 2.5
                lexical_force = lexical_force or "comma"
            if trimmed_prev in self.rules.sentence_break_after:
                lexical_force = "period"

        # 粗粒度輸入時減少信心
        if coarse:
            comma_score *= 0.85
            sentence_score *= 0.85

        # 避免過短子句插入，同時保留適量標點
        if self.clause_chars < 6:
            comma_score *= 0.55
            sentence_score *= 0.7

        # 遇到語助詞或連接詞時降低信心，避免在句中斷句
        def _should_penalize(word: str) -> bool:
            if not word:
                return False
            w = word.strip().rstrip("，、；：")
            if not w:
                return False
            if len(w) <= 1 and w not in self.rules.question_endings:
                return True
            if w in self.rules.continuation_words:
                return True
            if w.endswith(self.rules.no_break_suffixes):
                return True
            return False

        penal_prev = _should_penalize(prev)
        penal_next = _should_penalize(upcoming)
        if penal_prev:
            comma_score *= 0.6
            sentence_score *= 0.75

        if penal_next:
            comma_score *= 0.7
            sentence_score *= 0.8

        if penal_prev and penal_next:
            comma_score *= 0.65
            sentence_score *= 0.7

        if lexical_force == "period" and gap >= max(comma_th * 0.4, 0.35):
            sentence_score += 3.0
        elif lexical_force == "comma" and gap >= max(comma_th * 0.3, 0.25):
            comma_score += 1.8

        if sentence_score >= max(3.5, comma_score + 0.8):
            return "period"
        if comma_score >= max(2.2, sentence_score + 0.3):
            return "comma"
        return ""



    def _maybe_insert(self, gap: float, upcoming: str, comma_th: float, period_th: float, coarse: bool) -> None:
        """
        根據時間間隔與語境決定是否在前一個 token 後插入標點。
        - 若 gap <= 0：強制加逗號，避免句子緊接或重疊時無斷點。
        - 其他情況：交由 _decide_punc() 判斷句號／逗號。
        """
        if self.tokens and has_terminal_punctuation(self.tokens[-1]):
            return
        # 若 gap 為負或零（重疊、緊接），仍強制插入逗號
        if gap <= 0 and self.tokens:
            self._append_to_last("，")
            return

        # 一般決策邏輯
        decision = self._decide_punc(gap, upcoming, comma_th, period_th, coarse)
        if not decision:
            return

        if decision == "period":
            punc = self._choose_sentence_end()
        else:
            punc = "，"

        self._append_to_last(punc)

    def add_word(self, word: str, start: float | None, end: float | None, comma_th: float, period_th: float) -> None:
        word = word.strip()
        if not word:
            return
        if self.prev_end is None:
            gap = float(start or 0.0)
        else:
            gap = float((start or 0.0) - (self.prev_end or 0.0))
        self._maybe_insert(gap, word, comma_th, period_th, coarse=False)
        self._push_word(word)
        if end is not None:
            self.prev_end = float(end)
        elif self.prev_end is None:
            self.prev_end = float(start or 0.0)

    def add_chunk(self, text: str, start: float | None, end: float | None, comma_th: float, period_th: float) -> None:
        """安全支援 None 時間輸入"""
        text = text.strip()
        if not text:
            return
        s_val = float(start or 0.0)
        e_val = float(end or s_val)
        gap = s_val - (self.prev_end or 0.0) if self.prev_end is not None else s_val
        self._maybe_insert(gap, text, comma_th, period_th, coarse=True)
        self._push_word(text)
        self.prev_end = e_val

    def ensure_terminal(self) -> None:
        if not self.tokens:
            return
        last = self.tokens[-1].rstrip()
        if not last:
            return
        if has_terminal_punctuation(last):
            return
        self.tokens[-1] = last + self._choose_sentence_end()
        self._after_punc(self.tokens[-1][-1])



def _read_srt_blocks(srt_text: str) -> List[Tuple[float, float, str]]:
    blocks: List[Tuple[float, float, str]] = []
    for raw_block in re.split(r"\n\s*\n", srt_text.strip()):
        lines = [line.strip() for line in raw_block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        time_line = next((line for line in lines if "-->" in line), "")
        if not time_line:
            continue
        try:
            start_str, end_str = time_line.split("-->", 1)
            start_sec = to_seconds(start_str.strip())
            end_sec = to_seconds(end_str.strip())
        except Exception:
            start_sec, end_sec = 0.0, 0.0
        text_idx = lines.index(time_line) + 1
        text = " ".join(lines[text_idx:]).strip()
        if text:
            blocks.append((start_sec, end_sec, text))
    return blocks


_WHISPERCPP_SEGMENT_RE = re.compile(
    r"^\[(\d{2}:\d{2}:\d{2}[,.]\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}[,.]\d{3})\]\s*(.*)$"
)


def _parse_whispercpp_timecode(ts: str) -> float:
    return to_seconds(ts.replace(".", ","))


def _srt_blocks_to_srt(blocks: List[Tuple[float, float, str]], time_offset_sec: float = 0.0) -> str:
    out: List[str] = []
    for idx, (start, end, text) in enumerate(blocks, 1):
        out.append(str(idx))
        out.append(f"{start_timecode(start + time_offset_sec)} --> {start_timecode(end + time_offset_sec)}")
        out.append(text)
        out.append("")
    return "\n".join(out)


def _final_text_from_blocks(
    blocks: List[Tuple[float, float, str]],
    punc_rule: bool,
    lexical_rules: LexicalRules | None,
    punc_settings: PunctuationSettings | None,
):
    lexical_cfg = lexical_rules or LEXICAL_RULES
    punctuation_cfg = punc_settings or CURRENT_PUNCT_SETTINGS
    normalized_blocks = [
        (start, end, normalize_zh_punc(apply_common_corrections(cc.convert(text).strip())))
        for start, end, text in blocks
        if text and text.strip()
    ]
    source_has_punctuation = any(has_any_punctuation(text) for _, _, text in normalized_blocks)

    if punc_rule and not source_has_punctuation:
        final_punc = SmartPunctuator(lexical_cfg)
        for start, end, text in normalized_blocks:
            if text:
                final_punc.add_chunk(text, start, end, punctuation_cfg.comma_threshold, punctuation_cfg.period_threshold)
        final_punc.ensure_terminal()
        tokens_fixed = []
        for token in final_punc.tokens:
            token = token.rstrip()
            if not token:
                continue
            if not has_terminal_punctuation(token):
                token += "。"
            tokens_fixed.append(token)
        return normalize_zh_punc("\n".join(tokens_fixed))

    joiner = "\n" if source_has_punctuation else ""
    final_text = joiner.join(text for _, _, text in normalized_blocks)
    if final_text and not has_terminal_punctuation(final_text):
        final_text += "。"
    return final_text


def _resolve_whispercpp_cli(cli_path: str) -> str:
    cli_path = (cli_path or "").strip()
    if cli_path:
        expanded = str(Path(cli_path).expanduser())
        if Path(expanded).exists():
            return expanded
        found = shutil.which(cli_path)
        if found:
            return found
    found = shutil.which("whisper-cli") or shutil.which("main")
    if found:
        return found
    raise FileNotFoundError("找不到 whisper.cpp CLI。請填入 whisper-cli/main 的完整路徑。")


@st.cache_data(show_spinner=False)
def _whispercpp_cli_supports_arg(cli_path: str, arg: str) -> bool:
    try:
        result = subprocess.run([cli_path, "-h"], capture_output=True, text=True, timeout=5)
    except Exception:
        return False
    return arg in ((result.stdout or "") + (result.stderr or ""))


def transcribe_one_whispercpp(
    media_path: str,
    cli_path: str,
    model_path: str,
    language: str,
    beam_size: int,
    initial_prompt: str | None,
    punc_rule: bool,
    ui_area,
    progress_area,
    stats_area,
    threads: int,
    time_offset_sec: float = 0.0,
    total_sec_for_progress: float | None = None,
    lexical_rules: LexicalRules | None = None,
    punc_settings: PunctuationSettings | None = None,
    progress_label: str = "whisper.cpp 轉寫中…",
):
    """Run whisper.cpp CLI and adapt its SRT output to the existing post-processing flow."""
    live_box = ui_area.empty()
    t0 = time.time()
    total_sec = total_sec_for_progress or 1.0
    prog = progress_area.progress(0.0, text=progress_label)

    cli = _resolve_whispercpp_cli(cli_path)
    model = str(Path(model_path).expanduser()) if model_path else ""
    if not model or not Path(model).exists():
        raise FileNotFoundError("找不到 whisper.cpp 模型檔。請填入 .bin/.gguf 模型完整路徑。")

    out_prefix = str(Path(media_path).with_suffix("")) + "_whispercpp"
    for stale in Path(media_path).parent.glob(Path(out_prefix).name + ".*"):
        try:
            stale.unlink()
        except OSError:
            pass

    cmd = [
        cli,
        "-m", model,
        "-f", media_path,
        "-l", language or "zh",
        "-osrt",
        "-of", out_prefix,
        "-t", str(max(1, int(threads or 1))),
        "-bs", str(max(1, int(beam_size or 1))),
    ]
    if _whispercpp_cli_supports_arg(cli, "-fa"):
        cmd.append("-fa")
    if initial_prompt:
        cmd.extend(["--prompt", initial_prompt])

    status_text = "正在載入模型…"
    prog.progress(0.02, text=f"{status_text}｜耗時 00:00:00")
    live_blocks: List[Tuple[float, float, str]] = []
    log_tail: List[str] = []
    saw_processing = False
    last_prelude_update = 0.0
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        log_tail.append(line)
        log_tail = log_tail[-40:]
        match = _WHISPERCPP_SEGMENT_RE.match(line)
        if not match:
            if "processing" in line:
                saw_processing = True
                elapsed = time.time() - t0
                prog.progress(0.06, text=f"正在轉寫…｜已轉錄音時長 00:00:00 / {fmt_dur(total_sec)}，耗時 {fmt_dur(elapsed)}")
                last_prelude_update = time.time()
            elif not live_blocks and time.time() - last_prelude_update > 3:
                elapsed = time.time() - t0
                prelude_progress = 0.05 if saw_processing else 0.03
                label = "等待第一段文字…" if saw_processing else "正在載入模型…"
                prog.progress(prelude_progress, text=f"{label}｜已轉錄音時長 00:00:00 / {fmt_dur(total_sec)}，耗時 {fmt_dur(elapsed)}")
                last_prelude_update = time.time()
            continue
        start_sec = _parse_whispercpp_timecode(match.group(1))
        end_sec = _parse_whispercpp_timecode(match.group(2))
        text = normalize_zh_punc(apply_common_corrections(cc.convert(match.group(3).strip())))
        if not text:
            continue
        live_blocks.append((start_sec, end_sec, text))
        processed_sec = max(0.0, end_sec)
        elapsed = time.time() - t0
        eta_txt = ""
        if processed_sec > 5 and elapsed > 2:
            rate = processed_sec / max(elapsed, 1e-6)
            remain_audio = max(0.0, total_sec - processed_sec)
            est_remain = remain_audio / max(rate, 1e-6)
            finish_ts = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(time.time() + est_remain)
            )
            eta_txt = f"｜預計剩餘 {fmt_dur(est_remain)}｜預計完成 {finish_ts}"
        live_box.markdown("  \n".join(block_text for _, _, block_text in live_blocks))
        prog.progress(
            min(0.99, max(0.01, processed_sec / max(total_sec, 1e-6))),
            text=f"已轉錄音時長 {fmt_dur(processed_sec)} / {fmt_dur(total_sec)}，耗時 {fmt_dur(elapsed)}{eta_txt}",
        )
    returncode = proc.wait()
    if returncode != 0:
        detail = "\n".join(log_tail).strip()
        raise RuntimeError(f"whisper.cpp 轉寫失敗：{detail[-1200:]}")

    srt_path = Path(out_prefix + ".srt")
    if not srt_path.exists():
        raise FileNotFoundError(f"whisper.cpp 沒有產生 SRT：{srt_path}")

    raw_srt = srt_path.read_text(encoding="utf-8", errors="replace")
    blocks = []
    for start, end, text in _read_srt_blocks(raw_srt):
        text = normalize_zh_punc(apply_common_corrections(cc.convert(text).strip()))
        if text:
            blocks.append((start, end, text))

    live_box.markdown("  \n".join(text for _, _, text in blocks))
    final_srt = _srt_blocks_to_srt(blocks, time_offset_sec=time_offset_sec)
    final_text = _final_text_from_blocks(blocks, punc_rule, lexical_rules, punc_settings)

    elapsed = time.time() - t0
    processed_sec = max((end for _, end, _ in blocks), default=total_sec)
    prog.progress(1.0, text=f"已轉錄音時長 {fmt_dur(processed_sec)} / {fmt_dur(total_sec)}，耗時 {fmt_dur(elapsed)}")
    stats_area.info(f"已轉錄音時長：{fmt_dur(processed_sec)}；耗時：{fmt_dur(elapsed)}")

    try:
        srt_path.unlink()
    except OSError:
        pass
    return final_text, final_srt, processed_sec, elapsed


def transcribe_one(
    media_path: str,
    model: WhisperModel,
    language: str,
    vad_filter: bool,
    beam_size: int,
    initial_prompt: str | None,
    punc_rule: bool,
    vad_parameters: dict | None,
    ui_area,
    progress_area,
    stats_area,
    time_offset_sec: float = 0.0,
    total_sec_for_progress: float | None = None,
    lexical_rules: LexicalRules | None = None,
    punc_settings: PunctuationSettings | None = None,
    progress_label: str = "分段處理中…",
):
    """單段轉錄，SRT 無標點，即時顯示為原始小句，最終整合使用 SmartPunctuator 補標點。"""
    live_box = ui_area.empty()
    acc = []  # 收集原始小句
    prev_end = 0.0
    srt_lines, idx = [], 1
    t0 = time.time()
    processed_sec = 0.0
    prog = progress_area.progress(0.0, text=progress_label)
    total_sec = total_sec_for_progress or 1.0
    lexical_cfg = lexical_rules or LEXICAL_RULES
    punctuation_cfg = punc_settings or CURRENT_PUNCT_SETTINGS


    # === 轉錄 ===
    segments, info = model.transcribe(
        media_path,
        language=language,
        vad_filter=vad_filter,
        beam_size=beam_size,
        temperature=0.2,
        # best_of=5,
        condition_on_previous_text=True,
        task="transcribe",
        initial_prompt=initial_prompt or None,
        word_timestamps=True,
        vad_parameters=vad_parameters,
    )

    displayed_idx = 0
    for seg in segments:
        words = getattr(seg, "words", None)
        c_th = punctuation_cfg.comma_threshold
        p_th = punctuation_cfg.period_threshold
        adaptive_enabled = punctuation_cfg.adaptive and punc_rule
        if adaptive_enabled and words:
            tg = typical_gap_from_words(words)
            if tg > 0:
                c_th = max(0.10, min(c_th, 0.60 * tg))
                p_th = max(c_th + 0.03, min(p_th, 1.20 * tg))

        # --- 不在即時階段加標點，只收集純文本 ---
        raw_text = normalize_zh_punc(cc.convert(seg.text)).strip()
        if not raw_text:
            continue
        corrected_text = apply_common_corrections(raw_text)
        acc.append(raw_text)
        prev_end = seg.end or prev_end

        # --- 寫入 SRT（無標點小句） ---
        adj_start = (seg.start or 0.0) + time_offset_sec
        adj_end = (seg.end or 0.0) + time_offset_sec
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start_timecode(adj_start)} --> {start_timecode(adj_end)}")
        srt_lines.append(corrected_text)  # 無標點
        srt_lines.append("")
        idx += 1

        # --- 即時顯示：所有已生成小句，無重複 ---
        all_lines = [
            apply_common_corrections(line.strip())
            for i, line in enumerate(srt_lines)
            if i % 4 == 2 and line.strip()
        ]
        new_lines = all_lines[displayed_idx:]
        if new_lines:
            displayed_idx = len(all_lines)
            cur_display_text = "  \n".join(all_lines)
            live_box.markdown(cur_display_text)

            # ⏺️ Incremental save during transcription
            txt_path = st.session_state.get("current_txt_path")
            srt_path = st.session_state.get("current_srt_path")
            if txt_path and srt_path:
                # Write to TXT
                with open(txt_path, "a", encoding="utf-8") as f_txt:
                    f_txt.write(corrected_text + "\n")
                # Write full SRT block
                with open(srt_path, "a", encoding="utf-8") as f_srt:
                    f_srt.write(
                        srt_lines[-4] + "\n" +  # index
                        srt_lines[-3] + "\n" +  # timecode
                        srt_lines[-2] + "\n\n"  # text + blank
                    )

        # --- 更新進度 ---
        processed_sec = max(processed_sec, float(seg.end or 0.0))
        elapsed = time.time() - t0
        eta_txt = ""
        if processed_sec > 5 and elapsed > 2:
            rate = processed_sec / max(elapsed, 1e-6)
            remain_audio = max(0.0, total_sec - processed_sec)
            est_remain = remain_audio / max(rate, 1e-6)
            finish_ts = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(time.time() + est_remain)
            )
            eta_txt = f"｜預計剩餘 {fmt_dur(est_remain)}｜預計完成 {finish_ts}"
        prog.progress(
            min(1.0, processed_sec / total_sec),
            text=f"已轉錄音時長 {fmt_dur(processed_sec)} / {fmt_dur(total_sec)}，耗時 {fmt_dur(elapsed)}{eta_txt}",
        )



    if punc_rule:
        final_punc = SmartPunctuator(lexical_cfg)
        srt_chunks = []
        # Collect the raw SRT payload (text + timing) so punctuation can respect pauses
        for i in range(0, len(srt_lines), 4):
            if i + 2 >= len(srt_lines):
                continue
            time_line = srt_lines[i + 1]
            text_line = srt_lines[i + 2].strip()
            if not text_line:
                continue
            try:
                start_str, end_str = time_line.split("-->")
                start_sec = to_seconds(start_str.strip())
                end_sec = to_seconds(end_str.strip())
            except Exception:
                start_sec, end_sec = None, None
            srt_chunks.append((text_line, start_sec, end_sec))

        normalized_chunks = [
            (normalize_zh_punc(apply_common_corrections(text)), start, end)
            for text, start, end in srt_chunks
            if text and text.strip()
        ]
        source_has_punctuation = any(has_any_punctuation(text) for text, _, _ in normalized_chunks)

        if source_has_punctuation:
            tokens_fixed = []
            for text, _, _ in normalized_chunks:
                text = text.rstrip()
                if text:
                    tokens_fixed.append(text)
            if tokens_fixed and not has_terminal_punctuation(tokens_fixed[-1]):
                tokens_fixed[-1] += "。"
        else:
            # SmartPunctuator 補標點
            for text, start, end in normalized_chunks:
                final_punc.add_chunk(text, start, end, punctuation_cfg.comma_threshold, punctuation_cfg.period_threshold)

            final_punc.ensure_terminal()

            tokens_fixed = []
            for t in final_punc.tokens:
                t = t.rstrip()
                if not t:
                    continue
                if not has_terminal_punctuation(t):
                    t += "。"
                tokens_fixed.append(t)



        # 每句換行輸出（每句已加標點）
        final_text = "\n".join(tokens_fixed)
        final_text = normalize_zh_punc(final_text)

    else:
        # 沒開標點規則就單純 join
        final_text = "".join(
            normalize_zh_punc(apply_common_corrections(line.strip()))
            for i, line in enumerate(srt_lines)
            if i % 4 == 2
        )
        if final_text and final_text[-1] not in "。！？":
            final_text += "。"

    if 'reflow_enable' in globals() and reflow_enable:
        final_text = reflow_text(final_text, max_len=reflow_width)


    elapsed = time.time() - t0
    stats_area.info(f"已轉錄音時長：{fmt_dur(processed_sec)}；耗時：{fmt_dur(elapsed)}")

    return final_text, "\n".join(srt_lines), processed_sec, elapsed



if not IS_WORKER:
    # ==== 輸入 ====
    uploaded = None
    uploaded_name = None
    uploaded = st.file_uploader("上傳音檔", type=["m4a","mp3","wav","flac"], key="main_audio_uploader")
    st.markdown(
        """
        <style>
        /* 只放大「上傳音檔」區塊，不影響其他 uploader */
        section[data-testid="stFileUploaderDropzone"][aria-label*="上傳音檔"] {
            min-height: 30rem !important;
        }
        </style>
        <script>
        (function() {
            const w = window.parent || window;
            if (w.__whispertc_global_drop_bound) return;

            function findMainAudioInput() {
                const sections = Array.from(w.document.querySelectorAll('section[data-testid="stFileUploaderDropzone"]'));
                for (const sec of sections) {
                    const aria = (sec.getAttribute('aria-label') || '');
                    if (!aria.includes('上傳音檔')) continue;
                    const input = sec.querySelector('input[type="file"][data-testid="stFileUploaderDropzoneInput"]');
                    if (input) return input;
                }
                return null;
            }

            function handleDragOver(e) {
                const hasFiles = e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files');
                if (hasFiles) {
                    e.preventDefault();
                }
            }

            function handleDrop(e) {
                const files = e.dataTransfer && e.dataTransfer.files;
                if (!files || !files.length) return;
                e.preventDefault();
                const target = findMainAudioInput();
                if (!target) return;
                const dt = new DataTransfer();
                Array.from(files).forEach(f => dt.items.add(f));
                target.files = dt.files;
                target.dispatchEvent(new Event('change', { bubbles: true }));
            }

            w.addEventListener('dragover', handleDragOver);
            w.addEventListener('drop', handleDrop);
            w.__whispertc_global_drop_bound = true;
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )
    if uploaded is not None:
        uploaded_name = uploaded.name

        # 設定上傳狀態
        st.session_state['file_uploaded'] = True


    # 顯示/下載區
    status = st.empty()
    top_info = st.empty()


    # === 顯示模式控制（放在整合顯示區最上面） ===

    # 初始化狀態
    transcribing = st.session_state.get("transcribing", False)

    # 顯示灰化樣式 + 禁用游標
    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"] label[aria-disabled="true"] {
            opacity: 0.4 !important;
            pointer-events: none !important;
            cursor: not-allowed !important;
        }
        div[data-testid="stIFrame"][height="0"],
        iframe[height="0"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 顯示模式選項
    show_line_mode = st.radio(
        "顯示模式",
        ["每句換行", "整段顯示"],
        index=st.session_state.get("show_line_mode_idx", 0),
        key="show_line_mode",
        horizontal=True,
        disabled=transcribing,  # 若正在轉錄則禁用
        help="切換顯示方式，不影響下載內容",
    )

    if transcribing:
        st.caption("🔒 轉錄進行中，暫時無法切換顯示模式。")

    # 讓即時進度條固定在顯示模式切換的下方
    realtime_progress_area = st.empty()

    # === 顯示結果區 ===
    final_box = st.empty()

    status_message = st.session_state.get("transcribe_status_message")
    success_message = st.session_state.get("transcribe_success_message")
    if success_message:
        status.success(success_message)
    elif status_message:
        status.info(status_message)

    # 若有快取結果（上次轉錄），自動載入
    last_txt = st.session_state.get("last_txt")
    if last_txt:
        line_break_version = re.sub(r"\s*([。！？])\s*", r"\1\n", last_txt)
        line_break_version = re.sub(r"\n+", "\n", line_break_version).strip()
        display_text = (
            line_break_version
            if show_line_mode == "每句換行"
            else last_txt.replace("\n", " ")
        )





# === 鎖定顯示模式與主流程 ===
if not IS_WORKER and st.session_state.get("start_transcribe_pending"):
    if uploaded is None:
        status.warning("請上傳音訊檔再開始轉寫")
        st.session_state["transcribing"] = False
        st.session_state["start_transcribe_pending"] = False
        st.session_state["transcribe_status_message"] = ""
        st.session_state["transcribe_success_message"] = ""
        if "run_button_primary" in st.session_state:
            del st.session_state["run_button_primary"]
        st.stop()

    st.session_state["transcribe_status_message"] = "⚙️ 正在準備載入模型…"
    status.info(st.session_state["transcribe_status_message"])

    # --- 從這裡開始進入主流程 ---
    st.session_state["transcribing"] = True

    # 準備臨時目錄
    tmp_dir = os.getcwd()
    ts = int(time.time())

    # === 設定本次 run 的暫存/增量儲存路徑（以 timestamp 區分） ===
    base_filename = os.path.splitext(uploaded_name)[0]
    txt_path_run = os.path.join(tmp_dir, f"{base_filename}_{ts}.txt")
    srt_path_run = os.path.join(tmp_dir, f"{base_filename}_{ts}.srt")
    st.session_state["current_txt_path"] = txt_path_run
    st.session_state["current_srt_path"] = srt_path_run
    st.session_state["current_base_filename"] = base_filename
    st.session_state["current_ts"] = ts

    # 取得 bytes 與時長
    raw = uploaded.read()
    seg, total_sec = audio_info_from_bytes(raw, uploaded_name)

    upload_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    top_info.info(f"上傳錄音時長：{fmt_dur(total_sec)}｜上傳時間點：{upload_ts}")

    # 轉為單一 wav 檔（作為後續切段基礎）
    base_path = os.path.join(tmp_dir, f"_tmp_{ts}.wav")
    export_seg_to_wav(seg, base_path)

    # 構建初始提示（合併使用者提示與領域詞彙）
    combined_prompt = (init_prompt or "").strip()
    domain_prompt_preview = ""
    domain_terms: List[str] = []
    candidate_domain_files = combined_domain_files if 'combined_domain_files' in globals() else []
    if 'auto_use_domain' in globals() and auto_use_domain and candidate_domain_files:
        status.info("正在擷取領域詞彙…")
        st.session_state["transcribe_status_message"] = "正在擷取領域詞彙…"
        try:
            domain_prompt_preview, domain_terms = build_domain_prompt(candidate_domain_files, top_k=terms_topk)
        except Exception:
            domain_prompt_preview = ""
            domain_terms = []
        if domain_prompt_preview:
            combined_prompt = (combined_prompt + ("，" if combined_prompt else "") + domain_prompt_preview).strip("， ")
            with st.sidebar:
                st.caption("已帶入領域詞彙（前 800 字內）：")
                st.code(domain_prompt_preview[:400] + ("…" if len(domain_prompt_preview) > 400 else ""))

    # 載入模型
    if asr_backend == "whisper.cpp":
        if not _is_whispercpp_optimal(whispercpp_cli, whispercpp_model, asr_backend, beam_size):
            st.error("STT 未使用最佳配置。請確認後端為 whisper.cpp、模型為 medium-q5_0 或 medium、beam=1。")
            st.session_state["transcribing"] = False
            st.session_state["start_transcribe_pending"] = False
            st.session_state["transcribe_status_message"] = ""
            if "run_button_primary" in st.session_state:
                del st.session_state["run_button_primary"]
            st.stop()
        status.info(f"準備 whisper.cpp：{Path(whispercpp_model).name if whispercpp_model else '尚未指定模型'} / beam={beam_size}")
        st.session_state["transcribe_status_message"] = f"準備 whisper.cpp / beam={beam_size}"
        model = None
    else:
        status.info(f"載入模型：{model_name} / {compute} / beam={beam_size}")
        st.session_state["transcribe_status_message"] = f"載入模型：{model_name} / {compute} / beam={beam_size}"
        model = load_model(model_name, compute)

    # === 切段與並行策略 ===
    device_now = "whisper.cpp" if asr_backend == "whisper.cpp" else st.session_state.get("device_used", "cpu")
    overlap = 5.0

    def make_chunks(total_s: float, k: int):
        """Create k roughly equal segments with slight overlap to protect word boundaries."""
        step = total_s / max(1, k)
        out = []
        for i in range(k):
            start = max(0.0, i*step - (overlap/2 if i>0 else 0.0))
            end = min(total_s, (i+1)*step + (overlap/2 if i<k-1 else 0.0))
            out.append((start, end))
        return out

    # 是否啟用自動分段：預設為單段處理，以降低錯誤風險
    if not auto_chunk_audio:
        k = 1
    else:
        # Metal 上單段最優；CPU 視長度切 3~5 段
        if device_now == "metal":
            k = 1
        else:
            if total_sec <= 20*60:
                k = 1
            elif total_sec <= 40*60:
                k = 3
            elif total_sec <= 60*60:
                k = 4
            else:
                k = 5

    chunks = make_chunks(total_sec, k)

    # 匯出每段 wav
    chunk_paths = []
    for i,(a,b) in enumerate(chunks, 1):
        p = os.path.join(tmp_dir, f"_tmp_{ts}_{i}.wav")
        export_seg_to_wav(seg[a*1000:b*1000], p)
        chunk_paths.append((p,a,b))

    if k == 1:
        device_msg = f"裝置：{device_now}；整段轉寫；並行：否"
    else:
        device_msg = f"裝置：{device_now}；切段 {k}（含 {overlap}s 重疊）；並行：否"
    combined_status = f"{device_msg}｜📝 正在轉錄音檔…"
    status.info(combined_status)
    st.session_state["transcribe_status_message"] = combined_status

    final_texts, final_srts = [], []
    total_elapsed = 0.0

    if k == 1:
        # 單段即時顯示
        with st.container():
            st.subheader("轉寫進度")
            one_live = st.empty()
            one_stats = st.empty()
        progress_area = realtime_progress_area
        seg_start_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"開始轉換時間點：{seg_start_ts}")
        status.info("開始轉寫…")
        st.session_state["transcribe_status_message"] = "開始轉寫…"
        if asr_backend == "whisper.cpp":
            final_text, final_srt, _, elapsed = transcribe_one_whispercpp(
                os.path.join(tmp_dir, f"_tmp_{ts}_1.wav"),
                cli_path=whispercpp_cli,
                model_path=whispercpp_model,
                language="zh",
                beam_size=beam_size,
                initial_prompt=combined_prompt,
                punc_rule=punc_rule,
                ui_area=one_live,
                progress_area=progress_area,
                stats_area=one_stats,
                threads=whispercpp_threads,
                time_offset_sec=chunks[0][0],
                total_sec_for_progress=(chunks[0][1]-chunks[0][0]),
                lexical_rules=LEXICAL_RULES,
                punc_settings=CURRENT_PUNCT_SETTINGS,
                progress_label="whisper.cpp 轉寫中…",
            )
        else:
            final_text, final_srt, _, elapsed = transcribe_one(
                os.path.join(tmp_dir, f"_tmp_{ts}_1.wav"), model,
                language="zh", vad_filter=vad, beam_size=beam_size,
                initial_prompt=combined_prompt, punc_rule=punc_rule,
                vad_parameters=vad_params,
                ui_area=one_live, progress_area=progress_area, stats_area=one_stats,
                time_offset_sec=chunks[0][0], total_sec_for_progress=(chunks[0][1]-chunks[0][0]),
                lexical_rules=LEXICAL_RULES,
                punc_settings=CURRENT_PUNCT_SETTINGS,
                progress_label="轉寫中…",
            )
        total_elapsed = elapsed
        final_texts.append(final_text)
        final_srts.append(final_srt)
    else:
        # 順序分段處理（不嘗試並行）
        st.subheader("順序分段處理")
        bar = st.progress(0.0, text="初始化…")
        span_total = sum(b-a for _,a,b in chunk_paths)
        span_done = 0.0
        t0 = time.time()
        for i,(p,a,b) in enumerate(chunk_paths, 1):
            seg_start_ts = time.strftime("%Y-%m-%d %H:%M:%S")
            st.caption(f"分段 {i} 開始轉換時間點：{seg_start_ts}")
            part_live = st.empty(); part_prog = st.empty(); part_stats = st.empty()
            if asr_backend == "whisper.cpp":
                text_i, srt_i, _, _ = transcribe_one_whispercpp(
                    p,
                    cli_path=whispercpp_cli,
                    model_path=whispercpp_model,
                    language="zh",
                    beam_size=beam_size,
                    initial_prompt=combined_prompt,
                    punc_rule=punc_rule,
                    ui_area=part_live,
                    progress_area=part_prog,
                    stats_area=part_stats,
                    threads=whispercpp_threads,
                    time_offset_sec=a,
                    total_sec_for_progress=(b-a),
                    lexical_rules=LEXICAL_RULES,
                    punc_settings=CURRENT_PUNCT_SETTINGS,
                )
            else:
                text_i, srt_i, _, _ = transcribe_one(
                    p, model, language="zh", vad_filter=vad, beam_size=beam_size,
                    initial_prompt=combined_prompt, punc_rule=punc_rule,
                    vad_parameters=vad_params,
                    ui_area=part_live, progress_area=part_prog, stats_area=part_stats,
                    time_offset_sec=a, total_sec_for_progress=(b-a),
                    lexical_rules=LEXICAL_RULES,
                    punc_settings=CURRENT_PUNCT_SETTINGS,
                )
            final_texts.append(text_i)
            final_srts.append(srt_i)
            span_done += (b - a)
            elapsed = time.time() - t0
            # 以目前平均速率估計剩餘（整體）
            done_ratio = max(1e-6, span_done / max(span_total, 1e-6))
            eta_txt = ""
            if done_ratio > 0.02:
                est_total = elapsed / done_ratio
                est_remain = max(0.0, est_total - elapsed)
                finish_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + est_remain))
                eta_txt = f"｜預計剩餘 {fmt_dur(est_remain)}｜預計完成 {finish_ts}"
            bar.progress(min(1.0, span_done/span_total), text=f"完成 {i}/{k} 段；耗時 {fmt_dur(elapsed)}{eta_txt}")
        total_elapsed = time.time() - t0

    # 合併

    # 🟢替換後開始
    # === 最終整合階段 ===
    # 合併所有分段 SRT（供下載與時間軸使用）
    final_srt = "\n".join(final_srts)

    # 直接合併每段的最終文字（這些文字在 transcribe_one 裡已加好標點與換行）
    final_text = "\n".join(final_texts)

    # 若開啟 reflow，再統一重排
    if 'reflow_enable' in globals() and reflow_enable:
        final_text = reflow_text(final_text, max_len=reflow_width)

    # === 自動保存到本地 ===
    # 自動保存到本地
    if uploaded_name:
        base_filename = st.session_state.get("current_base_filename") or os.path.splitext(uploaded_name)[0]
        txt_path = os.path.join(tmp_dir, f"{base_filename}.txt")
        srt_path = os.path.join(tmp_dir, f"{base_filename}.srt")
        with open(txt_path, "w", encoding="utf-8") as f_txt:
            f_txt.write(final_text)
        with open(srt_path, "w", encoding="utf-8") as f_srt:
            f_srt.write(final_srt)

        # 清理同名的增量檔（base_時間戳.*），避免目錄內出現重複版本
        for pattern in (f"{base_filename}_*.txt", f"{base_filename}_*.srt"):
            for stale in Path(tmp_dir).glob(pattern):
                stale_path = str(stale)
                if stale_path in (txt_path, srt_path):
                    continue
                try:
                    os.remove(stale_path)
                except OSError:
                    pass

        st.sidebar.success(f"已自動保存到本地：{txt_path} 和 {srt_path}")



    # 去除標點後多餘空白，再換行
    line_break_version = re.sub(r"\s*([。！？])\s*", r"\1\n", final_text)
    line_break_version = re.sub(r"\n+", "\n", line_break_version).strip()

    if show_line_mode == "每句換行":
        display_text = line_break_version
    else:
        display_text = final_text.replace("\n", " ")



    # === 顯示結果與耗時 ===
    final_box.markdown(display_text.replace("\n", "  \n"))
    success_msg = f"完成；總耗時 {fmt_dur(total_elapsed)}"
    st.session_state["transcribe_success_message"] = success_msg

    # === 狀態重置 ===
    st.session_state["transcribing"] = False
    st.session_state["start_transcribe_pending"] = False
    st.session_state["transcribe_status_message"] = ""

    # 安全重置按鈕狀態
    if "run_button_primary" in st.session_state:
        del st.session_state["run_button_primary"]

    # === 快取結果 ===
    st.session_state["last_txt"] = final_text
    st.session_state["last_srt"] = final_srt

    # === 清理暫存檔 ===
    for p_tuple in [(base_path, None, None)] + chunk_paths:
        try:
            os.remove(p_tuple[0])
        except Exception:
            pass

    _trigger_rerun()


    # === 統一顯示區（無論是否剛轉完錄音） ===
if not IS_WORKER and st.session_state.get("last_txt"):

    st.markdown("---")
    st.subheader("轉錄結果")

    # === 下載按鈕（保持顯示） ===
    with st.expander("📥 下載檔案", expanded=True):
        dl_cols = st.columns(2)
        new_txt = st.session_state["last_txt"].encode("utf-8")
        new_srt = st.session_state.get("last_srt", "").encode("utf-8")
        with dl_cols[0]:
            base_name = st.session_state.get("current_base_filename", "transcript")
            ts_val = st.session_state.get("current_ts", "")
            fname_txt = f"{base_name}_{ts_val}.txt" if ts_val else f"{base_name}.txt"
            st.download_button("下載 TXT", data=new_txt, file_name=fname_txt, key="dl_txt")
        with dl_cols[1]:
            base_name = st.session_state.get("current_base_filename", "transcript")
            ts_val = st.session_state.get("current_ts", "")
            fname_srt = f"{base_name}_{ts_val}.srt" if ts_val else f"{base_name}.srt"
            st.download_button("下載 SRT", data=new_srt, file_name=fname_srt, key="dl_srt")

    # 根據顯示模式渲染內容
    cached_text = st.session_state["last_txt"]
    line_break_version = re.sub(r"\s*([。！？])\s*", r"\1\n", cached_text)
    line_break_version = re.sub(r"\n+", "\n", line_break_version).strip()
    display_text = (
        line_break_version
        if st.session_state.get("show_line_mode", "每句換行") == "每句換行"
        else cached_text.replace("\n", " ")
    )

    st.markdown(display_text.replace("\n", "  \n"))

    # === 側邊欄一鍵複製 ===
    with st.sidebar:
        show_line_mode = st.session_state.get("show_line_mode", "每句換行")
        text_to_copy = (
            re.sub(r"\s*([。！？])\s*", r"\1\n", st.session_state["last_txt"])
            if show_line_mode == "每句換行"
            else st.session_state["last_txt"].replace("\n", " ")
        )
        if st.button("📋 顯示可複製內容", key="copy_btn"):
            st.session_state["copy_buffer"] = text_to_copy
            st.toast("已顯示可複製內容，請在下方區塊手動複製。", icon="📋")

        if "copy_buffer" in st.session_state:
            st.text_area("目前顯示內容（可全選後複製）", st.session_state["copy_buffer"], height=300)
