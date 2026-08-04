# fw_streamlit.py
import streamlit as st
from opencc import OpenCC
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import io, os, time, json, hashlib, subprocess, shutil, tempfile
import importlib.util
import re
from typing import List, Tuple, Dict, Any, Pattern
from dataclasses import dataclass

from pathlib import Path

# Detect worker mode for child processes
IS_WORKER = os.environ.get("ASR_WORKER") == "1"

# --- Domain knowledge defaults ---
CONFIG_DEFAULT_PATH = Path(__file__).resolve().parent / "config" / "domain_defaults.json"
DOMAIN_TERM_CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "domain_terms_v2.json"
DOMAIN_CACHE_TERMS_PER_FILE = 5000
DOMAIN_CACHE_MAX_FILES = 512
# whisper 的 initial prompt 硬上限是 n_text_ctx/2 = 224 個 token（whisper.cpp 的 `--prompt`
# 說明就寫這個數字）。中文大致一個字一個 token，超過的部分模型不會看到，直接被丟掉。
# 先前這裡設 2400 字，等於九成以上的詞彙送進去就被截斷，還會把排在前面的專有名詞擠掉。
WHISPER_PROMPT_TOKEN_LIMIT = 224
WHISPER_PROMPT_CHAR_BUDGET = 200
DOMAIN_PROMPT_CHAR_LIMIT = WHISPER_PROMPT_CHAR_BUDGET
LARGE_UPLOAD_THRESHOLD_BYTES = 200 * 1024 * 1024


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


def _is_pyannote_installed() -> bool:
    try:
        return importlib.util.find_spec("pyannote.audio") is not None
    except ModuleNotFoundError:
        return False


DEFAULTS_CONFIG = _load_defaults_config()

WHISPERCPP_DEFAULT_CLI = str(DEFAULTS_CONFIG.get("whispercpp_cli", "") or os.environ.get("WHISPERCPP_CLI", ""))
WHISPERCPP_DEFAULT_MODEL = str(DEFAULTS_CONFIG.get("whispercpp_model", "") or os.environ.get("WHISPERCPP_MODEL", ""))
WHISPERCPP_COMMON_DIRS = [
    Path.home() / "Documents" / "whisper.cpp",
    Path.home() / "whisper.cpp",
    Path("/opt/homebrew/opt/whisper-cpp"),
]
WHISPERCPP_OPTIMAL_MODEL_NAMES = (
    # large-v3-turbo 的解碼層只有 4 層，速度接近 medium，中文準確度高一截；
    # q5_0 量化後檔案大小（約 574MB）與 medium-q5_0（539MB）差不多，M3 Air 16GB 吃得下。
    "ggml-large-v3-turbo-q5_0.bin",
    "ggml-large-v3-turbo-q8_0.bin",
    "ggml-medium-q5_0.bin",
    "ggml-medium-q8_0.bin",
    "ggml-medium.bin",
    "ggml-small-q8_0.bin",
    "ggml-small.bin",
)
WHISPERCPP_VAD_MODEL_NAMES = (
    "ggml-silero-v5.1.2.bin",
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
    return PunctuationSettings(comma_ms=comma_ms, period_ms=period_ms)


DEFAULT_PUNCTUATION_SETTINGS = _load_default_punctuation_settings(DEFAULTS_CONFIG)
CURRENT_PUNCT_SETTINGS: PunctuationSettings = DEFAULT_PUNCTUATION_SETTINGS


@dataclass(frozen=True)
class SpeakerSegment:
    start: float
    end: float
    label: str


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


def _detect_whispercpp_vad_model() -> str:
    """找 silero VAD 模型；whisper.cpp 的 --vad 需要它，缺了會直接報錯。"""
    for base in WHISPERCPP_COMMON_DIRS:
        model_dir = base / "models"
        for name in WHISPERCPP_VAD_MODEL_NAMES:
            cand = model_dir / name
            if cand.exists():
                return str(cand)
        if model_dir.exists():
            candidates = sorted(model_dir.glob("ggml-silero-*.bin"))
            candidates = [p for p in candidates if not p.name.startswith("for-tests-")]
            if candidates:
                return str(candidates[-1])
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
    asr_backend = "whisper.cpp"
    beam_size = 1
    whispercpp_cli = WHISPERCPP_DEFAULT_CLI or detected_whispercpp_cli
    whispercpp_model = WHISPERCPP_DEFAULT_MODEL or detected_whispercpp_model
    whispercpp_threads = max(1, min(6, (os.cpu_count() or 4) - 2))
    punc_rule = True
    default_punc = DEFAULT_PUNCTUATION_SETTINGS
    comma_ms = default_punc.comma_ms
    period_ms = default_punc.period_ms
    auto_chunk_audio = False
    clean_audio_for_asr = True
    with sidebar.expander("進階設定", expanded=False):
        st.caption("STT 後端固定為 whisper.cpp。")
        whispercpp_cli = st.text_input("CLI 路徑", whispercpp_cli)
        whispercpp_model = st.text_input("模型檔路徑", whispercpp_model)
        whispercpp_threads = st.slider("CPU threads", 1, max(1, os.cpu_count() or 8), min(whispercpp_threads, max(1, os.cpu_count() or 8)), 1)
        st.caption("本機最佳性能：使用 Metal 編譯的 whisper.cpp；M3 Air 16GB 建議 large-v3-turbo-q5_0，beam=1。medium-q5_0 仍可用。")
        if _is_whispercpp_optimal(whispercpp_cli, whispercpp_model, asr_backend, beam_size):
            st.success("最佳配置已啟用")
        else:
            st.warning("目前不是最佳配置；請使用 whisper.cpp、medium-q5_0、beam=1。")
        punc_rule = st.checkbox("依停頓自動補標點", punc_rule)
        with st.expander("標點靈敏度", expanded=False):
            comma_ms = st.slider("逗號門檻 (ms)", 80, 400, comma_ms, 10)
            period_ms = st.slider("句號門檻 (ms)", 120, 800, period_ms, 10)
        clean_audio_for_asr = st.checkbox(
            "轉寫前清理底噪並正規化音量",
            clean_audio_for_asr,
            help="建議 WAV/長錄音開啟。不刪除靜音、不改變音訊長度，只做濾波、降噪與音量正規化。",
        )
        auto_chunk_audio = st.checkbox("長錄音自動分段轉寫", auto_chunk_audio)
        _vad_model_path = _detect_whispercpp_vad_model()
        st.checkbox(
            "用 VAD 切語音段（建議開）",
            value=True,
            key="use_vad",
            disabled=not _vad_model_path,
            help=(
                "whisper.cpp 內建 silero VAD。靜音段是重複幻覺的主要來源，開了可以在轉寫前就避掉，"
                "不必全靠事後的重複偵測救。"
                if _vad_model_path
                else "找不到 silero VAD 模型。到 whisper.cpp 目錄跑 bash models/download-vad-model.sh silero-v5.1.2"
            ),
        )
    sidebar.markdown("---")
    enable_speaker_diarization = sidebar.checkbox(
        "辨識說話者",
        False,
        help="需要另裝 pyannote.audio 並提供 Hugging Face token。只有明確偵測到多位說話者時才加標籤。",
    )
    speaker_count_choice = sidebar.selectbox(
        "說話者人數",
        ["自動", "2", "3", "4", "5", "6"],
        index=0,
        disabled=not enable_speaker_diarization,
    )
    speaker_hf_token = sidebar.text_input(
        "Hugging Face token",
        value=os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or "",
        type="password",
        disabled=not enable_speaker_diarization,
        help="pyannote speaker-diarization 模型需要授權 token；也可用 HUGGINGFACE_TOKEN 或 HF_TOKEN 環境變數。",
    )
    if enable_speaker_diarization and not _is_pyannote_installed():
        sidebar.warning("目前 Python 環境未安裝 pyannote.audio，說話者辨識會被略過。")
    CURRENT_PUNCT_SETTINGS = PunctuationSettings(
        comma_ms=int(comma_ms),
        period_ms=int(period_ms),
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
    terms_topk = sidebar.slider(
        "最多帶入詞彙數", 100, 5000, 150, 50,
        help=f"提示最後只留得下約 {WHISPER_PROMPT_CHAR_BUDGET} 字，抽再多詞也送不進去，只是多花抽詞時間。",
    )
    auto_use_domain = sidebar.checkbox("在轉寫時自動帶入領域詞彙", True)

    init_prompt = sidebar.text_input("初始提示（可留空）", "請使用繁體中文輸出。")
    combined_domain_files = list(preset_domain_files)
    if domain_files:
        combined_domain_files.extend(domain_files)

cc = OpenCC("s2t")

# Worker mode needs these globals because the Streamlit sidebar is not initialized.
if not IS_WORKER:
    pass
else:
    CORRECTION_MAP = dict(DEFAULT_CORRECTION_MAP)
    REGEX_CORRECTION_RULES = _compile_regex_correction_rules(DEFAULT_REGEX_CORRECTION_MAP)
    CURRENT_PUNCT_SETTINGS = DEFAULT_PUNCTUATION_SETTINGS

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


def linebreak_after_punctuation(s: str) -> str:
    """Format display text so every clause/sentence punctuation starts a new line."""
    if not s:
        return s
    # Normalize surrounding whitespace first so live Markdown rendering is stable.
    s = re.sub(r"\s*\n+\s*", " ", s)
    out = re.sub(r"\s*([，。、；：！？…．])\s*", r"\1\n", s)
    return re.sub(r"\n+", "\n", out).strip()


def markdown_linebreaks(s: str) -> str:
    return (s or "").replace("\n", "  \n")

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

_RE_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")
_RE_EN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")
_CJK_TERM_MIN_LEN = 2
_CJK_TERM_MAX_LEN = 10
_CJK_BAD_BOUNDARY = set("的一是在有和與及或而就都也很可把被於之其該各等並跟")

def _domain_file_digest(name: str, data: bytes) -> str:
    return hashlib.sha256((name or "").encode("utf-8", errors="ignore") + b"\0" + data).hexdigest()


def _load_domain_term_cache() -> Dict[str, Any]:
    try:
        if DOMAIN_TERM_CACHE_PATH.exists():
            data = json.loads(DOMAIN_TERM_CACHE_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _save_domain_term_cache(cache: Dict[str, Any]) -> None:
    try:
        if len(cache) > DOMAIN_CACHE_MAX_FILES:
            items = sorted(
                cache.items(),
                key=lambda item: int(item[1].get("used_at", 0)) if isinstance(item[1], dict) else 0,
                reverse=True,
            )
            cache = dict(items[:DOMAIN_CACHE_MAX_FILES])
        DOMAIN_TERM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        DOMAIN_TERM_CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    except Exception:
        pass


def _term_stats_from_text(text: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    counts: Dict[str, int] = {}
    seen_case: Dict[str, str] = {}
    if not text:
        return counts, seen_case
    for m in _RE_CJK_RUN.finditer(text):
        run = m.group(0)
        run_len = len(run)
        for n in range(_CJK_TERM_MIN_LEN, min(_CJK_TERM_MAX_LEN, run_len) + 1):
            for idx in range(0, run_len - n + 1):
                tok = run[idx:idx + n]
                if tok[0] in _CJK_BAD_BOUNDARY or tok[-1] in _CJK_BAD_BOUNDARY:
                    continue
                counts[tok] = counts.get(tok, 0) + max(1, n - 1)
    for term in set(DEFAULT_CORRECTION_MAP.values()):
        if not term or len(term) < 2:
            continue
        hits = text.count(term)
        if hits:
            counts[term] = counts.get(term, 0) + hits * max(20, len(term) * 3)
    for m in _RE_EN.finditer(text):
        raw = m.group(0)
        key = raw.lower()
        if key not in seen_case:
            seen_case[key] = raw
        counts[key] = counts.get(key, 0) + max(1, min(len(raw), 12) // 3)
    return counts, seen_case


def _trim_term_stats(counts: Dict[str, int], seen_case: Dict[str, str]) -> Tuple[Dict[str, int], Dict[str, str]]:
    items = []
    for key, count in counts.items():
        shown = seen_case.get(key, key)
        score = count * max(1, min(len(shown), _CJK_TERM_MAX_LEN))
        items.append((score, count, len(shown), shown, key))
    items.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    keep = {key for _, _, _, _, key in items[:DOMAIN_CACHE_TERMS_PER_FILE]}
    return {key: counts[key] for key in keep}, {key: value for key, value in seen_case.items() if key in keep}


def _terms_from_stats(counts: Dict[str, int], seen_case: Dict[str, str], top_k: int) -> List[str]:
    items = []
    for key, count in counts.items():
        shown = seen_case.get(key, key)
        score = count * max(1, min(len(shown), _CJK_TERM_MAX_LEN))
        items.append((score, count, len(shown), shown))
    items.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))
    terms = []
    for _, _, _, shown in items:
        if shown not in terms:
            terms.append(shown)
        if len(terms) >= max(10, top_k):
            break
    return terms


def extract_terms(text: str, top_k: int = 60) -> List[str]:
    if not text:
        return []
    counts, seen_case = _term_stats_from_text(text)
    return _terms_from_stats(counts, seen_case, top_k)

def fit_prompt_to_budget(prompt: str, budget: int = WHISPER_PROMPT_CHAR_BUDGET) -> str:
    """把 prompt 修到 whisper 真的讀得進去的長度，並且切在詞的邊界上。"""
    prompt = (prompt or "").strip("， ")
    if len(prompt) <= budget:
        return prompt
    kept: List[str] = []
    used = 0
    for term in prompt.split("，"):
        term = term.strip()
        if not term:
            continue
        cost = len(term) + (1 if kept else 0)
        if used + cost > budget:
            break
        kept.append(term)
        used += cost
    if not kept:
        return prompt[:budget]
    return "，".join(kept)


def build_domain_prompt(files, top_k: int = 60) -> tuple[str, List[str]]:
    combined_counts: Dict[str, int] = {}
    combined_seen_case: Dict[str, str] = {}
    cache = _load_domain_term_cache()
    cache_changed = False
    for f in files or []:
        name = getattr(f, "name", "") or ""
        data = f.read()
        if hasattr(f, "seek"):
            try:
                f.seek(0)
            except Exception:
                pass
        digest = _domain_file_digest(name, data)
        cached = cache.get(digest)
        if isinstance(cached, dict) and isinstance(cached.get("counts"), dict):
            counts = {str(k): int(v) for k, v in cached.get("counts", {}).items() if v}
            seen_case = {str(k): str(v) for k, v in cached.get("seen_case", {}).items()}
            cached["used_at"] = int(time.time())
            cache_changed = True
        else:
            counts = {}
            seen_case = {}
            text = ""
            try:
                if name.lower().endswith(".pdf"):
                    text = _read_pdf_bytes(name, data)
                elif name.lower().endswith(".md"):
                    text = _md_to_text(data.decode("utf-8", errors="ignore"))
                else:
                    text = data.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
            if text:
                counts, seen_case = _term_stats_from_text(text)
                counts, seen_case = _trim_term_stats(counts, seen_case)
                cache[digest] = {
                    "name": name,
                    "size": len(data),
                    "used_at": int(time.time()),
                    "counts": counts,
                    "seen_case": seen_case,
                }
                cache_changed = True
        if not counts:
            continue
        weight = 1
        for hint in DOMAIN_PRIORITY_HINTS:
            if hint and hint in name:
                weight = 5
                break
        for key, count in counts.items():
            combined_counts[key] = combined_counts.get(key, 0) + int(count) * weight
        for key, value in seen_case.items():
            combined_seen_case.setdefault(key, value)
    if cache_changed:
        _save_domain_term_cache(cache)
    terms = _terms_from_stats(combined_counts, combined_seen_case, top_k=top_k)
    # 以全形逗號分隔；長度切在 whisper 讀得進去的範圍內，且不切斷詞
    prompt = fit_prompt_to_budget("，".join(terms), DOMAIN_PROMPT_CHAR_LIMIT)
    return prompt, terms

def audio_info_from_bytes(b: bytes, filename: str|None=None) -> Tuple[AudioSegment, float]:
    fmt = None
    if filename and "." in filename:
        fmt = filename.rsplit(".",1)[-1].lower()
    suffix = f".{fmt}" if fmt else ""
    tmp_path = ""
    wav_tmp_path = ""
    try:
        # Some M4A/MP4 files need seeking to read the moov atom reliably.
        # Decoding from a real file avoids ffmpeg cache:pipe seekback failures.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(b)
            tmp_path = tmp.name
        try:
            seg = AudioSegment.from_file(tmp_path, format=fmt)
        except CouldntDecodeError as first_exc:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
                wav_tmp_path = wav_tmp.name
            ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
            result = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-y",
                    "-err_detect", "ignore_err",
                    "-i", tmp_path,
                    "-vn",
                    "-ac", "1",
                    "-ar", "16000",
                    "-sample_fmt", "s16",
                    wav_tmp_path,
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0 and Path(wav_tmp_path).exists() and Path(wav_tmp_path).stat().st_size > 44:
                seg = AudioSegment.from_file(wav_tmp_path, format="wav")
            else:
                detail = (result.stderr or result.stdout or "").strip()
                raise CouldntDecodeError(
                    f"{first_exc}\n\nffmpeg direct WAV fallback failed:\n{detail[-2000:]}"
                ) from first_exc
    finally:
        for path in (tmp_path, wav_tmp_path):
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
    dur = seg.duration_seconds
    return seg, dur


def audio_info_from_path(path: str) -> Tuple[AudioSegment, float]:
    source_path = str(Path(path).expanduser())
    fmt = Path(source_path).suffix.lower().lstrip(".") or None
    wav_tmp_path = ""
    try:
        try:
            seg = AudioSegment.from_file(source_path, format=fmt)
        except CouldntDecodeError as first_exc:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
                wav_tmp_path = wav_tmp.name
            ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
            result = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-y",
                    "-err_detect", "ignore_err",
                    "-i", source_path,
                    "-vn",
                    "-ac", "1",
                    "-ar", "16000",
                    "-sample_fmt", "s16",
                    wav_tmp_path,
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0 and Path(wav_tmp_path).exists() and Path(wav_tmp_path).stat().st_size > 44:
                seg = AudioSegment.from_file(wav_tmp_path, format="wav")
            else:
                detail = (result.stderr or result.stdout or "").strip()
                raise CouldntDecodeError(
                    f"{first_exc}\n\nffmpeg direct WAV fallback failed:\n{detail[-2000:]}"
                ) from first_exc
    finally:
        if wav_tmp_path:
            try:
                os.remove(wav_tmp_path)
            except OSError:
                pass
    return seg, seg.duration_seconds


def maybe_compress_large_upload(
    b: bytes,
    filename: str | None,
    work_dir: str,
    ts: int,
    status_area=None,
) -> Tuple[bytes, str | None, bool, int, int]:
    """Allow large uploads to continue without altering the original audio."""
    original_size = len(b)
    if original_size > LARGE_UPLOAD_THRESHOLD_BYTES and status_area is not None:
        status_area.info("音檔超過 200MB，已用 Streamlit 設定允許上傳；將直接使用原檔處理，不再壓縮。")
    return b, filename, False, original_size, original_size

def export_seg_to_wav(seg: AudioSegment, path: str, clean_audio: bool = False):
    normalized = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    if not clean_audio:
        normalized.export(path, format="wav")
        return

    raw_path = str(Path(path).with_suffix(".raw.wav"))
    try:
        normalized.export(raw_path, format="wav")
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-y",
                "-i", raw_path,
                "-af",
                "highpass=f=80,lowpass=f=7600,afftdn=nf=-25,dynaudnorm=f=250:g=15:p=0.95",
                "-ac", "1",
                "-ar", "16000",
                "-sample_fmt", "s16",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode != 0 or not Path(path).exists() or Path(path).stat().st_size <= 44:
            normalized.export(path, format="wav")
    finally:
        try:
            if os.path.exists(raw_path):
                os.remove(raw_path)
        except OSError:
            pass

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

REPEAT_RECOVERY_THRESHOLD = 4
REPEAT_RECOVERY_BACKTRACK_SEC = 1.0
REPEAT_RECOVERY_MAX_RESTARTS_PER_LOCATION = 2
REPEAT_RECOVERY_LOCATION_BUCKET_SEC = 1.0
REPEAT_RECOVERY_SKIP_SEC = 8.0
REPEAT_RECOVERY_WINDOW_BLOCKS = 6
REPEAT_RECOVERY_WINDOW_REPEAT_THRESHOLD = 4
REPEAT_RECOVERY_ARTIFACT_THRESHOLD = 3
REPEAT_ARTIFACT_MIN_PHRASE_REPEATS = 4
REPEAT_ARTIFACT_MIN_COVERAGE = 0.55
SUBTITLE_ARTIFACT_PATTERNS = (
    re.compile(r"字幕.{0,12}(不明顯|無法|不能|太多|翻譯|使用|接受)"),
    re.compile(r"(不明顯|無法|不能).{0,12}(翻譯|使用|接受)"),
    re.compile(r"請勿(?:使)?用"),
    re.compile(r"不要(?:輸出|使用|翻譯)"),
)
LOW_CONTENT_ARTIFACT_PATTERNS = (
    re.compile(r"^[※＊*#·・]+$"),
    re.compile(r"^(?:請勿(?:使)?用|不要(?:輸出|使用|翻譯))[※＊*#·・]*$"),
)


def _repeat_detection_key(text: str) -> str:
    return re.sub(r"[\s，。、；：！？…．,.!?;:※＊*#·・\-—_]+", "", text or "").strip()


def _repeat_recovery_location_key(global_start_sec: float) -> int:
    return int(round(global_start_sec / max(REPEAT_RECOVERY_LOCATION_BUCKET_SEC, 1e-6)))


def _repetition_artifact_reason(text: str) -> str:
    """Detect hallucinated subtitle loops before they enter live/output buffers."""
    raw = text or ""
    compact_raw = re.sub(r"[\s，。、；：！？…．,.!?;:「」『』（）()\[\]]+", "", raw)
    if compact_raw:
        for pattern in LOW_CONTENT_ARTIFACT_PATTERNS:
            if pattern.fullmatch(compact_raw):
                return "疑似低資訊提示符號"

    if "字幕" in raw or "翻譯" in raw or "請勿" in raw or "不要" in raw:
        for pattern in SUBTITLE_ARTIFACT_PATTERNS:
            if pattern.search(compact_raw):
                return "疑似字幕/翻譯提示語"

    key = _repeat_detection_key(text)
    if len(key) < 12:
        return ""

    for size in range(4, min(16, len(key) // 2) + 1):
        counts: Dict[str, int] = {}
        for idx in range(0, len(key) - size + 1):
            gram = key[idx:idx + size]
            counts[gram] = counts.get(gram, 0) + 1
        phrase, repeats = max(counts.items(), key=lambda item: item[1])
        coverage = (len(phrase) * repeats) / max(len(key), 1)
        if repeats >= REPEAT_ARTIFACT_MIN_PHRASE_REPEATS and coverage >= REPEAT_ARTIFACT_MIN_COVERAGE:
            return f"短語「{phrase[:12]}」重複 {repeats} 次"

    if key.count("字幕") >= REPEAT_ARTIFACT_MIN_PHRASE_REPEATS and len(set(key)) <= 24:
        return "疑似字幕提示語重複"
    return ""


def _is_immediate_repetition_artifact(reason: str) -> bool:
    return reason.startswith("短語") or reason == "疑似字幕提示語重複"


def _transcript_loop_signal(blocks: List[Tuple[float, float, str]]) -> Tuple[str, float]:
    """Return a recovery reason/start time when recent output appears stuck in a loop."""
    if not blocks:
        return "", 0.0

    last_start, _, last_text = blocks[-1]
    last_reason = _repetition_artifact_reason(last_text)
    if last_reason and _is_immediate_repetition_artifact(last_reason):
        return last_reason, last_start

    recent = blocks[-REPEAT_RECOVERY_WINDOW_BLOCKS:]
    artifact_hits: List[Tuple[float, str]] = []
    key_hits: Dict[str, List[float]] = {}
    for start, _, text in recent:
        reason = _repetition_artifact_reason(text)
        if reason:
            artifact_hits.append((start, reason))
        key = _repeat_detection_key(text)
        if len(key) >= 4:
            key_hits.setdefault(key, []).append(start)

    if len(artifact_hits) >= REPEAT_RECOVERY_ARTIFACT_THRESHOLD:
        first_start = artifact_hits[0][0]
        return f"短時間出現無意義提示語 {len(artifact_hits)} 次", first_start

    repeated = [
        (key, starts)
        for key, starts in key_hits.items()
        if len(starts) >= REPEAT_RECOVERY_WINDOW_REPEAT_THRESHOLD
    ]
    if repeated:
        key, starts = max(repeated, key=lambda item: len(item[1]))
        return f"短時間重複「{key[:12]}」{len(starts)} 次", starts[0]

    return "", 0.0


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _export_tail_wav(source_path: str, dest_path: str, start_sec: float) -> None:
    audio = AudioSegment.from_file(source_path)
    start_ms = max(0, int(start_sec * 1000))
    audio[start_ms:].set_channels(1).set_frame_rate(16000).set_sample_width(2).export(dest_path, format="wav")


def _cleanup_whispercpp_outputs(out_prefix: str) -> None:
    for stale in Path(out_prefix).parent.glob(Path(out_prefix).name + ".*"):
        try:
            stale.unlink()
        except OSError:
            pass


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


@st.cache_resource(show_spinner=False)
def _load_pyannote_pipeline(hf_token: str):
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except Exception as exc:
        raise RuntimeError("尚未安裝 pyannote.audio。請先執行：pip install pyannote.audio") from exc

    try:
        return Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    except TypeError:
        return Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )


def _run_speaker_diarization(
    audio_path: str,
    hf_token: str,
    requested_speakers: int | None,
) -> List[SpeakerSegment]:
    token = (hf_token or "").strip()
    if not token:
        raise RuntimeError("未提供 Hugging Face token，無法啟用說話者辨識。")

    pipeline = _load_pyannote_pipeline(token)
    kwargs: Dict[str, Any] = {}
    if requested_speakers and requested_speakers >= 2:
        kwargs["num_speakers"] = requested_speakers
    diarization = pipeline(audio_path, **kwargs)

    first_seen: Dict[str, float] = {}
    raw_segments: List[Tuple[float, float, str]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = float(getattr(turn, "start", 0.0))
        end = float(getattr(turn, "end", start))
        if end <= start:
            continue
        raw_label = str(speaker)
        first_seen.setdefault(raw_label, start)
        raw_segments.append((start, end, raw_label))

    ordered_labels = {
        raw_label: f"說話者 {idx}"
        for idx, raw_label in enumerate(
            sorted(first_seen, key=lambda label: first_seen[label]),
            1,
        )
    }
    return [
        SpeakerSegment(start=start, end=end, label=ordered_labels[raw_label])
        for start, end, raw_label in sorted(raw_segments, key=lambda item: (item[0], item[1]))
    ]


def _speaker_durations(segments: List[SpeakerSegment]) -> Dict[str, float]:
    durations: Dict[str, float] = {}
    for seg in segments:
        durations[seg.label] = durations.get(seg.label, 0.0) + max(0.0, seg.end - seg.start)
    return durations


def _format_speaker_summary(segments: List[SpeakerSegment]) -> str:
    durations = _speaker_durations(segments)
    if not durations:
        return "未取得任何說話者片段"
    return "、".join(
        f"{label} {fmt_dur(dur)}"
        for label, dur in sorted(durations.items(), key=lambda item: item[0])
    )


def _has_clear_multiple_speakers(
    segments: List[SpeakerSegment],
    total_sec: float,
    requested_speakers: int | None = None,
) -> bool:
    durations = _speaker_durations(segments)
    if len(durations) < 2:
        return False
    if requested_speakers and requested_speakers >= 2:
        # If the user explicitly picked a speaker count, trust pyannote's
        # assignment instead of hiding labels because one speaker was brief.
        return True
    meaningful = [
        label
        for label, dur in durations.items()
        if dur >= 1.0 and dur / max(total_sec, 1.0) >= 0.005
    ]
    return len(meaningful) >= 2


def _speaker_for_block(
    start: float,
    end: float,
    speaker_segments: List[SpeakerSegment],
    previous_label: str = "",
) -> str:
    block_start = float(start)
    block_end = max(float(end), block_start + 0.01)
    overlaps: Dict[str, float] = {}
    for seg in speaker_segments:
        overlap = max(0.0, min(block_end, seg.end) - max(block_start, seg.start))
        if overlap > 0:
            overlaps[seg.label] = overlaps.get(seg.label, 0.0) + overlap
    if overlaps:
        return max(overlaps.items(), key=lambda item: item[1])[0]

    midpoint = (block_start + block_end) / 2
    nearest = ""
    nearest_dist = float("inf")
    for seg in speaker_segments:
        if seg.start <= midpoint <= seg.end:
            return seg.label
        dist = min(abs(midpoint - seg.start), abs(midpoint - seg.end))
        if dist < nearest_dist:
            nearest = seg.label
            nearest_dist = dist
    return nearest or previous_label or "說話者 1"


def _label_blocks_with_speakers(
    blocks: List[Tuple[float, float, str]],
    speaker_segments: List[SpeakerSegment],
) -> List[Tuple[float, float, str, str]]:
    labeled: List[Tuple[float, float, str, str]] = []
    previous_label = ""
    for start, end, text in blocks:
        label = _speaker_for_block(start, end, speaker_segments, previous_label)
        previous_label = label
        labeled.append((start, end, text, label))
    return labeled


def _srt_from_labeled_blocks(blocks: List[Tuple[float, float, str, str]]) -> str:
    return _srt_blocks_to_srt(
        [(start, end, f"{label}: {text}") for start, end, text, label in blocks]
    )


def _split_long_unpunctuated_unit(text: str, max_chars: int = 24) -> List[str]:
    text = text.strip()
    if not text or len(text) <= max_chars:
        return [text] if text else []

    cue_words = sorted(
        set(LEXICAL_RULES.sentence_starters)
        | set(LEXICAL_RULES.clause_starters)
        | set(LEXICAL_RULES.clause_break_before),
        key=len,
        reverse=True,
    )
    pieces: List[str] = []
    cursor = 0
    while cursor < len(text):
        limit = min(len(text), cursor + max_chars)
        cut_at = 0
        search_start = cursor + max(8, max_chars // 3)
        for idx in range(limit, search_start, -1):
            tail = text[idx:]
            if any(tail.startswith(cue) for cue in cue_words):
                cut_at = idx
                break
        if not cut_at:
            cut_at = limit
        piece = text[cursor:cut_at].strip()
        if piece:
            pieces.append(piece)
        cursor = cut_at
    return pieces


def _split_unpunctuated_text_units(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    if has_any_punctuation(text):
        return [text]

    raw_units = [part.strip() for part in text.split(" ") if part.strip()]
    if len(raw_units) <= 1:
        raw_units = _split_long_unpunctuated_unit(text, max_chars=24)
    units: List[str] = []
    for raw in raw_units:
        units.extend(_split_long_unpunctuated_unit(raw, max_chars=24))
    return units or [text]


def _expand_blocks_for_punctuation(
    blocks: List[Tuple[float, float, str]],
) -> List[Tuple[float, float, str]]:
    expanded: List[Tuple[float, float, str]] = []
    for start, end, text in blocks:
        units = _split_unpunctuated_text_units(text)
        if len(units) <= 1:
            expanded.append((start, end, text))
            continue

        duration = max(0.01, float(end) - float(start))
        gap = min(0.03, duration / max(len(units) * 20, 1))
        available = max(0.01, duration - gap * max(0, len(units) - 1))
        total_chars = sum(max(1, len(unit)) for unit in units)
        cursor = float(start)
        for idx, unit in enumerate(units):
            unit_share = max(1, len(unit)) / max(total_chars, 1)
            unit_duration = max(0.01, available * unit_share)
            unit_end = min(float(end), cursor + unit_duration)
            if idx == len(units) - 1:
                unit_end = float(end)
            expanded.append((cursor, max(unit_end, cursor + 0.01), unit))
            cursor = unit_end + gap
    return expanded


def _punctuate_unpunctuated_units(blocks: List[Tuple[float, float, str]]) -> str:
    return "\n".join(text for _, _, text in _punctuate_unpunctuated_blocks(blocks))


def _punctuate_unpunctuated_blocks(
    blocks: List[Tuple[float, float, str]],
) -> List[Tuple[float, float, str]]:
    units = [(start, end, text.strip()) for start, end, text in blocks if text and text.strip()]
    if not units:
        return []

    sentence_starters = set(LEXICAL_RULES.sentence_starters) | {"好", "那", "接著", "再來", "首先", "第二"}
    soft_sentence_starters = {"雖然", "不過", "但是", "所以", "因此", "好"}
    punctuated: List[Tuple[float, float, str]] = []
    sentence_chars = 0
    clause_count = 0

    for idx, (start, end, unit) in enumerate(units):
        next_unit = units[idx + 1][2] if idx + 1 < len(units) else ""
        sentence_chars += sum(1 for ch in unit if not ch.isspace())
        clause_count += 1

        is_last = idx == len(units) - 1
        next_starts_sentence = any(next_unit.startswith(starter) for starter in sentence_starters)
        next_starts_soft_sentence = any(next_unit.startswith(starter) for starter in soft_sentence_starters)
        long_enough = sentence_chars >= 42
        many_clauses = clause_count >= 3 and sentence_chars >= 28
        should_end = is_last or long_enough or many_clauses or (
            next_starts_sentence and sentence_chars >= 18
        ) or (
            next_starts_soft_sentence and sentence_chars >= 24
        )

        if is_last:
            punc = "。" if not unit.endswith(tuple(SENTENCE_END_PUNCS)) else ""
        elif should_end:
            punc = "。"
        else:
            punc = "，"

        punctuated.append((start, end, unit.rstrip(ANY_END_PUNCS) + punc))
        if punc == "。":
            sentence_chars = 0
            clause_count = 0

    return punctuated


def _output_blocks_from_blocks(
    blocks: List[Tuple[float, float, str]],
    punc_rule: bool,
) -> List[Tuple[float, float, str]]:
    normalized_blocks = [
        (start, end, normalize_zh_punc(apply_common_corrections(cc.convert(text).strip())))
        for start, end, text in blocks
        if text and text.strip() and not _repetition_artifact_reason(text)
    ]
    if not normalized_blocks:
        return []

    expanded = _expand_blocks_for_punctuation(normalized_blocks)
    if len(expanded) > len(normalized_blocks):
        return _punctuate_unpunctuated_blocks(expanded)
    return normalized_blocks


def _display_text_from_blocks(
    blocks: List[Tuple[float, float, str]],
    punc_rule: bool,
    lexical_rules: LexicalRules | None,
    punc_settings: PunctuationSettings | None,
) -> str:
    output_blocks = _output_blocks_from_blocks(blocks, punc_rule)
    if output_blocks and output_blocks != blocks:
        text = "\n".join(text for _, _, text in output_blocks)
    else:
        text = _final_text_from_blocks(blocks, punc_rule, lexical_rules, punc_settings)
    return linebreak_after_punctuation(text)


def _final_text_from_labeled_blocks(
    blocks: List[Tuple[float, float, str, str]],
    punc_rule: bool,
    lexical_rules: LexicalRules | None,
    punc_settings: PunctuationSettings | None,
) -> str:
    parts: List[str] = []
    current_label = ""
    current_blocks: List[Tuple[float, float, str]] = []

    def flush_group() -> None:
        if not current_blocks or not current_label:
            return
        text = _final_text_from_blocks(current_blocks, punc_rule, lexical_rules, punc_settings).strip()
        if text:
            parts.append(f"{current_label}: {text}")

    for start, end, text, label in blocks:
        if label != current_label:
            flush_group()
            current_label = label
            current_blocks = []
        current_blocks.append((start, end, text))
    flush_group()
    return "\n".join(parts)


def apply_speaker_labels_to_outputs(
    audio_path: str,
    srt_text: str,
    hf_token: str,
    requested_speakers: int | None,
    total_sec: float,
    punc_rule: bool,
    lexical_rules: LexicalRules | None,
    punc_settings: PunctuationSettings | None,
    status_area=None,
) -> Tuple[str, str, str]:
    blocks = _read_srt_blocks(srt_text)
    if not blocks:
        return "", srt_text, "沒有可對齊的 SRT 區塊，已略過說話者辨識。"

    if status_area is not None:
        status_area.info("正在辨識說話者…")
    speaker_segments = _run_speaker_diarization(audio_path, hf_token, requested_speakers)
    speaker_summary = _format_speaker_summary(speaker_segments)
    if not _has_clear_multiple_speakers(speaker_segments, total_sec, requested_speakers):
        return "", srt_text, f"未偵測到足夠多位說話者，保留原始轉錄結果。（{speaker_summary}）"

    labeled_blocks = _label_blocks_with_speakers(blocks, speaker_segments)
    labeled_text = _final_text_from_labeled_blocks(
        labeled_blocks,
        punc_rule=punc_rule,
        lexical_rules=lexical_rules,
        punc_settings=punc_settings,
    )
    labeled_srt = _srt_from_labeled_blocks(labeled_blocks)
    speaker_count = len(_speaker_durations(speaker_segments))
    return labeled_text, labeled_srt, f"已辨識 {speaker_count} 位說話者並加上標籤。（{speaker_summary}）"


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
        if text and text.strip() and not _repetition_artifact_reason(text)
    ]
    source_has_punctuation = any(has_any_punctuation(text) for _, _, text in normalized_blocks)

    if punc_rule and not source_has_punctuation:
        original_block_count = len(normalized_blocks)
        normalized_blocks = _expand_blocks_for_punctuation(normalized_blocks)
        if len(normalized_blocks) > max(3, original_block_count):
            return normalize_zh_punc(_punctuate_unpunctuated_units(normalized_blocks))
        final_punc = SmartPunctuator(lexical_cfg)
        for start, end, text in normalized_blocks:
            if text:
                final_punc.add_chunk(text, start, end, punctuation_cfg.comma_threshold, punctuation_cfg.period_threshold)
        final_punc.ensure_terminal()
        joined = "".join(token.strip() for token in final_punc.tokens if token and token.strip())
        return normalize_zh_punc(linebreak_after_punctuation(joined))

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

    status_text = "正在載入模型…"
    prog.progress(0.02, text=f"{status_text}｜耗時 00:00:00")
    accepted_blocks: List[Tuple[float, float, str]] = []
    log_tail: List[str] = []
    supports_fa = _whispercpp_cli_supports_arg(cli, "-fa")
    supports_suppress_nst = _whispercpp_cli_supports_arg(cli, "--suppress-nst")
    no_context_arg = "--no-context" if _whispercpp_cli_supports_arg(cli, "--no-context") else ""
    if not no_context_arg and _whispercpp_cli_supports_arg(cli, "-nc"):
        no_context_arg = "-nc"
    vad_model = _detect_whispercpp_vad_model()
    use_vad = bool(vad_model) and _whispercpp_cli_supports_arg(cli, "--vad")
    if use_vad and not IS_WORKER:
        use_vad = bool(st.session_state.get("use_vad", True))
    supports_entropy_thold = _whispercpp_cli_supports_arg(cli, "--entropy-thold")
    supports_logprob_thold = _whispercpp_cli_supports_arg(cli, "--logprob-thold")
    supports_no_speech_thold = _whispercpp_cli_supports_arg(cli, "--no-speech-thold")
    base_media_path = media_path
    current_media_path = media_path
    current_offset_sec = 0.0
    recovery_paths: List[str] = []
    recoveries = 0
    recovery_attempts_by_location: Dict[int, int] = {}

    while True:
        out_prefix = str(Path(current_media_path).with_suffix("")) + "_whispercpp"
        _cleanup_whispercpp_outputs(out_prefix)

        cmd = [
            cli,
            "-m", model,
            "-f", current_media_path,
            "-l", language or "zh",
            "-osrt",
            "-of", out_prefix,
            "-t", str(max(1, int(threads or 1))),
            "-bs", str(max(1, int(beam_size or 1))),
        ]
        if supports_fa:
            cmd.append("-fa")
        if supports_suppress_nst:
            cmd.append("--suppress-nst")
        if no_context_arg:
            cmd.append(no_context_arg)
        if supports_entropy_thold:
            cmd.extend(["--entropy-thold", "2.4"])
        if supports_logprob_thold:
            cmd.extend(["--logprob-thold", "-1.0"])
        if supports_no_speech_thold:
            cmd.extend(["--no-speech-thold", "0.6"])
        if initial_prompt:
            cmd.extend(["--prompt", fit_prompt_to_budget(initial_prompt)])
        if use_vad:
            cmd.extend(["--vad", "--vad-model", vad_model])

        live_blocks: List[Tuple[float, float, str]] = []
        live_trace_blocks: List[Tuple[float, float, str]] = []
        saw_processing = False
        last_prelude_update = 0.0
        repeat_key = ""
        repeat_count = 0
        repeat_run_start = 0.0
        recovery_cut_current = 0.0
        recovery_requested = False
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
                    done = min(total_sec, max((end for _, end, _ in accepted_blocks), default=current_offset_sec))
                    prog.progress(
                        min(0.99, max(0.01, done / max(total_sec, 1e-6))),
                        text=f"正在轉寫…｜已轉錄音時長 {fmt_dur(done)} / {fmt_dur(total_sec)}，耗時 {fmt_dur(elapsed)}",
                    )
                    last_prelude_update = time.time()
                elif not live_blocks and time.time() - last_prelude_update > 3:
                    elapsed = time.time() - t0
                    done = min(total_sec, max((end for _, end, _ in accepted_blocks), default=current_offset_sec))
                    prelude_progress = max(
                        0.05 if saw_processing else 0.03,
                        min(0.99, done / max(total_sec, 1e-6)),
                    )
                    if done > 0.5:
                        label = f"正在從 {fmt_dur(done)} 接續轉寫…"
                    else:
                        label = "等待第一段文字…" if saw_processing else "正在載入模型…"
                    prog.progress(prelude_progress, text=f"{label}｜已轉錄音時長 {fmt_dur(done)} / {fmt_dur(total_sec)}，耗時 {fmt_dur(elapsed)}")
                    last_prelude_update = time.time()
                continue

            start_sec = _parse_whispercpp_timecode(match.group(1))
            end_sec = _parse_whispercpp_timecode(match.group(2))
            text = normalize_zh_punc(apply_common_corrections(cc.convert(match.group(3).strip())))
            if not text:
                continue

            global_start = current_offset_sec + start_sec
            global_end = current_offset_sec + end_sec
            key = _repeat_detection_key(text)
            if key and key == repeat_key and len(key) >= 4:
                repeat_count += 1
            else:
                repeat_key = key
                repeat_count = 1
                repeat_run_start = start_sec

            current_block = (global_start, global_end, text)
            loop_reason, loop_start_global = _transcript_loop_signal(live_trace_blocks + [current_block])
            repeat_reason = ""
            if repeat_count >= REPEAT_RECOVERY_THRESHOLD:
                repeat_reason = f"連續重複 {repeat_count} 次"
                if not loop_reason:
                    loop_reason = repeat_reason
                    loop_start_global = current_offset_sec + repeat_run_start

            if loop_reason:
                loop_start = max(0.0, loop_start_global - current_offset_sec)
                retry_cut = max(0.0, loop_start - REPEAT_RECOVERY_BACKTRACK_SEC)
                retry_global_start = current_offset_sec + retry_cut
                recovery_location_key = _repeat_recovery_location_key(retry_global_start)
                location_attempts = recovery_attempts_by_location.get(recovery_location_key, 0)
                if location_attempts < REPEAT_RECOVERY_MAX_RESTARTS_PER_LOCATION:
                    recovery_attempts_by_location[recovery_location_key] = location_attempts + 1
                    recovery_cut_current = retry_cut
                    action = (
                        f"從 {fmt_dur(retry_global_start)} 前後重試接續"
                        f"（此處第 {location_attempts + 1}/{REPEAT_RECOVERY_MAX_RESTARTS_PER_LOCATION} 次）"
                    )
                    progress_at = retry_global_start
                else:
                    remaining_sec = max(0.0, total_sec - current_offset_sec)
                    recovery_cut_current = min(
                        max(0.0, loop_start + REPEAT_RECOVERY_SKIP_SEC),
                        max(0.0, remaining_sec - 0.5),
                    )
                    progress_at = current_offset_sec + recovery_cut_current
                    action = f"已重試仍重複，略過到 {fmt_dur(progress_at)} 繼續"

                preserve_until = current_offset_sec + retry_cut
                accepted_blocks.extend(block for block in live_blocks if block[1] <= preserve_until)
                accepted_blocks = sorted(accepted_blocks, key=lambda item: (item[0], item[1]))
                _terminate_process(proc)
                _cleanup_whispercpp_outputs(out_prefix)
                recovery_requested = True
                elapsed = time.time() - t0
                prog.progress(
                    min(0.99, max(0.01, progress_at / max(total_sec, 1e-6))),
                    text=f"偵測到異常重複輸出（{loop_reason}），{action}｜耗時 {fmt_dur(elapsed)}",
                )
                break

            live_trace_blocks.append(current_block)
            if _repetition_artifact_reason(text):
                continue

            live_blocks.append(current_block)
            processed_sec = max(0.0, global_end)
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
            display_blocks = accepted_blocks + live_blocks
            live_text = _display_text_from_blocks(display_blocks, punc_rule, lexical_rules, punc_settings)
            live_box.markdown(markdown_linebreaks(live_text))
            prog.progress(
                min(0.99, max(0.01, processed_sec / max(total_sec, 1e-6))),
                text=f"已轉錄音時長 {fmt_dur(processed_sec)} / {fmt_dur(total_sec)}，耗時 {fmt_dur(elapsed)}{eta_txt}",
            )

        if recovery_requested:
            recoveries += 1
            next_offset = current_offset_sec + recovery_cut_current
            recovery_path = str(Path(base_media_path).with_suffix(f".recover{recoveries}.wav"))
            _export_tail_wav(current_media_path, recovery_path, recovery_cut_current)
            recovery_paths.append(recovery_path)
            current_media_path = recovery_path
            current_offset_sec = next_offset
            continue

        returncode = proc.wait()
        if returncode != 0:
            detail = "\n".join(log_tail).strip()
            raise RuntimeError(f"whisper.cpp 轉寫失敗：{detail[-1200:]}")

        srt_path = Path(out_prefix + ".srt")
        if not srt_path.exists():
            raise FileNotFoundError(f"whisper.cpp 沒有產生 SRT：{srt_path}")

        raw_srt = srt_path.read_text(encoding="utf-8", errors="replace")
        srt_trace_blocks: List[Tuple[float, float, str]] = []
        srt_accept_blocks: List[Tuple[float, float, str]] = []
        for start, end, text in _read_srt_blocks(raw_srt):
            text = normalize_zh_punc(apply_common_corrections(cc.convert(text).strip()))
            if not text:
                continue

            current_block = (current_offset_sec + start, current_offset_sec + end, text)
            loop_reason, loop_start_global = _transcript_loop_signal(srt_trace_blocks + [current_block])
            if loop_reason:
                loop_start = max(0.0, loop_start_global - current_offset_sec)
                retry_cut = max(0.0, loop_start - REPEAT_RECOVERY_BACKTRACK_SEC)
                retry_global_start = current_offset_sec + retry_cut
                recovery_location_key = _repeat_recovery_location_key(retry_global_start)
                location_attempts = recovery_attempts_by_location.get(recovery_location_key, 0)
                if location_attempts < REPEAT_RECOVERY_MAX_RESTARTS_PER_LOCATION:
                    recovery_attempts_by_location[recovery_location_key] = location_attempts + 1
                    recovery_cut_current = retry_cut
                    action = (
                        f"從 {fmt_dur(retry_global_start)} 前後重試接續"
                        f"（此處第 {location_attempts + 1}/{REPEAT_RECOVERY_MAX_RESTARTS_PER_LOCATION} 次）"
                    )
                    progress_at = retry_global_start
                else:
                    remaining_sec = max(0.0, total_sec - current_offset_sec)
                    recovery_cut_current = min(
                        max(0.0, loop_start + REPEAT_RECOVERY_SKIP_SEC),
                        max(0.0, remaining_sec - 0.5),
                    )
                    progress_at = current_offset_sec + recovery_cut_current
                    action = f"已重試仍重複，略過到 {fmt_dur(progress_at)} 繼續"

                preserve_until = current_offset_sec + retry_cut
                accepted_blocks.extend(block for block in srt_accept_blocks if block[1] <= preserve_until)
                accepted_blocks = sorted(accepted_blocks, key=lambda item: (item[0], item[1]))
                recovery_requested = True
                elapsed = time.time() - t0
                prog.progress(
                    min(0.99, max(0.01, progress_at / max(total_sec, 1e-6))),
                    text=f"完成輸出檢查時偵測到異常重複（{loop_reason}），{action}｜耗時 {fmt_dur(elapsed)}",
                )
                break

            srt_trace_blocks.append(current_block)
            if not _repetition_artifact_reason(text):
                srt_accept_blocks.append(current_block)

        if not recovery_requested:
            accepted_blocks.extend(srt_accept_blocks)
        try:
            srt_path.unlink()
        except OSError:
            pass
        if recovery_requested:
            recoveries += 1
            next_offset = current_offset_sec + recovery_cut_current
            recovery_path = str(Path(base_media_path).with_suffix(f".recover{recoveries}.wav"))
            _export_tail_wav(current_media_path, recovery_path, recovery_cut_current)
            recovery_paths.append(recovery_path)
            current_media_path = recovery_path
            current_offset_sec = next_offset
            continue
        break

    blocks = sorted(accepted_blocks, key=lambda item: (item[0], item[1]))

    output_blocks = _output_blocks_from_blocks(blocks, punc_rule)
    if not output_blocks:
        output_blocks = blocks

    live_text = _display_text_from_blocks(output_blocks, punc_rule, lexical_rules, punc_settings)
    live_box.markdown(markdown_linebreaks(live_text))
    final_srt = _srt_blocks_to_srt(output_blocks, time_offset_sec=time_offset_sec)
    if output_blocks != blocks:
        final_text = "\n".join(text for _, _, text in output_blocks)
    else:
        final_text = _final_text_from_blocks(blocks, punc_rule, lexical_rules, punc_settings)

    elapsed = time.time() - t0
    processed_sec = max((end for _, end, _ in blocks), default=total_sec)
    prog.progress(1.0, text=f"已轉錄音時長 {fmt_dur(processed_sec)} / {fmt_dur(total_sec)}，耗時 {fmt_dur(elapsed)}")
    stats_area.info(f"已轉錄音時長：{fmt_dur(processed_sec)}；耗時：{fmt_dur(elapsed)}")

    for path in recovery_paths:
        try:
            os.remove(path)
        except OSError:
            pass
    return final_text, final_srt, processed_sec, elapsed


if not IS_WORKER:
    # ==== 輸入 ====
    uploaded = None
    uploaded_name = None
    local_audio_path = ""
    uploaded = st.file_uploader("上傳音檔", type=["m4a","mp3","wav","flac"], key="main_audio_uploader")
    local_audio_path = st.text_input(
        "或輸入本機音檔路徑（大檔建議）",
        value="",
        placeholder="/Users/iw/Downloads/recording.m4a",
        help="本機執行 Streamlit 時可直接讀取檔案，避免大檔經瀏覽器上傳。",
        key="local_audio_path",
    ).strip()
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
    if local_audio_path:
        uploaded_name = Path(local_audio_path).expanduser().name
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
        line_break_version = linebreak_after_punctuation(last_txt)
        display_text = (
            line_break_version
            if show_line_mode == "每句換行"
            else last_txt.replace("\n", " ")
        )





# === 鎖定顯示模式與主流程 ===
if not IS_WORKER and st.session_state.get("start_transcribe_pending"):
    local_audio_file = Path(local_audio_path).expanduser() if local_audio_path else None
    if uploaded is None and local_audio_file is None:
        status.warning("請上傳音訊檔，或輸入本機音檔路徑，再開始轉寫")
        st.session_state["transcribing"] = False
        st.session_state["start_transcribe_pending"] = False
        st.session_state["transcribe_status_message"] = ""
        st.session_state["transcribe_success_message"] = ""
        if "run_button_primary" in st.session_state:
            del st.session_state["run_button_primary"]
        st.stop()
    if local_audio_file is not None and (not local_audio_file.exists() or not local_audio_file.is_file()):
        status.warning(f"找不到本機音檔：{local_audio_file}")
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

    try:
        if local_audio_file is not None:
            original_size = local_audio_file.stat().st_size
            processing_name = local_audio_file.name
            if original_size > LARGE_UPLOAD_THRESHOLD_BYTES:
                status.info("使用本機音檔路徑直接讀取大檔，已繞過瀏覽器上傳。")
            seg, total_sec = audio_info_from_path(str(local_audio_file))
        else:
            # 取得 bytes；超過 200MB 仍直接使用原檔，靠 Streamlit server 設定放寬上傳限制。
            raw = uploaded.getvalue()
            raw, processing_name, compressed_large_upload, original_size, processing_size = maybe_compress_large_upload(
                raw,
                uploaded_name,
                tmp_dir,
                ts,
                status_area=status,
            )
            seg, total_sec = audio_info_from_bytes(raw, processing_name)
    except Exception as exc:
        status.error("音檔前處理失敗，無法進入轉寫。")
        with st.expander("ffmpeg 錯誤細節", expanded=False):
            st.code(str(exc)[-2000:])
        st.session_state["transcribing"] = False
        st.session_state["start_transcribe_pending"] = False
        st.session_state["transcribe_status_message"] = ""
        if "run_button_primary" in st.session_state:
            del st.session_state["run_button_primary"]
        st.stop()

    upload_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    size_note = ""
    if original_size > LARGE_UPLOAD_THRESHOLD_BYTES:
        source_note = "本機路徑" if local_audio_file is not None else "未壓縮上傳"
        size_note = f"｜原檔大小：{original_size / (1024 * 1024):.1f}MB（{source_note}）"
    top_info.info(f"上傳錄音時長：{fmt_dur(total_sec)}｜上傳時間點：{upload_ts}{size_note}")

    # 轉為單一 wav 檔（作為後續切段基礎）
    base_path = os.path.join(tmp_dir, f"_tmp_{ts}.wav")
    export_seg_to_wav(seg, base_path, clean_audio=clean_audio_for_asr)

    # 構建初始提示（合併使用者提示與領域詞彙）
    anti_artifact_prompt = "僅轉錄實際聽到的中文語音；忽略字幕、水印、畫面文字、系統提示與非語音內容；沒有清楚語音時留空。"
    combined_prompt = (init_prompt or "").strip()
    combined_prompt = (combined_prompt + ("，" if combined_prompt else "") + anti_artifact_prompt).strip("， ")
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

    # whisper 只讀得進 224 個 token，這裡切在詞的邊界上，並且把真正送進去的內容顯示出來
    raw_prompt_len = len(combined_prompt)
    combined_prompt = fit_prompt_to_budget(combined_prompt)
    with st.sidebar:
        if raw_prompt_len > len(combined_prompt):
            st.caption(
                f"初始提示已裁到 {len(combined_prompt)} 字（原本 {raw_prompt_len} 字）；"
                f"whisper 上限約 {WHISPER_PROMPT_TOKEN_LIMIT} 個 token，超出的不會被讀到。"
            )
        else:
            st.caption(f"初始提示 {len(combined_prompt)} 字，未超出 whisper 上限。")
        st.code(combined_prompt)

    # 檢查 whisper.cpp 最佳配置
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

    # === 切段與並行策略 ===
    device_now = "whisper.cpp"
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
        export_seg_to_wav(seg[a*1000:b*1000], p, clean_audio=clean_audio_for_asr)
        chunk_paths.append((p,a,b))

    if k == 1:
        device_msg = f"裝置：{device_now}；整段轉寫；並行：否"
    else:
        device_msg = f"裝置：{device_now}；切段 {k}（含 {overlap}s 重疊）；並行：否"
    if clean_audio_for_asr:
        device_msg += "；已清理底噪/正規化音量"
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
    final_text = "".join(text.strip() for text in final_texts if text and text.strip())

    if enable_speaker_diarization:
        requested_speakers = None
        if speaker_count_choice != "自動":
            try:
                requested_speakers = int(speaker_count_choice)
            except ValueError:
                requested_speakers = None
        try:
            labeled_text, labeled_srt, speaker_msg = apply_speaker_labels_to_outputs(
                audio_path=base_path,
                srt_text=final_srt,
                hf_token=speaker_hf_token,
                requested_speakers=requested_speakers,
                total_sec=total_sec,
                punc_rule=punc_rule,
                lexical_rules=LEXICAL_RULES,
                punc_settings=CURRENT_PUNCT_SETTINGS,
                status_area=status,
            )
            if labeled_text:
                final_text = labeled_text
                final_srt = labeled_srt
                st.session_state["speaker_diarization_message"] = speaker_msg
                status.info(speaker_msg)
            else:
                st.session_state["speaker_diarization_message"] = speaker_msg
                status.info(speaker_msg)
        except Exception as exc:
            speaker_msg = f"說話者辨識未完成：{str(exc)[-300:]}"
            st.session_state["speaker_diarization_message"] = speaker_msg
            status.warning(speaker_msg)
    else:
        st.session_state["speaker_diarization_message"] = ""

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
    line_break_version = linebreak_after_punctuation(final_text)

    if show_line_mode == "每句換行":
        display_text = line_break_version
    else:
        display_text = final_text.replace("\n", " ")



    # === 顯示結果與耗時 ===
    final_box.markdown(markdown_linebreaks(display_text))
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
    speaker_diarization_message = st.session_state.get("speaker_diarization_message")
    if speaker_diarization_message:
        st.caption(speaker_diarization_message)

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
    line_break_version = linebreak_after_punctuation(cached_text)
    display_text = (
        line_break_version
        if st.session_state.get("show_line_mode", "每句換行") == "每句換行"
        else cached_text.replace("\n", " ")
    )

    st.markdown(markdown_linebreaks(display_text))

    # === 側邊欄一鍵複製 ===
    with st.sidebar:
        show_line_mode = st.session_state.get("show_line_mode", "每句換行")
        text_to_copy = (
            linebreak_after_punctuation(st.session_state["last_txt"])
            if show_line_mode == "每句換行"
            else st.session_state["last_txt"].replace("\n", " ")
        )
        if st.button("📋 顯示可複製內容", key="copy_btn"):
            st.session_state["copy_buffer"] = text_to_copy
            st.toast("已顯示可複製內容，請在下方區塊手動複製。", icon="📋")

        if "copy_buffer" in st.session_state:
            copy_buffer = st.session_state["copy_buffer"]
            if show_line_mode == "每句換行":
                copy_buffer = linebreak_after_punctuation(copy_buffer)
                st.session_state["copy_buffer"] = copy_buffer
            st.text_area("目前顯示內容（可全選後複製）", copy_buffer, height=300)
