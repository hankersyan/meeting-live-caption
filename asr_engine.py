"""
FunASR ASR Engine with Speaker Diarization

Provides real-time and offline speech recognition with speaker diarization
using FunASR's AutoModel API.

Two processing modes:
- Native diarization: For models that support timestamps (paraformer-zh),
  uses spk_model="cam++" directly in the AutoModel pipeline.
- Two-pass diarization: For models without timestamp support (SenseVoiceSmall,
  Fun-ASR-Nano), runs VAD + ASR + cam++ embedding extraction separately,
  then clusters embeddings via SpeakerRegistry.
"""

import os
import time
import queue
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable

import numpy as np

try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
except ImportError:
    AutoModel = None
    FUNASR_AVAILABLE = False

try:
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
except ImportError:
    rich_transcription_postprocess = None


# ------------------------------
# Data Structures
# ------------------------------

@dataclass
class TranscriptionResult:
    """A single transcription result with optional speaker label."""
    text: str
    speaker: Optional[str] = None     # e.g. "SPEAKER_0"
    start_ms: int = 0
    end_ms: int = 0
    is_final: bool = True


# ------------------------------
# Speaker Registry
# ------------------------------

class SpeakerRegistry:
    """Maintains cross-chunk speaker identity consistency.

    Stores speaker embedding vectors and matches new embeddings via
    cosine similarity. When a new embedding doesn't match any existing
    speaker above the threshold, a new speaker is registered.
    """

    def __init__(self, similarity_threshold: float = 0.65):
        self._embeddings: List[np.ndarray] = []
        self._speaker_ids: List[str] = []
        self._next_id: int = 0
        self._threshold = similarity_threshold
        self._lock = threading.Lock()

    def match_or_register(self, embedding: np.ndarray) -> str:
        """Match an embedding to an existing speaker, or register a new one.

        Args:
            embedding: Speaker embedding vector (1-D numpy array).

        Returns:
            Speaker ID string, e.g. "SPEAKER_0".
        """
        if embedding is None or len(embedding) == 0:
            return f"SPEAKER_{self._next_id}"

        embedding = embedding.flatten().astype(np.float64)
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return f"SPEAKER_{self._next_id}"
        embedding = embedding / norm

        with self._lock:
            best_id = None
            best_sim = -1.0

            for i, stored in enumerate(self._embeddings):
                stored_norm = stored / (np.linalg.norm(stored) + 1e-8)
                sim = float(np.dot(embedding, stored_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_id = self._speaker_ids[i]

            if best_sim >= self._threshold and best_id is not None:
                # Update embedding with exponential moving average
                idx = self._speaker_ids.index(best_id)
                self._embeddings[idx] = 0.7 * self._embeddings[idx] + 0.3 * embedding
                return best_id

            # Register new speaker
            new_id = f"SPEAKER_{self._next_id}"
            self._next_id += 1
            self._speaker_ids.append(new_id)
            self._embeddings.append(embedding.copy())
            return new_id

    def get_speaker_count(self) -> int:
        with self._lock:
            return len(self._speaker_ids)

    def reset(self):
        with self._lock:
            self._embeddings.clear()
            self._speaker_ids.clear()
            self._next_id = 0


# ------------------------------
# FunASR Transcriber
# ------------------------------

# Model metadata.
# timestamp_support=True means the model natively supports timestamps,
# so spk_model="cam++" can be used directly in the AutoModel pipeline.
# timestamp_support=False means we must use a two-pass approach:
#   VAD → ASR + cam++ embedding → cluster
# Model IDs must use the ModelScope namespace/name format.
MODEL_CONFIG = {
    "iic/SenseVoiceSmall": {
        "timestamp_support": False,
        "languages": ["auto", "en", "zh", "ja", "ko", "yue"],
        "description": "Multilingual (EN/ZH/JA/KO/YUE), two-pass diarization",
    },
    "FunAudioLLM/Fun-ASR-Nano-2512": {
        "timestamp_support": False,
        "languages": ["auto", "en", "zh", "ja", "ko"],
        "description": "31 languages, two-pass diarization",
    },
    "paraformer-zh": {
        "timestamp_support": True,
        "languages": ["zh", "auto"],
        "description": "Chinese-optimized, native diarization",
    },
}


class FunASRTranscriber:
    """Processes audio chunks from an AudioInputBase queue using FunASR.

    Automatically selects the processing mode based on the model:
    - Native mode: for models with timestamp_support=True (paraformer-zh)
    - Two-pass mode: for models with timestamp_support=False (SenseVoiceSmall, etc.)
    """

    def __init__(
        self,
        audio_input,
        model_name: str = "iic/SenseVoiceSmall",
        language: str = "en",
        device: str = "cpu",
        records_folder: str = "records",
        disable_update: bool = True,
    ):
        self.audio_input = audio_input
        self.model_name = model_name
        self.language = language
        self.device = device
        self.records_folder = records_folder
        self.disable_update = disable_update

        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._speaker_registry = SpeakerRegistry()

        self.text_callback: Optional[Callable[[TranscriptionResult], None]] = None
        self._text_file = None
        self._text_path: Optional[str] = None

        self._model_config = MODEL_CONFIG.get(model_name, MODEL_CONFIG["iic/SenseVoiceSmall"])

        # Models for native mode
        self._model = None

        # Models for two-pass mode
        self._vad_model = None
        self._asr_model = None
        self._spk_model = None

        # Processing mode: "native" or "twopass"
        self._mode = "native" if self._model_config.get("timestamp_support", False) else "twopass"

        # Cumulative timestamp offset for two-pass mode
        self._chunk_offset_ms: int = 0

    def set_text_callback(self, callback: Callable[[TranscriptionResult], None]):
        self.text_callback = callback

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._speaker_registry.reset()
        self._chunk_offset_ms = 0

        # Prepare text file
        base = self.audio_input.get_base_filename()
        if base:
            os.makedirs(self.records_folder, exist_ok=True)
            self._text_path = os.path.join(self.records_folder, base + ".txt")
            self._text_file = open(self._text_path, "w", encoding="utf-8")
            self._text_file.write(f"Meeting Transcription - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._text_file.write("=" * 50 + "\n\n")
            self._text_file.flush()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._text_file:
            self._text_file.write("\n\n" + "=" * 50 + "\n")
            self._text_file.write(f"Recording ended - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._text_file.close()
            self._text_file = None

    def _save_text(self, text: str):
        try:
            if self._text_file:
                self._text_file.write(text)
                self._text_file.flush()
        except Exception as e:
            print(f"[ASREngine] Text save error: {e}")

    # ----------------------------------------------------------
    # Model Loading
    # ----------------------------------------------------------

    def _load_model(self):
        """Load FunASR models based on the processing mode."""
        try:
            if self._mode == "native":
                self._load_model_native()
            else:
                self._load_model_twopass()
        except Exception as e:
            print(f"[ASREngine] Failed to load FunASR model: {e}")
            if self.text_callback:
                self.text_callback(TranscriptionResult(
                    text=f"[Error] Failed to load model: {e}",
                    is_final=True,
                ))
            raise

    def _load_model_native(self):
        """Load a single AutoModel with native spk_model support (paraformer-zh)."""
        model_kwargs = {
            "device": self.device,
            "disable_update": self.disable_update,
            "hub": "ms",
        }
        self._model = AutoModel(
            model=self.model_name,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            punc_model="ct-punc",
            spk_model="cam++",
            **model_kwargs,
        )
        print(f"[ASREngine] Native mode: {self.model_name} + cam++ loaded")

    def _load_model_twopass(self):
        """Load separate VAD, ASR, and speaker embedding models for two-pass mode."""
        model_kwargs = {
            "device": self.device,
            "disable_update": self.disable_update,
            "hub": "ms",
        }

        # 1. VAD model — segments audio into speech intervals
        self._vad_model = AutoModel(model="fsmn-vad", **model_kwargs)

        # 2. ASR model — transcribes each segment (no spk_model)
        self._asr_model = AutoModel(
            model=self.model_name,
            punc_model="ct-punc",
            **model_kwargs,
        )

        # 3. Speaker embedding model — extracts 192-d embeddings per segment
        self._spk_model = AutoModel(model="cam++", **model_kwargs)

        print(f"[ASREngine] Two-pass mode: VAD + {self.model_name} + cam++ loaded")

    # ----------------------------------------------------------
    # Processing Loop
    # ----------------------------------------------------------

    def _run_loop(self):
        """Main processing loop."""
        try:
            self._load_model()
        except Exception:
            self.is_running = False
            return

        while self.is_running:
            try:
                audio_chunk = self.audio_input.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                if self._mode == "native":
                    self._process_chunk_native(audio_chunk)
                else:
                    self._process_chunk_twopass(audio_chunk)
            except Exception as e:
                print(f"[ASREngine] Processing error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    # ----------------------------------------------------------
    # Native Mode Processing (paraformer-zh)
    # ----------------------------------------------------------

    def _process_chunk_native(self, audio_int16: np.ndarray):
        """Process a chunk using the native pipeline with spk_model."""
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        generate_kwargs = {
            "input": audio_float32,
            "batch_size_s": 300,
        }

        results = self._model.generate(**generate_kwargs)
        if not results or len(results) == 0:
            return

        result = results[0]
        sentences = result.get("sentences", [])

        if sentences:
            for sent in sentences:
                text = sent.get("text", "").strip()
                if not text:
                    continue
                spk_idx = sent.get("spk", 0)
                tr = TranscriptionResult(
                    text=text + " ",
                    speaker=f"SPEAKER_{spk_idx}",
                    start_ms=sent.get("start", 0),
                    end_ms=sent.get("end", 0),
                    is_final=True,
                )
                self._emit_result(tr)
        else:
            text = result.get("text", "").strip()
            if text:
                tr = TranscriptionResult(
                    text=text + " ",
                    speaker=None,
                    is_final=True,
                )
                self._emit_result(tr)

    # ----------------------------------------------------------
    # Two-Pass Mode Processing (SenseVoiceSmall, Fun-ASR-Nano)
    # ----------------------------------------------------------

    def _process_chunk_twopass(self, audio_int16: np.ndarray):
        """Process a chunk using two-pass: VAD → ASR + cam++ embedding → cluster.

        Steps:
        1. Run fsmn-vad on the chunk → get speech segments [[start_ms, end_ms], ...]
        2. For each VAD segment:
           a. Extract the segment audio
           b. Run ASR (SenseVoiceSmall) to get text
           c. Run cam++ to extract speaker embedding
           d. Match/register embedding via SpeakerRegistry
           e. Emit result with speaker label
        """
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        chunk_len_ms = int(len(audio_float32) / 16)  # 16 samples/ms at 16kHz

        # Step 1: VAD — detect speech segments
        vad_res = self._vad_model.generate(input=audio_float32, batch_size_s=300)
        if not vad_res or len(vad_res) == 0:
            self._chunk_offset_ms += chunk_len_ms
            return

        segments = vad_res[0].get("value", [])
        if not segments:
            self._chunk_offset_ms += chunk_len_ms
            return

        # Step 2: Process each VAD segment
        for seg_start_ms, seg_end_ms in segments:
            seg_start_sample = int(seg_start_ms / 1000 * 16000)
            seg_end_sample = int(seg_end_ms / 1000 * 16000)

            # Clamp to audio bounds
            seg_start_sample = max(0, min(seg_start_sample, len(audio_float32)))
            seg_end_sample = max(0, min(seg_end_sample, len(audio_float32)))

            if seg_end_sample - seg_start_sample < 1600:
                # Skip segments shorter than 100ms
                continue

            segment_audio = audio_float32[seg_start_sample:seg_end_sample]

            # Step 2a: ASR transcription
            text = self._run_asr_on_segment(segment_audio)
            if not text:
                continue

            # Step 2b: Speaker embedding extraction + clustering
            speaker_label = self._get_speaker_label(segment_audio)

            # Absolute timestamps
            abs_start_ms = self._chunk_offset_ms + seg_start_ms
            abs_end_ms = self._chunk_offset_ms + seg_end_ms

            tr = TranscriptionResult(
                text=text + " ",
                speaker=speaker_label,
                start_ms=abs_start_ms,
                end_ms=abs_end_ms,
                is_final=True,
            )
            self._emit_result(tr)

        self._chunk_offset_ms += chunk_len_ms

    def _run_asr_on_segment(self, segment_audio: np.ndarray) -> str:
        """Run ASR on a single VAD segment and return the text."""
        generate_kwargs = {"input": segment_audio}

        if "SenseVoiceSmall" in self.model_name:
            generate_kwargs.update({
                "language": self.language,
                "use_itn": True,
                "cache": {},
            })
        elif "Fun-ASR-Nano" in self.model_name:
            generate_kwargs.update({"cache": {}})

        try:
            results = self._asr_model.generate(**generate_kwargs)
        except Exception as e:
            print(f"[ASREngine] ASR error on segment: {e}")
            return ""

        if not results or len(results) == 0:
            return ""

        text = results[0].get("text", "")
        if "SenseVoiceSmall" in self.model_name and rich_transcription_postprocess:
            text = rich_transcription_postprocess(text)

        return text.strip()

    def _get_speaker_label(self, segment_audio: np.ndarray) -> Optional[str]:
        """Extract speaker embedding from a segment and match/register it."""
        try:
            spk_res = self._spk_model.generate(input=segment_audio)
            if not spk_res or len(spk_res) == 0:
                return None

            embedding = spk_res[0].get("spk_embedding", None)
            if embedding is None:
                return None

            return self._speaker_registry.match_or_register(embedding)
        except Exception as e:
            print(f"[ASREngine] Speaker embedding error: {e}")
            return None

    # ----------------------------------------------------------
    # Result Output
    # ----------------------------------------------------------

    def _emit_result(self, result: TranscriptionResult):
        """Emit a transcription result to the callback and file."""
        if self.text_callback:
            try:
                self.text_callback(result)
            except Exception as e:
                print(f"[ASREngine] Callback error: {e}")

        # Save to text file
        if result.speaker:
            self._save_text(f"[{result.speaker}] {result.text}")
        else:
            self._save_text(result.text)


# ------------------------------
# Standalone File Transcription
# ------------------------------

def transcribe_file(
    file_path: str,
    model_name: str = "iic/SenseVoiceSmall",
    language: str = "en",
    device: str = "cpu",
    output_path: Optional[str] = None,
) -> List[TranscriptionResult]:
    """Transcribe an entire audio file with speaker diarization.

    Automatically selects native or two-pass mode based on model.

    Args:
        file_path: Path to audio file (WAV, MP3, MP4, etc.)
        model_name: FunASR model name.
        language: Language code.
        device: "cpu" or "cuda".
        output_path: Optional output text file path.

    Returns:
        List of TranscriptionResult objects.
    """
    if not FUNASR_AVAILABLE:
        raise ImportError("FunASR is not installed. Run: pip install funasr")

    import soundfile as sf

    # Load audio
    speech, sr = sf.read(file_path, dtype='int16')
    if speech.ndim > 1:
        speech = speech.mean(axis=1).astype(np.int16)
    if sr != 16000:
        ratio = 16000 / sr
        indices = np.arange(0, len(speech), 1 / ratio)
        indices = indices[indices < len(speech)]
        speech = np.interp(indices, np.arange(len(speech)), speech).astype(np.int16)
        sr = 16000

    audio_float32 = speech.astype(np.float32) / 32768.0

    # Determine processing mode
    model_config = MODEL_CONFIG.get(model_name, MODEL_CONFIG["iic/SenseVoiceSmall"])
    timestamp_support = model_config.get("timestamp_support", False)
    model_kwargs = {"device": device, "disable_update": True, "hub": "ms"}

    if timestamp_support:
        # Native mode: single pipeline with spk_model
        model = AutoModel(
            model=model_name,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            punc_model="ct-punc",
            spk_model="cam++",
            **model_kwargs,
        )
        results = model.generate(input=audio_float32, batch_size_s=300)

        transcription_results = []
        if results:
            result = results[0]
            sentences = result.get("sentences", [])
            if sentences:
                for sent in sentences:
                    text = sent.get("text", "").strip()
                    if not text:
                        continue
                    spk_idx = sent.get("spk", 0)
                    transcription_results.append(TranscriptionResult(
                        text=text,
                        speaker=f"SPEAKER_{spk_idx}",
                        start_ms=sent.get("start", 0),
                        end_ms=sent.get("end", 0),
                        is_final=True,
                    ))
            else:
                text = result.get("text", "").strip()
                if text:
                    transcription_results.append(TranscriptionResult(
                        text=text,
                        is_final=True,
                    ))
    else:
        # Two-pass mode: VAD → ASR + cam++ embedding → cluster
        vad_model = AutoModel(model="fsmn-vad", **model_kwargs)
        asr_model = AutoModel(model=model_name, punc_model="ct-punc", **model_kwargs)
        spk_model = AutoModel(model="cam++", **model_kwargs)
        speaker_registry = SpeakerRegistry()

        # Step 1: VAD
        vad_res = vad_model.generate(input=audio_float32, batch_size_s=300)
        segments = vad_res[0].get("value", []) if vad_res else []

        # Step 2: Process each VAD segment
        transcription_results = []
        for seg_start_ms, seg_end_ms in segments:
            seg_start = int(seg_start_ms / 1000 * 16000)
            seg_end = int(seg_end_ms / 1000 * 16000)
            seg_end = min(seg_end, len(audio_float32))

            if seg_end - seg_start < 1600:
                continue

            segment_audio = audio_float32[seg_start:seg_end]

            # ASR
            asr_kwargs = {"input": segment_audio}
            if "SenseVoiceSmall" in model_name:
                asr_kwargs.update({"language": language, "use_itn": True, "cache": {}})
            elif "Fun-ASR-Nano" in model_name:
                asr_kwargs.update({"cache": {}})

            asr_res = asr_model.generate(**asr_kwargs)
            if not asr_res:
                continue

            text = asr_res[0].get("text", "")
            if "SenseVoiceSmall" in model_name and rich_transcription_postprocess:
                text = rich_transcription_postprocess(text)
            text = text.strip()
            if not text:
                continue

            # Speaker embedding
            speaker_label = None
            try:
                spk_res = spk_model.generate(input=segment_audio)
                if spk_res:
                    embedding = spk_res[0].get("spk_embedding", None)
                    if embedding is not None:
                        speaker_label = speaker_registry.match_or_register(embedding)
            except Exception as e:
                print(f"[transcribe_file] Speaker embedding error: {e}")

            transcription_results.append(TranscriptionResult(
                text=text,
                speaker=speaker_label,
                start_ms=seg_start_ms,
                end_ms=seg_end_ms,
                is_final=True,
            ))

    # Save to file
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Speaker-labeled Transcription\n")
            f.write("=" * 50 + "\n\n")
            for tr in transcription_results:
                if tr.speaker:
                    f.write(f"[{tr.start_ms/1000:.1f}s - {tr.end_ms/1000:.1f}s] [{tr.speaker}] {tr.text}\n")
                else:
                    f.write(f"{tr.text}\n")
            f.write("\n\n" + "=" * 50 + "\nPer-Speaker Summary\n" + "=" * 50 + "\n")
            speaker_texts: Dict[str, List[str]] = {}
            for tr in transcription_results:
                s = tr.speaker or "UNKNOWN"
                speaker_texts.setdefault(s, []).append(tr.text)
            for speaker, texts in speaker_texts.items():
                f.write(f"\n{speaker}:\n  {' '.join(texts)}\n")

    return transcription_results
