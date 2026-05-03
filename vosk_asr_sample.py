"""
Vosk ASR Sample with Speaker Diarization — three modes:
  1. Microphone streaming  (real-time partial + final results)
  2. Speaker/system audio streaming (WASAPI loopback, Windows only)
  3. WAV file transcription (offline, with word timestamps)

All modes support optional speaker diarization via pyannote.audio.

Usage:
  python vosk_asr_sample.py mic                                   # live mic
  python vosk_asr_sample.py mic --diarize --hf-token hf_xxx       # live mic + diarization
  python vosk_asr_sample.py speaker                               # system audio (WASAPI loopback)
  python vosk_asr_sample.py speaker --device 12                   # specific loopback device
  python vosk_asr_sample.py speaker --diarize --hf-token hf_xxx   # system audio + diarization
  python vosk_asr_sample.py speaker --list-devices                # list available loopback devices
  python vosk_asr_sample.py file audio.wav                        # transcribe a WAV file
  python vosk_asr_sample.py file audio.wav --diarize --hf-token hf_xxx  # transcribe + diarize

Requirements:
  pip install vosk sounddevice numpy
  (for speaker mode): pip install pyaudiowpatch
  (for diarization):  pip install pyannote.audio torch scipy

Model:
  Download from https://alphacephei.com/vosk/models
  Default: vosk-model-small-en-us-0.15 (in current directory)
"""

import argparse
import json
import os
import queue
import sys
import tempfile
import threading
import time
import wave

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer, SetLogLevel

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000  # 0.5 s per block

# Diarization defaults
DIARIZATION_WINDOW = 5.0   # seconds of audio analyzed per diarization pass
DIARIZATION_INTERVAL = 3.0 # seconds between diarization passes (mic mode)

# ANSI colors for speakers
SPEAKER_COLORS = [
    "\033[94m",  # blue
    "\033[92m",  # green
    "\033[93m",  # yellow
    "\033[91m",  # red
    "\033[95m",  # magenta
    "\033[96m",  # cyan
]
RESET_COLOR = "\033[0m"


def _speaker_color(speaker: str) -> str:
    """Pick an ANSI color for a speaker label."""
    try:
        num = int(speaker.split("_")[-1])
    except (ValueError, IndexError):
        num = hash(speaker) % len(SPEAKER_COLORS)
    return SPEAKER_COLORS[num % len(SPEAKER_COLORS)]


def _format_speaker(speaker: str) -> str:
    """Pretty-print speaker label, e.g. SPEAKER_00 -> S0."""
    if speaker.startswith("SPEAKER_"):
        try:
            return f"S{int(speaker.split('_')[-1])}"
        except ValueError:
            pass
    return speaker


# ---------------------------------------------------------------------------
# Speaker mapping utility
# ---------------------------------------------------------------------------
def map_speakers_to_words(words, diarization_segments, time_eps=0.3):
    """Map speaker labels from diarization to Vosk word-level results.

    Args:
        words: list of dicts with 'start', 'end', 'word' keys
        diarization_segments: list of (start, end, speaker) tuples
        time_eps: minimum overlap (seconds) to assign a speaker

    Returns:
        list of dicts with added 'speaker' key
    """
    labeled = []
    for w in words:
        best_speaker = None
        max_overlap = 0.0
        w_start, w_end = w["start"], w["end"]
        for d_start, d_end, speaker in diarization_segments:
            overlap = max(0.0, min(w_end, d_end) - max(w_start, d_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = speaker
        entry = dict(w)
        if max_overlap >= time_eps and best_speaker is not None:
            entry["speaker"] = best_speaker
        else:
            # Fallback: nearest diarization segment by midpoint
            mid = (w_start + w_end) / 2
            best_gap = float("inf")
            for d_start, d_end, speaker in diarization_segments:
                if d_start <= mid <= d_end:
                    best_speaker = speaker
                    break
                gap = min(abs(mid - d_start), abs(mid - d_end))
                if gap < best_gap:
                    best_gap = gap
                    best_speaker = speaker
            entry["speaker"] = best_speaker
        labeled.append(entry)
    return labeled


def group_words_by_speaker(words):
    """Group consecutive words with the same speaker into segments.

    Returns:
        list of dicts with 'start', 'end', 'speaker', 'text' keys
    """
    if not words:
        return []
    segments = []
    current = {
        "speaker": words[0].get("speaker", "UNKNOWN"),
        "start": words[0]["start"],
        "end": words[0]["end"],
        "text": words[0]["word"],
    }
    for w in words[1:]:
        spk = w.get("speaker", "UNKNOWN")
        if spk == current["speaker"]:
            current["end"] = w["end"]
            current["text"] += " " + w["word"]
        else:
            segments.append(current)
            current = {
                "speaker": spk,
                "start": w["start"],
                "end": w["end"],
                "text": w["word"],
            }
    segments.append(current)
    return segments


# ---------------------------------------------------------------------------
# Pyannote diarization helpers
# ---------------------------------------------------------------------------
def load_diarization_pipeline(hf_token: str, device: str = "cpu"):
    """Load pyannote.audio speaker diarization pipeline."""
    import torch
    import warnings
    warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")
    from pyannote.audio import Pipeline

    print("Loading pyannote speaker-diarization pipeline ...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    if device == "cuda" and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("  Diarization on CUDA")
    else:
        print("  Diarization on CPU")
    return pipeline


def diarize_waveform(pipeline, waveform_np: np.ndarray, sample_rate: int = 16000,
                     num_speakers=None):
    """Run diarization on a numpy int16 waveform array.

    Args:
        pipeline: pyannote Pipeline
        waveform_np: 1-D numpy array (int16 or float32)
        sample_rate: sample rate
        num_speakers: optional hint for number of speakers

    Returns:
        list of (start, end, speaker) tuples
    """
    import torch

    if waveform_np.dtype == np.int16:
        wav_float = waveform_np.astype(np.float32) / 32768.0
    else:
        wav_float = waveform_np.astype(np.float32)

    if wav_float.ndim == 2:
        wav_float = wav_float.mean(axis=1)

    waveform_torch = torch.from_numpy(wav_float).unsqueeze(0)  # (1, T)

    kwargs = {"waveform": waveform_torch, "sample_rate": sample_rate}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    output = pipeline(kwargs)
    annotation = output.speaker_diarization

    results = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        results.append((round(turn.start, 2), round(turn.end, 2), speaker))
    return results


def diarize_wav_file(pipeline, wav_path: str, num_speakers=None):
    """Run diarization on a WAV file path (in-memory loading)."""
    from scipy.io import wavfile

    sr, waveform = wavfile.read(wav_path)
    return diarize_waveform(pipeline, waveform, sr, num_speakers)


# ---------------------------------------------------------------------------
# Mode 1 — Microphone streaming (optionally with diarization)
# ---------------------------------------------------------------------------
def mic_streaming(model_path: str, diarize: bool = False, hf_token: str = "",
                  num_speakers=None):
    SetLogLevel(0)

    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Download from https://alphacephei.com/vosk/models")
        sys.exit(1)

    print(f"Loading Vosk model from {model_path} ...")
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)

    # Diarization setup
    diar_pipeline = None
    if diarize:
        if not hf_token:
            hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            print("ERROR: --hf-token (or HF_TOKEN env var) required for diarization")
            sys.exit(1)
        diar_pipeline = load_diarization_pipeline(hf_token)

    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        audio_queue.put(bytes(indata))

    print(f"Opening microphone ({SAMPLE_RATE} Hz, mono) ...")
    print("Speak into the microphone. Press Ctrl+C to stop.\n")

    # Shared state
    word_results = []       # list of [word, start, end, speaker]
    audio_buffer = np.empty((0,), dtype=np.int16)
    buffer_lock = threading.Lock()
    stop_event = threading.Event()
    last_labeled_idx = 0    # track how many words have been printed with speaker

    # ------------------------------------------------------------------
    # Diarization worker — runs in background, labels words, prints
    # newly labeled segments in real-time
    # ------------------------------------------------------------------
    def diarization_worker():
        nonlocal last_labeled_idx
        while not stop_event.is_set():
            time.sleep(DIARIZATION_INTERVAL)

            with buffer_lock:
                window_samples = int(DIARIZATION_WINDOW * SAMPLE_RATE)
                if len(audio_buffer) < window_samples:
                    continue
                window = audio_buffer[-window_samples:].copy()
                window_start_time = (len(audio_buffer) - window_samples) / SAMPLE_RATE

            # Write to temp WAV for pyannote
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(window.tobytes())

            try:
                segments = diarize_wav_file(diar_pipeline, wav_path, num_speakers)
            except Exception as e:
                print(f"\n[Diarization error] {e}", file=sys.stderr)
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass
                continue

            try:
                os.unlink(wav_path)
            except OSError:
                pass

            # Label words in the window
            with buffer_lock:
                for w in word_results:
                    if w[1] >= window_start_time and w[2] <= window_start_time + DIARIZATION_WINDOW:
                        midpoint = (w[1] + w[2]) / 2
                        for s_start, s_end, spk in segments:
                            if s_start <= (midpoint - window_start_time) <= s_end:
                                w[3] = spk
                                break

                # Print newly labeled segments (since last_labeled_idx)
                _print_newly_labeled(word_results, last_labeled_idx, label="  >> ")
                last_labeled_idx = len(word_results)

    def _print_newly_labeled(words, from_idx, label="  "):
        """Group and print words that now have speaker labels, starting from from_idx."""
        newly = [w for w in words[from_idx:] if w[3] is not None]
        if not newly:
            return
        word_dicts = [
            {"word": w[0], "start": w[1], "end": w[2], "speaker": w[3]}
            for w in newly
        ]
        segs = group_words_by_speaker(word_dicts)
        for seg in segs:
            spk = seg["speaker"] or "UNKNOWN"
            color = _speaker_color(spk)
            spk_label = _format_speaker(spk)
            ts = f"{seg['start']:6.1f}s-{seg['end']:6.1f}s"
            print(f"{label}{color}[{spk_label}]{RESET_COLOR} {seg['text']}")

    # Start diarization thread
    diar_thread = None
    if diar_pipeline:
        diar_thread = threading.Thread(target=diarization_worker, daemon=True)
        diar_thread.start()

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while True:
                data = audio_queue.get()
                # Accumulate audio for diarization
                if diar_pipeline:
                    chunk = np.frombuffer(data, dtype=np.int16)
                    with buffer_lock:
                        audio_buffer = np.append(audio_buffer, chunk)

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    words = result.get("result", [])

                    if diar_pipeline and words:
                        with buffer_lock:
                            for w in words:
                                word_results.append([w["word"], w["start"], w["end"], None])
                            # Check if any of these words already have labels
                            # (from a diarization pass that ran while Vosk was buffering)
                            _print_newly_labeled(word_results, last_labeled_idx, label="  ")
                            last_labeled_idx = len(word_results)
                            if not any(w[3] is not None for w in word_results[-len(words):]):
                                # No speaker info yet — print plain, diarization will catch up
                                print(f"  {text}")
                    elif text:
                        print(f"  {text}")
                else:
                    partial = json.loads(recognizer.PartialResult())
                    text = partial.get("partial", "")
                    if text:
                        print(f"\r  [partial] {text}", end="", flush=True)

    except KeyboardInterrupt:
        pass

    # Flush remaining audio
    final = json.loads(recognizer.FinalResult())
    if final.get("text"):
        words = final.get("result", [])
        if diar_pipeline:
            with buffer_lock:
                for w in words:
                    word_results.append([w["word"], w["start"], w["end"], None])
        print(f"\n  {final['text']}")

    stop_event.set()
    if diar_thread:
        diar_thread.join(timeout=5.0)

    # Print final speaker-labeled transcript
    if diar_pipeline and word_results:
        print("\n" + "=" * 60)
        print("Final Speaker-labeled Transcript")
        print("=" * 60)
        word_dicts = [
            {"word": w[0], "start": w[1], "end": w[2], "speaker": w[3]}
            for w in word_results
        ]
        segments = group_words_by_speaker(word_dicts)
        for seg in segments:
            spk = seg["speaker"] or "UNKNOWN"
            color = _speaker_color(spk)
            label = _format_speaker(spk)
            ts = f"{seg['start']:6.1f}s - {seg['end']:6.1f}s"
            print(f"  {color}[{ts}] [{label}]{RESET_COLOR} {seg['text']}")

    print("\nStopped.")


# ---------------------------------------------------------------------------
# Mode 2 — WAV file transcription (optionally with diarization)
# ---------------------------------------------------------------------------
def transcribe_file(wav_path: str, model_path: str, diarize: bool = False,
                    hf_token: str = "", num_speakers=None):
    SetLogLevel(0)

    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Download from https://alphacephei.com/vosk/models")
        sys.exit(1)

    if not os.path.exists(wav_path):
        print(f"Audio file not found: {wav_path}")
        sys.exit(1)

    print(f"Loading Vosk model from {model_path} ...")
    model = Model(model_path)

    wf = wave.open(wav_path, "rb")
    channels = wf.getnchannels()
    sample_rate = wf.getframerate()
    sample_width = wf.getsampwidth()

    # Vosk requires 16 kHz mono 16-bit PCM
    needs_convert = channels != 1 or sample_rate != 16000 or sample_width != 2
    if needs_convert:
        print(
            f"WARNING: WAV format is channels={channels}, rate={sample_rate}, "
            f"width={sample_width}. Vosk expects mono 16kHz 16-bit PCM."
        )
        print("Converting on the fly ...")

    recognizer = KaldiRecognizer(model, 16000)
    recognizer.SetWords(True)

    print(f"Transcribing: {wav_path}\n")

    all_words = []
    # Also collect raw audio for diarization
    raw_audio_chunks = [] if diarize else None

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break

        if needs_convert:
            audio = np.frombuffer(data, dtype=np.int16)
            if channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
            if sample_rate != 16000:
                ratio = 16000 / sample_rate
                indices = np.arange(0, len(audio), 1 / ratio)
                indices = indices[indices < len(audio)].astype(int)
                audio = audio[indices]
            data = audio.tobytes()

        if diarize:
            raw_audio_chunks.append(np.frombuffer(data, dtype=np.int16).copy())

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text:
                print(f"  {text}")
            words = result.get("result", [])
            all_words.extend(words)

    # Final result
    final = json.loads(recognizer.FinalResult())
    if final.get("text"):
        print(f"  {final['text']}")
    all_words.extend(final.get("result", []))

    wf.close()

    # Save raw word timestamps
    output_path = wav_path.rsplit(".", 1)[0] + "_vosk.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for w in all_words:
            f.write(f"{w['start']:.2f}\t{w['end']:.2f}\t{w['word']}\n")
    print(f"\nWord timestamps saved to: {output_path}")

    # --- Diarization ---
    if diarize:
        if not hf_token:
            hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            print("ERROR: --hf-token (or HF_TOKEN env var) required for diarization")
            sys.exit(1)

        diar_pipeline = load_diarization_pipeline(hf_token)

        # Concatenate audio and run diarization
        full_audio = np.concatenate(raw_audio_chunks)
        print(f"\nRunning diarization on {len(full_audio)/SAMPLE_RATE:.1f}s of audio ...")

        diar_segments = diarize_waveform(diar_pipeline, full_audio, SAMPLE_RATE, num_speakers)

        unique_speakers = sorted(set(s for _, _, s in diar_segments))
        print(f"  Detected {len(unique_speakers)} speaker(s): {unique_speakers}")

        # Map speakers to words
        labeled_words = map_speakers_to_words(all_words, diar_segments)
        segments = group_words_by_speaker(labeled_words)

        # Print speaker-labeled output
        print("\n" + "=" * 60)
        print("Speaker-labeled Transcription")
        print("=" * 60)
        for seg in segments:
            spk = seg["speaker"] or "UNKNOWN"
            color = _speaker_color(spk)
            label = _format_speaker(spk)
            ts = f"{seg['start']:6.1f}s - {seg['end']:6.1f}s"
            print(f"  {color}[{ts}] [{label}]{RESET_COLOR} {seg['text']}")

        # Per-speaker summary
        print("\n" + "-" * 60)
        print("Per-Speaker Summary")
        print("-" * 60)
        speaker_texts = {}
        for seg in segments:
            spk = seg["speaker"] or "UNKNOWN"
            speaker_texts.setdefault(spk, []).append(seg["text"])
        for spk in sorted(speaker_texts.keys()):
            color = _speaker_color(spk)
            label = _format_speaker(spk)
            full_text = " ".join(speaker_texts[spk])
            print(f"\n  {color}{label}{RESET_COLOR}: {full_text}")

        # Save labeled output
        labeled_output_path = wav_path.rsplit(".", 1)[0] + "_vosk_labeled.txt"
        with open(labeled_output_path, "w", encoding="utf-8") as f:
            f.write("Speaker-labeled Transcription\n")
            f.write("=" * 50 + "\n\n")
            for seg in segments:
                spk = seg["speaker"] or "UNKNOWN"
                ts = f"{seg['start']:6.1f}s - {seg['end']:6.1f}s"
                f.write(f"[{ts}] [{spk}] {seg['text']}\n")
            f.write("\n\n" + "=" * 50 + "\nPer-Speaker Summary\n" + "=" * 50 + "\n")
            for spk in sorted(speaker_texts.keys()):
                f.write(f"\n{spk}:\n  {' '.join(speaker_texts[spk])}\n")
        print(f"\nLabeled transcript saved to: {labeled_output_path}")


# ---------------------------------------------------------------------------
# Mode 3 — Speaker / system audio streaming (WASAPI loopback, Windows)
# ---------------------------------------------------------------------------
def list_speaker_devices():
    """List available WASAPI loopback (speaker) devices."""
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        print("ERROR: pyaudiowpatch is required for speaker mode.")
        print("  pip install pyaudiowpatch")
        sys.exit(1)

    p = pyaudio.PyAudio()
    try:
        devices = list(p.get_loopback_device_info_generator())
        if not devices:
            print("No WASAPI loopback devices found.")
            return
        print(f"Found {len(devices)} WASAPI loopback device(s):\n")
        for d in devices:
            host_api = p.get_host_api_info_by_index(d["hostApi"])
            print(f"  [{d['index']:2d}] {d['name']}")
            print(f"       channels={d['maxInputChannels']}, rate={d['defaultSampleRate']}, host={host_api['name']}")
        print()
    finally:
        p.terminate()


def speaker_streaming(model_path: str, device_index: int = None,
                      diarize: bool = False, hf_token: str = "",
                      num_speakers=None, save_wav: bool = False):
    """Capture system audio via WASAPI loopback and transcribe with Vosk."""
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        print("ERROR: pyaudiowpatch is required for speaker mode.")
        print("  pip install pyaudiowpatch")
        sys.exit(1)

    SetLogLevel(0)

    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Download from https://alphacephei.com/vosk/models")
        sys.exit(1)

    # Discover loopback device
    p = pyaudio.PyAudio()
    try:
        loopback_devices = list(p.get_loopback_device_info_generator())
        if not loopback_devices:
            print("No WASAPI loopback devices found. Are you on Windows?")
            p.terminate()
            sys.exit(1)

        if device_index is None:
            # Auto-select first loopback device
            dev_info = loopback_devices[0]
            device_index = dev_info["index"]
        else:
            dev_info = None
            for d in loopback_devices:
                if d["index"] == device_index:
                    dev_info = d
                    break
            if dev_info is None:
                print(f"Device index {device_index} not found among loopback devices.")
                print("Use --list-devices to see available devices.")
                p.terminate()
                sys.exit(1)
    finally:
        p.terminate()

    source_channels = dev_info["maxInputChannels"]
    source_rate = int(dev_info["defaultSampleRate"])
    print(f"Loopback device: [{dev_info['index']}] {dev_info['name']}")
    print(f"  channels={source_channels}, rate={source_rate}")

    # Load Vosk model
    print(f"\nLoading Vosk model from {model_path} ...")
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)

    # Diarization setup
    diar_pipeline = None
    if diarize:
        if not hf_token:
            hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            print("ERROR: --hf-token (or HF_TOKEN env var) required for diarization")
            sys.exit(1)
        diar_pipeline = load_diarization_pipeline(hf_token)

    # Shared state
    word_results = []
    audio_buffer = np.empty((0,), dtype=np.int16)
    buffer_lock = threading.Lock()
    stop_event = threading.Event()
    last_labeled_idx = 0

    # WAV save
    wav_file = None
    wav_path = None

    def diarization_worker():
        nonlocal last_labeled_idx
        while not stop_event.is_set():
            time.sleep(DIARIZATION_INTERVAL)

            with buffer_lock:
                window_samples = int(DIARIZATION_WINDOW * SAMPLE_RATE)
                if len(audio_buffer) < window_samples:
                    continue
                window = audio_buffer[-window_samples:].copy()
                window_start_time = (len(audio_buffer) - window_samples) / SAMPLE_RATE

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav = tmp.name
                with wave.open(tmp_wav, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(window.tobytes())

            try:
                segments = diarize_wav_file(diar_pipeline, tmp_wav, num_speakers)
            except Exception as e:
                print(f"\n[Diarization error] {e}", file=sys.stderr)
                try:
                    os.unlink(tmp_wav)
                except OSError:
                    pass
                continue

            try:
                os.unlink(tmp_wav)
            except OSError:
                pass

            with buffer_lock:
                for w in word_results:
                    if w[1] >= window_start_time and w[2] <= window_start_time + DIARIZATION_WINDOW:
                        midpoint = (w[1] + w[2]) / 2
                        for s_start, s_end, spk in segments:
                            if s_start <= (midpoint - window_start_time) <= s_end:
                                w[3] = spk
                                break
                _print_newly_labeled(word_results, last_labeled_idx, label="  >> ")
                last_labeled_idx = len(word_results)

    def _print_newly_labeled(words, from_idx, label="  "):
        newly = [w for w in words[from_idx:] if w[3] is not None]
        if not newly:
            return
        word_dicts = [
            {"word": w[0], "start": w[1], "end": w[2], "speaker": w[3]}
            for w in newly
        ]
        segs = group_words_by_speaker(word_dicts)
        for seg in segs:
            spk = seg["speaker"] or "UNKNOWN"
            color = _speaker_color(spk)
            spk_label = _format_speaker(spk)
            ts = f"{seg['start']:6.1f}s-{seg['end']:6.1f}s"
            print(f"{label}{color}[{spk_label}]{RESET_COLOR} {seg['text']}")

    # Start diarization thread
    diar_thread = None
    if diar_pipeline:
        diar_thread = threading.Thread(target=diarization_worker, daemon=True)
        diar_thread.start()

    # Open WASAPI loopback stream
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=source_channels,
        rate=source_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=1024,
    )

    if save_wav:
        os.makedirs("records", exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join("records", f"speaker_{timestamp}.wav")
        wav_file = wave.open(wav_path, "wb")
        wav_file.setnchannels(source_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(source_rate)
        print(f"Saving audio to: {wav_path}")

    print(f"\nCapturing system audio. Press Ctrl+C to stop.\n")

    resample_ratio = SAMPLE_RATE / source_rate
    chunk_buffer = np.array([], dtype=np.int16)
    samples_needed = BLOCK_SIZE  # 0.5s at 16kHz

    try:
        while True:
            try:
                raw_data = stream.read(1024, exception_on_overflow=False)
            except Exception as e:
                print(f"Read error: {e}", file=sys.stderr)
                time.sleep(0.01)
                continue

            raw_chunk = np.frombuffer(raw_data, dtype=np.int16)

            # Save original audio
            if wav_file:
                wav_file.writeframes(raw_data)

            # Convert to mono
            if source_channels == 2:
                mono_chunk = raw_chunk.reshape(-1, 2).mean(axis=1).astype(np.int16)
            else:
                mono_chunk = raw_chunk

            # Resample to 16kHz
            if source_rate != SAMPLE_RATE:
                indices = np.arange(0, len(mono_chunk), 1 / resample_ratio)
                indices = indices[indices < len(mono_chunk)].astype(int)
                resampled = np.interp(indices, np.arange(len(mono_chunk)), mono_chunk).astype(np.int16)
            else:
                resampled = mono_chunk

            # Accumulate and feed to Vosk in BLOCK_SIZE chunks
            chunk_buffer = np.concatenate([chunk_buffer, resampled])

            while len(chunk_buffer) >= samples_needed:
                audio_chunk = chunk_buffer[:samples_needed]
                chunk_buffer = chunk_buffer[samples_needed:]
                data = audio_chunk.tobytes()

                # Accumulate for diarization
                if diar_pipeline:
                    with buffer_lock:
                        audio_buffer = np.append(audio_buffer, audio_chunk)

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    words = result.get("result", [])

                    if diar_pipeline and words:
                        with buffer_lock:
                            for w in words:
                                word_results.append([w["word"], w["start"], w["end"], None])
                            _print_newly_labeled(word_results, last_labeled_idx, label="  ")
                            last_labeled_idx = len(word_results)
                            if not any(w[3] is not None for w in word_results[-len(words):]):
                                print(f"  {text}")
                    elif text:
                        print(f"  {text}")
                else:
                    partial = json.loads(recognizer.PartialResult())
                    text = partial.get("partial", "")
                    if text:
                        print(f"\r  [partial] {text}", end="", flush=True)

    except KeyboardInterrupt:
        pass

    # Flush remaining audio
    final = json.loads(recognizer.FinalResult())
    if final.get("text"):
        words = final.get("result", [])
        if diar_pipeline:
            with buffer_lock:
                for w in words:
                    word_results.append([w["word"], w["start"], w["end"], None])
        print(f"\n  {final['text']}")

    stop_event.set()
    stream.stop_stream()
    stream.close()
    p.terminate()

    if wav_file:
        wav_file.close()
        print(f"Audio saved to: {wav_path}")

    if diar_thread:
        diar_thread.join(timeout=5.0)

    # Print final speaker-labeled transcript
    if diar_pipeline and word_results:
        print("\n" + "=" * 60)
        print("Final Speaker-labeled Transcript")
        print("=" * 60)
        word_dicts = [
            {"word": w[0], "start": w[1], "end": w[2], "speaker": w[3]}
            for w in word_results
        ]
        segments = group_words_by_speaker(word_dicts)
        for seg in segments:
            spk = seg["speaker"] or "UNKNOWN"
            color = _speaker_color(spk)
            label = _format_speaker(spk)
            ts = f"{seg['start']:6.1f}s - {seg['end']:6.1f}s"
            print(f"  {color}[{ts}] [{label}]{RESET_COLOR} {seg['text']}")

    print("\nStopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Vosk ASR Sample (with optional speaker diarization)"
    )
    parser.add_argument(
        "mode",
        choices=["mic", "speaker", "file"],
        help="'mic' for live microphone, 'speaker' for system audio (WASAPI loopback), 'file' for WAV transcription",
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=None,
        help="Path to WAV file (required for 'file' mode)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to Vosk model directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (requires pyannote.audio + HF token)",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        help="HuggingFace token for pyannote.audio (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Hint for number of speakers (optional, for diarization)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio device index for 'speaker' mode (use --list-devices to see options)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available WASAPI loopback devices and exit (speaker mode)",
    )
    parser.add_argument(
        "--save-wav",
        action="store_true",
        help="Save captured speaker audio to WAV file (speaker mode only)",
    )
    args = parser.parse_args()

    # --list-devices is a standalone flag
    if args.list_devices:
        list_speaker_devices()
        return

    if args.mode == "mic":
        mic_streaming(args.model, diarize=args.diarize, hf_token=args.hf_token,
                      num_speakers=args.num_speakers)
    elif args.mode == "speaker":
        speaker_streaming(args.model, device_index=args.device,
                          diarize=args.diarize, hf_token=args.hf_token,
                          num_speakers=args.num_speakers, save_wav=args.save_wav)
    else:
        if not args.audio_file:
            parser.error("'file' mode requires a WAV file path")
        transcribe_file(args.audio_file, args.model, diarize=args.diarize,
                        hf_token=args.hf_token, num_speakers=args.num_speakers)


if __name__ == "__main__":
    main()
