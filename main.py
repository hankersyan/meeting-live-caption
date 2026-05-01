"""
Real-time Meeting Text Recorder with Audio and Text Saving
Records system audio via WASAPI loopback, transcribes with Whisper,
and saves both audio (WAV) and transcription (TXT) in a 'records' folder.
"""

import sys
import os
import queue
import threading
import time
import wave
import json
import urllib.request
import urllib.error
import socket
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pyaudiowpatch as pyaudio

# GUI
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess

# Whisper
from faster_whisper import WhisperModel

# Suppress noisy torchcodec warning from pyannote.audio (unneeded: we feed in-memory waveforms)
import warnings
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")

# Optional: pyannote.audio for speaker diarization
try:
    from pyannote.audio import Pipeline
    HAS_PYANNOTE = True
except ImportError:
    HAS_PYANNOTE = False


# ------------------------------
# Data Structures for Speaker Diarization
# ------------------------------

@dataclass
class TranscriptionSegment:
    """A single transcribed segment with timing and optional speaker label."""
    start: float         # absolute start time (seconds)
    end: float           # absolute end time (seconds)
    text: str            # transcribed text content
    speaker: Optional[str] = None  # speaker label (filled by LabelMapper)


class SegmentStore:
    """Thread-safe store for transcription segments."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._segments: List[TranscriptionSegment] = []
    
    def add(self, segment: TranscriptionSegment):
        with self._lock:
            self._segments.append(segment)
    
    def add_all(self, segments: List[TranscriptionSegment]):
        with self._lock:
            self._segments.extend(segments)
    
    def get_all(self) -> List[TranscriptionSegment]:
        with self._lock:
            return list(self._segments)
    
    def get_unlabeled(self) -> List[TranscriptionSegment]:
        """Return segments that haven't been assigned a speaker label yet."""
        with self._lock:
            return [s for s in self._segments if s.speaker is None]
    
    def apply_speaker_labels(self, labels: List[Tuple[int, str]]):
        """Apply speaker labels by segment index. labels = [(index, speaker), ...]"""
        with self._lock:
            for idx, speaker in labels:
                if 0 <= idx < len(self._segments):
                    self._segments[idx].speaker = speaker
    
    def clear(self):
        with self._lock:
            self._segments.clear()
    
    def __len__(self):
        with self._lock:
            return len(self._segments)


class AudioBuffer:
    """Ring buffer that accumulates mono audio for diarization processing."""
    
    def __init__(self, max_duration: float = 120.0, sample_rate: int = 16000):
        self.max_samples = int(max_duration * sample_rate)
        self.sample_rate = sample_rate
        self._buffer = np.array([], dtype=np.int16)
        self._lock = threading.Lock()
    
    def append(self, chunk: np.ndarray):
        with self._lock:
            self._buffer = np.concatenate([self._buffer, chunk])
            if len(self._buffer) > self.max_samples:
                self._buffer = self._buffer[-self.max_samples:]
    
    def get_audio(self) -> np.ndarray:
        with self._lock:
            return self._buffer.copy()
    
    def clear(self):
        with self._lock:
            self._buffer = np.array([], dtype=np.int16)
    
    def duration(self) -> float:
        with self._lock:
            return len(self._buffer) / self.sample_rate


# ------------------------------
# Audio Recording Thread (with file saving)
# ------------------------------
class AudioRecorder:
    """Handles WASAPI loopback recording, pushes chunks to queue, and saves to WAV."""
    
    def __init__(self, device_index, sample_rate=16000, chunk_duration=2.0, save_audio=True):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.save_audio = save_audio
        
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        self.p = None
        self.stream = None
        self.thread = None
        
        # File writing attributes
        self.wave_file = None
        self.base_filename = None      # Without extension
        self.wave_fullpath = None
        self.write_thread = None
        self.write_queue = queue.Queue()
        self.source_rate = None
        self.channels = None
        
    def start(self, records_folder="records"):
        """Start recording in background threads."""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.audio_queue = queue.Queue()
        self.write_queue = queue.Queue()
        
        # Generate base filename with timestamp inside records folder
        if self.save_audio:
            os.makedirs(records_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.base_filename = f"meeting_{timestamp}"
            self.wave_fullpath = os.path.join(records_folder, self.base_filename + ".wav")
        
        # Start main recording thread
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        
        # Start file writer thread
        if self.save_audio:
            self.write_thread = threading.Thread(target=self._file_writer_loop, daemon=True)
            self.write_thread.start()
        
    def stop(self):
        """Stop recording and clean up."""
        self.is_recording = False
        if self.thread:
            self.thread.join(timeout=3.0)
        if self.write_thread:
            self.write_queue.put(None)  # Sentinel to stop writer
            self.write_thread.join(timeout=2.0)
        if self.wave_file:
            self.wave_file.close()
            self.wave_file = None
            
    def _record_loop(self):
        """Main recording loop."""
        try:
            self.p = pyaudio.PyAudio()
            device_info = self.p.get_device_info_by_index(self.device_index)
            self.channels = device_info["maxInputChannels"]
            if self.channels <= 0:
                raise ValueError("Selected device has no input channels.")
            
            self.source_rate = int(device_info["defaultSampleRate"])
            
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.source_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=1024,
            )
            
            if self.save_audio:
                self.wave_file = wave.open(self.wave_fullpath, 'wb')
                self.wave_file.setnchannels(self.channels)
                self.wave_file.setsampwidth(2)
                self.wave_file.setframerate(self.source_rate)
            
            resample_ratio = self.sample_rate / self.source_rate
            buffer = np.array([], dtype=np.int16)
            samples_needed = self.chunk_size
            
            while self.is_recording:
                try:
                    data = self.stream.read(1024, exception_on_overflow=False)
                    raw_chunk = np.frombuffer(data, dtype=np.int16)
                    
                    if self.save_audio:
                        self.write_queue.put(data)
                    
                    if self.channels == 2:
                        mono_chunk = raw_chunk.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    else:
                        mono_chunk = raw_chunk
                    
                    if self.source_rate != self.sample_rate:
                        indices = np.arange(0, len(mono_chunk), 1 / resample_ratio)
                        indices = indices[indices < len(mono_chunk)]
                        resampled = np.interp(indices, np.arange(len(mono_chunk)), mono_chunk).astype(np.int16)
                    else:
                        resampled = mono_chunk
                    
                    buffer = np.concatenate([buffer, resampled])
                    
                    while len(buffer) >= samples_needed:
                        audio_chunk = buffer[:samples_needed].copy()
                        buffer = buffer[samples_needed:]
                        self.audio_queue.put(audio_chunk)
                        
                except Exception as e:
                    print(f"Recording error: {e}")
                    time.sleep(0.01)
                    
        except Exception as e:
            print(f"Recorder thread error: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
            if self.save_audio:
                self.write_queue.put(None)
    
    def _file_writer_loop(self):
        """Write audio data to WAV file without blocking."""
        while True:
            item = self.write_queue.get()
            if item is None:
                break
            try:
                if self.wave_file:
                    self.wave_file.writeframes(item)
            except Exception as e:
                print(f"File write error: {e}")
    
    def get_base_filename(self):
        """Return the base filename (without extension)."""
        return self.base_filename
    
    def get_wav_path(self):
        """Return full path to WAV file."""
        return self.wave_fullpath


# ------------------------------
# Transcription Thread (with text saving)
# ------------------------------
class Transcriber:
    """Processes audio chunks from queue, performs Whisper transcription, and saves text."""
    
    def __init__(
        self,
        recorder: AudioRecorder,
        model_size="base",
        language="en",
        records_folder="records",
        whisper_device="cpu",
        local_files_only=False,
        segment_store: Optional[SegmentStore] = None,
        audio_buffer: Optional[AudioBuffer] = None,
    ):
        self.recorder = recorder
        self.model_size = model_size
        self.language = language
        self.records_folder = records_folder
        self.whisper_device = whisper_device
        self.local_files_only = local_files_only
        self.is_running = False
        self.thread = None
        
        self.text_callback = None
        self.model = None
        self.text_file = None
        self.text_path = None
        
        # Speaker diarization support
        self.segment_store = segment_store
        self.audio_buffer = audio_buffer
        self._time_offset = 0.0       # cumulative absolute time (seconds)
        self._overlap_duration = 0.5  # overlap between chunks
        
    def set_text_callback(self, callback):
        self.text_callback = callback
        
    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._time_offset = 0.0
        
        # Prepare text file for writing
        base = self.recorder.get_base_filename()
        if base:
            os.makedirs(self.records_folder, exist_ok=True)
            self.text_path = os.path.join(self.records_folder, base + ".txt")
            self.text_file = open(self.text_path, "w", encoding="utf-8")
            # Write a small header
            self.text_file.write(f"Meeting Transcription - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.text_file.write("=" * 50 + "\n\n")
            self.text_file.flush()
        
        self.thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=3.0)
        if self.text_file:
            self.text_file.write("\n\n" + "=" * 50 + "\n")
            self.text_file.write(f"Recording ended - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.text_file.close()
            self.text_file = None
            
    def _save_text(self, text):
        """Thread-safe method to write text to file."""
        try:
            if self.text_file:
                self.text_file.write(text)
                self.text_file.flush()
        except Exception as e:
            print(f"Error writing text file: {e}")
            
    def _transcribe_loop(self):
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.whisper_device,
                compute_type="int8",
                cpu_threads=4,
                num_workers=1,
                local_files_only=self.local_files_only,
            )
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            if self.text_callback:
                self.text_callback(f"[Error] Failed to load model: {e}\n")
            return
        
        overlap_samples = int(self.recorder.sample_rate * self._overlap_duration)
        prev_chunk = np.array([], dtype=np.int16)
        chunk_idx = 0
        
        while self.is_running:
            try:
                try:
                    audio_chunk = self.recorder.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Push raw audio to the diarization buffer
                if self.audio_buffer is not None:
                    self.audio_buffer.append(audio_chunk)
                
                if len(prev_chunk) > 0:
                    combined = np.concatenate([prev_chunk[-overlap_samples:], audio_chunk])
                else:
                    combined = audio_chunk
                
                audio_float = combined.astype(np.float32) / 32768.0
                
                segments, _ = self.model.transcribe(
                    audio_float,
                    language=self.language,
                    beam_size=5,
                    best_of=5,
                    temperature=0.0,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                
                full_text = ""
                chunk_segments = []
                
                for seg in segments:
                    # Calculate absolute timestamps
                    if chunk_idx == 0:
                        abs_start = seg.start
                        abs_end = seg.end
                    else:
                        abs_start = self._time_offset - self._overlap_duration + seg.start
                        abs_end = self._time_offset - self._overlap_duration + seg.end
                    
                    chunk_segments.append(TranscriptionSegment(
                        start=abs_start, end=abs_end, text=seg.text
                    ))
                    full_text += seg.text
                
                # Store segments if store is available
                if self.segment_store is not None and chunk_segments:
                    self.segment_store.add_all(chunk_segments)
                
                if full_text.strip():
                    full_text = full_text.strip() + " "
                    # Update UI
                    if self.text_callback:
                        self.text_callback(full_text)
                    # Save to file
                    self._save_text(full_text)
                
                prev_chunk = audio_chunk
                self._time_offset += self.recorder.chunk_duration
                chunk_idx += 1
                
            except Exception as e:
                print(f"Transcription error: {e}")
                time.sleep(0.1)


class KeyPointExtractor:
    """Periodically extracts brief key points from live captions via Ollama."""

    def __init__(self, interval_getter, text_provider, extract_callback, output_callback):
        self.interval_getter = interval_getter
        self.text_provider = text_provider
        self.extract_callback = extract_callback
        self.output_callback = output_callback
        self.is_running = False
        self.thread = None
        self._last_input_token = 0
        self._last_output = ""
        self._stop_event = threading.Event()

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=3.0)

    def _run_loop(self):
        while self.is_running:
            try:
                captions, input_token = self.text_provider()
                if captions and input_token > self._last_input_token:
                    key_points = self.extract_callback(captions)
                    if key_points:
                        normalized = key_points.strip()
                        if normalized and normalized != self._last_output:
                            self.output_callback(normalized)
                            self._last_output = normalized
                    self._last_input_token = input_token
                interval = max(3.0, self.interval_getter())
                self._stop_event.wait(interval)
            except Exception as e:
                self.output_callback(f"[Key point extraction error] {e}")
                self._stop_event.wait(5.0)


# ------------------------------
# Speaker Diarization (pyannote.audio)
# ------------------------------

def map_speakers_to_segments(
    segments: List[TranscriptionSegment],
    diarization_results: List[Tuple[float, float, str]],
    time_eps: float = 0.3,
) -> List[TranscriptionSegment]:
    """
    Map speaker labels from diarization results onto transcription segments
    based on timestamp overlap.

    Args:
        segments: List of transcription segments with absolute timestamps.
        diarization_results: List of (start, end, speaker_label) tuples.
        time_eps: Minimum overlap duration (seconds) to assign a speaker.

    Returns:
        A new list of TranscriptionSegments with speaker labels filled in.
    """
    labeled = [TranscriptionSegment(s.start, s.end, s.text, None) for s in segments]

    for seg in labeled:
        best_speaker = None
        max_overlap = 0.0
        for d_start, d_end, speaker in diarization_results:
            overlap_start = max(seg.start, d_start)
            overlap_end = min(seg.end, d_end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = speaker

        if max_overlap >= time_eps and best_speaker is not None:
            seg.speaker = best_speaker

    return labeled


class SpeakerDiarizer:
    """
    Periodically runs pyannote.audio speaker diarization on accumulated audio,
    then maps results to transcription segments via LabelMapper.
    """

    def __init__(
        self,
        hf_token: str,
        audio_buffer: AudioBuffer,
        segment_store: SegmentStore,
        diarization_callback,
        interval: float = 30.0,
        device: str = "cpu",
        max_speakers: Optional[int] = None,
    ):
        self.hf_token = hf_token
        self.audio_buffer = audio_buffer
        self.segment_store = segment_store
        self.diarization_callback = diarization_callback
        self.interval = interval
        self.device = device
        self.max_speakers = max_speakers

        self.is_running = False
        self.thread = None
        self._stop_event = threading.Event()
        self.pipeline = None
        
        # Track last-processed segment count to only process new audio
        self._last_segment_count = 0

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._stop_event.clear()
        self._last_segment_count = 0
        # Load pipeline and start thread
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)

    def _load_pipeline(self):
        """Load the pyannote speaker-diarization pipeline (called once in background thread)."""
        if self.pipeline is not None:
            return True
        try:
            if not HAS_PYANNOTE:
                self.diarization_callback("[Diarization] pyannote.audio is not installed. Install with: pip install pyannote.audio torch")
                return False
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token,
            )
            if self.device == "cuda":
                try:
                    import torch
                    self.pipeline.to(torch.device("cuda"))
                except Exception:
                    pass  # fall back to CPU
            return True
        except Exception as e:
            self.diarization_callback(f"[Diarization] Failed to load pipeline: {e}")
            return False

    def _run_loop(self):
        """Background loop: load pipeline, then periodically run diarization."""
        # First, load the pipeline (may take 10-30 seconds)
        self.diarization_callback("[Diarization] Loading pyannote pipeline (this may take a moment)...")
        pipeline_ok = self._load_pipeline()
        if not pipeline_ok:
            self.diarization_callback("[Diarization] Pipeline load failed, diarization disabled.")
            return
        
        self.diarization_callback("[Diarization] Pipeline loaded. Starting periodic speaker identification...")

        while self.is_running:
            try:
                audio = self.audio_buffer.get_audio()
                if len(audio) < self.audio_buffer.sample_rate * 3:  # need at least 3s
                    self._stop_event.wait(self.interval)
                    continue

                # Only process if there are new segments to label
                current_count = len(self.segment_store)
                if current_count <= self._last_segment_count:
                    self._stop_event.wait(self.interval)
                    continue

                waveform_np = audio.astype(np.float32) / 32768.0

                # Run diarization
                import torch
                waveform_torch = torch.from_numpy(waveform_np).unsqueeze(0)

                pipeline_kwargs = {
                    "waveform": waveform_torch,
                    "sample_rate": self.audio_buffer.sample_rate,
                }
                if self.max_speakers is not None:
                    pipeline_kwargs["num_speakers"] = self.max_speakers

                output = self.pipeline(pipeline_kwargs)

                # pyannote 3.1 returns DiarizeOutput dataclass; access .speaker_diarization
                if hasattr(output, "speaker_diarization"):
                    annotation = output.speaker_diarization
                else:
                    annotation = output  # fallback: assume Annotation directly

                # Extract results
                results: List[Tuple[float, float, str]] = []
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    results.append((turn.start, turn.end, speaker))

                # Fallback if yield_label returned nothing (rare edge case)
                if not results:
                    for turn, track in annotation.itertracks(yield_label=False):
                        labels = annotation.get_labels(track)
                        speaker = labels[0] if labels else "SPEAKER_UNKNOWN"
                        results.append((turn.start, turn.end, speaker))

                if not results:
                    self._stop_event.wait(self.interval)
                    continue

                # Map speakers to segments
                all_segments = self.segment_store.get_all()
                labeled = map_speakers_to_segments(all_segments, results)

                # Build index-based label updates
                label_updates = []
                for idx, seg in enumerate(labeled):
                    if seg.speaker is not None:
                        old_speaker = all_segments[idx].speaker
                        if old_speaker is None or old_speaker != seg.speaker:
                            label_updates.append((idx, seg.speaker))

                if label_updates:
                    self.segment_store.apply_speaker_labels(label_updates)
                    self.diarization_callback((labeled, label_updates))

                self._last_segment_count = current_count

            except Exception as e:
                self.diarization_callback(f"[Diarization] Error: {e}")

            self._stop_event.wait(self.interval)


# ------------------------------
# Main Application Window
# ------------------------------
class MeetingRecorderApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Meeting Live Caption")
        self.root.geometry("850x650")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.records_folder = "records"   # Folder for all output files
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        
        self.p = None
        self.recorder = None
        self.transcriber = None
        self.key_point_extractor = None
        self.speaker_diarizer = None
        self.segment_store = None
        self.audio_buffer = None
        self.is_recording = False
        self.available_devices = []
        self.caption_lock = threading.Lock()
        self.full_transcription_text = ""
        self.caption_total_chars = 0
        self._suspend_config_autosave = True
        self._config_save_after_id = None
        self.style = ttk.Style()
        
        # Bounds and batching for long-running stability
        self.MAX_CAPTION_HISTORY_CHARS = 200000
        self.MAX_TEXT_AREA_CHARS = 5000
        self.MAX_KEY_POINTS_CHARS = 2000
        self.MAX_SPEAKER_CHARS = 5000
        self.FLUSH_INTERVAL_MS = 200
        
        self._pending_lock = threading.Lock()
        self._pending_text_buffer = []
        self._flush_after_id = None
        
        self.setup_ui()
        self.load_config()
        self.setup_config_autosave()
        self._suspend_config_autosave = False
        self.apply_theme()
        self.apply_extractor_ui_state()
        self.apply_diarize_ui_state()
        self.refresh_devices()
        
    def setup_ui(self):
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(fill=tk.X)
        
        # Row 0: Device selection
        ttk.Label(self.control_frame, text="Audio Device:").grid(row=0, column=0, sticky=tk.W, padx=(0,5))
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(self.control_frame, textvariable=self.device_var, state="readonly", width=60)
        self.device_combo.grid(row=0, column=1, sticky=tk.W, padx=(0,10))
        self.device_combo.bind("<<ComboboxSelected>>", self.on_device_selected)
        ttk.Button(self.control_frame, text="Refresh", command=self.refresh_devices).grid(row=0, column=2, padx=(0,5))
        
        # Model & Language
        ttk.Label(self.control_frame, text="Model:").grid(row=0, column=3, sticky=tk.W, padx=(10,5))
        self.model_var = tk.StringVar(value="base")
        model_combo = ttk.Combobox(self.control_frame, textvariable=self.model_var,
                                   values=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                                   state="readonly", width=10)
        model_combo.grid(row=0, column=4, sticky=tk.W, padx=(0,5))
        
        ttk.Label(self.control_frame, text="Language:").grid(row=0, column=5, sticky=tk.W, padx=(10,5))
        self.lang_var = tk.StringVar(value="en")
        lang_combo = ttk.Combobox(self.control_frame, textvariable=self.lang_var,
                                  values=["en", "zh", "es", "fr", "de", "ja", "ko", "auto"],
                                  state="readonly", width=5)
        lang_combo.grid(row=0, column=6, sticky=tk.W)

        # Whisper runtime options
        self.whisper_opt_frame = ttk.Frame(self.root, padding="5")
        self.whisper_opt_frame.pack(fill=tk.X, padx=10)
        ttk.Label(self.whisper_opt_frame, text="Whisper Device:").pack(side=tk.LEFT)
        self.whisper_device_var = tk.StringVar(value="cpu")
        self.whisper_device_combo = ttk.Combobox(
            self.whisper_opt_frame,
            textvariable=self.whisper_device_var,
            values=["cpu", "cuda", "auto"],
            state="readonly",
            width=8,
        )
        self.whisper_device_combo.pack(side=tk.LEFT, padx=(5, 12))
        self.whisper_local_only_var = tk.BooleanVar(value=False)
        self.whisper_local_only_check = ttk.Checkbutton(
            self.whisper_opt_frame,
            text="Whisper local_files_only",
            variable=self.whisper_local_only_var,
        )
        self.whisper_local_only_check.pack(side=tk.LEFT)
        
        # Row 1: Save options
        self.save_frame = ttk.Frame(self.root, padding="5")
        self.save_frame.pack(fill=tk.X, padx=10)
        self.save_audio_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.save_frame, text="Save audio & text to 'records' folder", variable=self.save_audio_var).pack(side=tk.LEFT)
        self.file_label_var = tk.StringVar(value="No recording yet")
        ttk.Label(self.save_frame, textvariable=self.file_label_var, foreground="blue").pack(side=tk.LEFT, padx=(20,0))
        ttk.Button(self.save_frame, text="Open Records Folder", command=self.open_records_folder).pack(side=tk.RIGHT)

        # Key-point extraction config
        self.extract_cfg = ttk.LabelFrame(self.root, text="Key Point Extraction (Ollama)", padding="8")
        self.extract_cfg.pack(fill=tk.X, padx=10, pady=(0, 5))

        self.extract_enabled_var = tk.BooleanVar(value=True)
        self.extract_enable_check = ttk.Checkbutton(
            self.extract_cfg,
            text="Enable",
            variable=self.extract_enabled_var,
            command=self.on_extractor_enable_toggled,
        )
        self.extract_enable_check.grid(row=0, column=0, sticky=tk.W, padx=(0, 8))

        ttk.Label(self.extract_cfg, text="URL:").grid(row=0, column=1, sticky=tk.W)
        self.ollama_url_var = tk.StringVar(value="http://localhost:11434")
        self.ollama_url_entry = ttk.Entry(self.extract_cfg, textvariable=self.ollama_url_var, width=28)
        self.ollama_url_entry.grid(row=0, column=2, sticky=tk.W, padx=(5, 10))

        ttk.Label(self.extract_cfg, text="Model:").grid(row=0, column=3, sticky=tk.W)
        self.ollama_model_var = tk.StringVar(value="LiquidAI/lfm2.5-1.2b-instruct")
        self.ollama_model_entry = ttk.Entry(self.extract_cfg, textvariable=self.ollama_model_var, width=18)
        self.ollama_model_entry.grid(row=0, column=4, sticky=tk.W, padx=(5, 10))

        ttk.Label(self.extract_cfg, text="Refresh(s):").grid(row=0, column=5, sticky=tk.W)
        self.extract_interval_var = tk.StringVar(value="20")
        self.extract_interval_entry = ttk.Entry(self.extract_cfg, textvariable=self.extract_interval_var, width=8)
        self.extract_interval_entry.grid(row=0, column=6, sticky=tk.W, padx=(5, 0))

        ttk.Label(self.extract_cfg, text="Timeout(s):").grid(row=0, column=7, sticky=tk.W, padx=(10, 0))
        self.extract_timeout_var = tk.StringVar(value="90")
        self.extract_timeout_entry = ttk.Entry(self.extract_cfg, textvariable=self.extract_timeout_var, width=8)
        self.extract_timeout_entry.grid(row=0, column=8, sticky=tk.W, padx=(5, 0))

        ttk.Label(self.extract_cfg, text="Prompt:").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        self.extract_prompt_var = tk.StringVar(
            value="Extract 3-5 brief key points from the live meeting captions. Use concise bullet points and avoid repetition."
        )
        self.extract_prompt_entry = ttk.Entry(self.extract_cfg, textvariable=self.extract_prompt_var, width=120)
        self.extract_prompt_entry.grid(
            row=1, column=1, columnspan=6, sticky=tk.W + tk.E, padx=(5, 0), pady=(6, 0)
        )
        self.extract_cfg.columnconfigure(2, weight=1)
        self.extract_cfg.columnconfigure(4, weight=1)
        
        # Speaker Diarization config
        self.diarize_cfg = ttk.LabelFrame(self.root, text="Speaker Diarization (pyannote.audio)", padding="8")
        self.diarize_cfg.pack(fill=tk.X, padx=10, pady=(0, 5))

        self.diarize_enabled_var = tk.BooleanVar(value=False)
        self.diarize_enable_check = ttk.Checkbutton(
            self.diarize_cfg,
            text="Enable",
            variable=self.diarize_enabled_var,
            command=self.on_diarize_enable_toggled,
        )
        self.diarize_enable_check.grid(row=0, column=0, sticky=tk.W, padx=(0, 8))

        ttk.Label(self.diarize_cfg, text="HF Token:").grid(row=0, column=1, sticky=tk.W)
        self.hf_token_var = tk.StringVar(value="")
        self.hf_token_entry = ttk.Entry(self.diarize_cfg, textvariable=self.hf_token_var, width=30, show="*")
        self.hf_token_entry.grid(row=0, column=2, sticky=tk.W, padx=(5, 10))

        ttk.Label(self.diarize_cfg, text="Interval(s):").grid(row=0, column=3, sticky=tk.W)
        self.diarize_interval_var = tk.StringVar(value="30")
        self.diarize_interval_entry = ttk.Entry(self.diarize_cfg, textvariable=self.diarize_interval_var, width=8)
        self.diarize_interval_entry.grid(row=0, column=4, sticky=tk.W, padx=(5, 10))

        ttk.Label(self.diarize_cfg, text="Max Speakers:").grid(row=0, column=5, sticky=tk.W)
        self.diarize_max_speakers_var = tk.StringVar(value="0")
        self.diarize_max_speakers_entry = ttk.Entry(self.diarize_cfg, textvariable=self.diarize_max_speakers_var, width=5)
        self.diarize_max_speakers_entry.grid(row=0, column=6, sticky=tk.W, padx=(5, 0))

        ttk.Label(self.diarize_cfg, text="Device:").grid(row=0, column=7, sticky=tk.W, padx=(10, 0))
        self.diarize_device_var = tk.StringVar(value="cpu")
        self.diarize_device_combo = ttk.Combobox(
            self.diarize_cfg,
            textvariable=self.diarize_device_var,
            values=["cpu", "cuda"],
            state="readonly",
            width=5,
        )
        self.diarize_device_combo.grid(row=0, column=8, sticky=tk.W, padx=(5, 0))

        # Buttons
        self.button_frame = ttk.Frame(self.root, padding="10")
        self.button_frame.pack(fill=tk.X)
        self.start_btn = ttk.Button(self.button_frame, text="Start Recording", command=self.start_recording)
        self.start_btn.pack(side=tk.LEFT, padx=(0,5))
        self.stop_btn = ttk.Button(self.button_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0,5))
        self.clear_btn = ttk.Button(self.button_frame, text="Clear Text", command=self.clear_text)
        self.clear_btn.pack(side=tk.LEFT)
        self.dark_mode_var = tk.BooleanVar(value=False)
        self.dark_mode_check = ttk.Checkbutton(
            self.button_frame,
            text="Dark Mode",
            variable=self.dark_mode_var,
            command=self.on_theme_toggled,
        )
        self.dark_mode_check.pack(side=tk.RIGHT)
        
        # Status
        self.status_var = tk.StringVar(value="Ready. Click Refresh to list devices.")
        self.status_label = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding="5")
        self.status_label.pack(fill=tk.X, padx=10, pady=(0,5))
        
        # Output row: transcription and key points side-by-side
        self.output_row = ttk.Frame(self.root, padding="10")
        self.output_row.pack(fill=tk.BOTH, expand=True)
        self.output_row.columnconfigure(0, weight=3)
        self.output_row.columnconfigure(1, weight=2)
        self.output_row.rowconfigure(0, weight=1)

        self.text_frame = ttk.LabelFrame(self.output_row, text="Live Transcription", padding="10")
        self.text_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 5), pady=(0, 5))
        self.text_area = scrolledtext.ScrolledText(self.text_frame, wrap=tk.WORD, font=("Arial", 11))
        self.text_area.pack(fill=tk.BOTH, expand=True)

        # Key points output area
        self.points_frame = ttk.LabelFrame(self.output_row, text="Brief Key Points", padding="10")
        self.points_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(5, 0), pady=(0, 5))
        self.key_points_area = scrolledtext.ScrolledText(self.points_frame, wrap=tk.WORD, font=("Arial", 10), height=8)
        self.key_points_area.pack(fill=tk.BOTH, expand=True)

        # Speaker-labeled transcription area (below output row)
        self.speaker_frame = ttk.LabelFrame(self.root, text="Speaker-labeled Transcription", padding="10")
        self.speaker_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        self.speaker_text_area = scrolledtext.ScrolledText(
            self.speaker_frame, wrap=tk.WORD, font=("Consolas", 10), height=6
        )
        self.speaker_text_area.pack(fill=tk.BOTH, expand=True)
        # Define tags for speaker colors
        self._speaker_colors = [
            "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        ]
        for i, color in enumerate(self._speaker_colors):
            self.speaker_text_area.tag_configure(f"speaker_{i}", foreground=color)
        self.speaker_text_area.tag_configure("speaker_default", foreground="#555555")
        # Start hidden by default (diarization is opt-in)
        self.speaker_frame.pack_forget()

    def on_theme_toggled(self):
        self.apply_theme()
        self.schedule_config_save()

    def apply_theme(self):
        dark_mode = self.dark_mode_var.get()
        self.style.theme_use("clam")

        if dark_mode:
            palette = {
                "bg": "#17191d",
                "panel": "#20242b",
                "panel_alt": "#262b33",
                "text": "#e7ebf0",
                "muted": "#aab3bf",
                "accent": "#5aa2ff",
                "border": "#343b45",
                "insert": "#ffffff",
                "select": "#355a8a",
            }
        else:
            palette = {
                "bg": "#f3f5f8",
                "panel": "#ffffff",
                "panel_alt": "#eef2f7",
                "text": "#1b2430",
                "muted": "#5b6675",
                "accent": "#0a66c2",
                "border": "#c7d0db",
                "insert": "#1b2430",
                "select": "#cfe4ff",
            }

        self.root.configure(bg=palette["bg"])
        self.style.configure(".", background=palette["bg"], foreground=palette["text"])
        self.style.configure("TFrame", background=palette["bg"])
        self.style.configure("TLabel", background=palette["bg"], foreground=palette["text"])
        self.style.configure("TLabelframe", background=palette["bg"], bordercolor=palette["border"])
        self.style.configure("TLabelframe.Label", background=palette["bg"], foreground=palette["text"])
        self.style.configure(
            "TButton",
            background=palette["panel_alt"],
            foreground=palette["text"],
            bordercolor=palette["border"],
            lightcolor=palette["panel_alt"],
            darkcolor=palette["panel_alt"],
        )
        self.style.map(
            "TButton",
            background=[("active", palette["accent"]), ("disabled", palette["panel_alt"])],
            foreground=[("active", "#ffffff"), ("disabled", palette["muted"])],
        )
        self.style.configure("TCheckbutton", background=palette["bg"], foreground=palette["text"])
        self.style.map("TCheckbutton", foreground=[("disabled", palette["muted"])])
        self.style.configure(
            "TEntry",
            fieldbackground=palette["panel"],
            foreground=palette["text"],
            insertcolor=palette["insert"],
            bordercolor=palette["border"],
        )
        self.style.configure(
            "TCombobox",
            fieldbackground=palette["panel"],
            background=palette["panel"],
            foreground=palette["text"],
            arrowcolor=palette["text"],
            bordercolor=palette["border"],
        )
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", palette["panel"])],
            selectbackground=[("readonly", palette["panel"])],
            selectforeground=[("readonly", palette["text"])],
        )

        self.status_label.configure(background=palette["panel_alt"], foreground=palette["text"])

        text_widgets = [self.text_area, self.key_points_area, self.speaker_text_area]
        for widget in text_widgets:
            widget.configure(
                background=palette["panel"],
                foreground=palette["text"],
                insertbackground=palette["insert"],
                selectbackground=palette["select"],
                selectforeground=palette["text"],
                highlightbackground=palette["border"],
                highlightcolor=palette["accent"],
                relief=tk.FLAT,
            )

        for frame in (self.text_frame, self.points_frame, self.extract_cfg, self.diarize_cfg, self.speaker_frame):
            frame.configure(style="TLabelframe")
        
    def refresh_devices(self):
        try:
            p = pyaudio.PyAudio()
            self.available_devices = []
            device_list = []
            for device in p.get_loopback_device_info_generator():
                name = device["name"]
                idx = device["index"]
                host_api = p.get_host_api_info_by_index(device["hostApi"])
                host_name = host_api["name"]
                self.available_devices.append((idx, name, host_name))
                device_list.append(f"[{idx}] {name} ({host_name})")
            p.terminate()
            if device_list:
                self.device_combo["values"] = device_list
                self.device_var.set(device_list[0])
                self.status_var.set(f"Found {len(device_list)} WASAPI loopback device(s).")
            else:
                self.status_var.set("No WASAPI loopback devices found!")
                messagebox.showwarning("No Devices", "No WASAPI loopback devices found.")
        except Exception as e:
            self.status_var.set(f"Error detecting devices: {e}")
            messagebox.showerror("Error", f"Failed to detect audio devices:\n{e}")
    
    def on_device_selected(self, event=None):
        selection = self.device_combo.current()
        if selection >= 0:
            idx, name, host = self.available_devices[selection]
            self.status_var.set(f"Selected: [{idx}] {name}")
    
    def start_recording(self):
        if self.is_recording:
            return
        selection = self.device_combo.current()
        if selection < 0:
            messagebox.showwarning("No Device", "Please select an audio device first.")
            return

        self.save_config()
        
        device_index = self.available_devices[selection][0]
        save_audio = self.save_audio_var.get()
        
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.device_combo.config(state=tk.DISABLED)
        
        self.recorder = AudioRecorder(device_index, sample_rate=16000, chunk_duration=2.0, save_audio=save_audio)
        self.recorder.start(records_folder=self.records_folder)

        with self.caption_lock:
            self.full_transcription_text = ""
            self.caption_total_chars = 0
        self.key_points_area.delete(1.0, tk.END)
        self.speaker_text_area.delete(1.0, tk.END)
        
        # Create segment store and audio buffer for diarization
        self.segment_store = SegmentStore()
        self.audio_buffer = AudioBuffer(max_duration=120.0, sample_rate=16000)
        
        self.transcriber = Transcriber(
            self.recorder,
            model_size=self.model_var.get(),
            language=self.lang_var.get(),
            records_folder=self.records_folder,
            whisper_device=self.whisper_device_var.get(),
            local_files_only=self.whisper_local_only_var.get(),
            segment_store=self.segment_store,
            audio_buffer=self.audio_buffer,
        )
        self.transcriber.set_text_callback(self.append_text)
        self.transcriber.start()

        self.is_recording = True
        self.set_recording_layout(is_recording=True)
        self.ensure_key_point_extractor_state()
        self.ensure_diarizer_state()

        if save_audio:
            base = self.recorder.get_base_filename()
            self.file_label_var.set(f"Saving to: {self.records_folder}\\{base}.[wav,txt]")
        else:
            self.file_label_var.set("Audio/Text saving disabled")
        self.status_var.set("Recording and transcribing...")
        
    def stop_recording(self):
        if not self.is_recording:
            return
        self.status_var.set("Stopping...")
        # Cancel pending flush and drain buffer so no text is lost
        with self._pending_lock:
            if self._flush_after_id is not None:
                try:
                    self.root.after_cancel(self._flush_after_id)
                except Exception:
                    pass
                self._flush_after_id = None
            final_buffer = self._pending_text_buffer
            self._pending_text_buffer = []
        if final_buffer:
            self._append_text_impl("".join(final_buffer))
        
        if self.transcriber:
            self.transcriber.stop()
            self.transcriber = None
        if self.key_point_extractor:
            self.key_point_extractor.stop()
            self.key_point_extractor = None
        if self.speaker_diarizer:
            self.speaker_diarizer.stop()
            self.speaker_diarizer = None
        if self.recorder:
            wav_path = self.recorder.get_wav_path()
            self.recorder.stop()
            self.recorder = None
            if wav_path:
                self.file_label_var.set(f"Saved: {wav_path} (and .txt)")
        # Write final speaker-labeled transcripts to the text file
        if self.segment_store is not None:
            try:
                labeled_segments = self.segment_store.get_all()
                base = self.recorder.get_base_filename() if self.recorder else None
                if labeled_segments and base:
                    has_labels = any(s.speaker for s in labeled_segments)
                    labeled_path = os.path.join(self.records_folder, base + "_labeled.txt")
                    with open(labeled_path, "w", encoding="utf-8") as f:
                        f.write(f"Speaker-labeled Transcription - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("=" * 50 + "\n\n")
                        for seg in labeled_segments:
                            if has_labels and seg.speaker:
                                f.write(f"[{seg.speaker}] {seg.text.strip()}\n")
                            else:
                                f.write(f"{seg.text.strip()}\n")
                    if has_labels:
                        self.file_label_var.set(f"Saved: {base}.[wav,txt] + {base}_labeled.txt")
            except Exception as e:
                print(f"Error writing labeled transcript: {e}")
        
        if self.segment_store is not None:
            self.segment_store.clear()
            self.segment_store = None
        if self.audio_buffer is not None:
            self.audio_buffer.clear()
            self.audio_buffer = None
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.device_combo.config(state="readonly")
        self.is_recording = False
        self.set_recording_layout(is_recording=False)
        self.status_var.set("Stopped. Files saved in 'records' folder.")

    def set_recording_layout(self, is_recording):
        parameter_frames = [
            self.control_frame,
            self.whisper_opt_frame,
            self.save_frame,
            self.extract_cfg,
            self.diarize_cfg,
        ]

        if is_recording:
            for frame in parameter_frames:
                frame.pack_forget()
            return

        self.control_frame.pack(fill=tk.X, before=self.button_frame)
        self.whisper_opt_frame.pack(fill=tk.X, padx=10, before=self.button_frame)
        self.save_frame.pack(fill=tk.X, padx=10, before=self.button_frame)
        self.extract_cfg.pack(fill=tk.X, padx=10, pady=(0, 5), before=self.button_frame)
        self.diarize_cfg.pack(fill=tk.X, padx=10, pady=(0, 5), before=self.button_frame)
    
    def append_text(self, text):
        with self.caption_lock:
            self.full_transcription_text += text
            self.caption_total_chars += len(text)
            if len(self.full_transcription_text) > self.MAX_CAPTION_HISTORY_CHARS:
                self.full_transcription_text = self.full_transcription_text[-self.MAX_CAPTION_HISTORY_CHARS:]
        
        with self._pending_lock:
            self._pending_text_buffer.append(text)
            if self._flush_after_id is None:
                self._flush_after_id = self.root.after(self.FLUSH_INTERVAL_MS, self._flush_pending_text)
    
    def _flush_pending_text(self):
        with self._pending_lock:
            self._flush_after_id = None
            buffer = self._pending_text_buffer
            self._pending_text_buffer = []
        
        if buffer:
            self._append_text_impl("".join(buffer))
    
    def _trim_widget(self, widget, max_chars):
        """Remove oldest content from a text widget when it exceeds max_chars."""
        content = widget.get(1.0, tk.END)
        if len(content) > max_chars:
            trim_chars = len(content) - max_chars
            end_trim = widget.index(f"1.0 + {trim_chars} chars")
            widget.delete(1.0, end_trim)
    
    def _append_text_impl(self, text):
        self.text_area.insert(tk.END, text)
        self.text_area.see(tk.END)
        self._trim_widget(self.text_area, self.MAX_TEXT_AREA_CHARS)

    def append_key_points(self, text):
        self.root.after(0, self._append_key_points_impl, text)

    def _append_key_points_impl(self, text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.key_points_area.insert(tk.END, f"[{timestamp}]\n{text.strip()}\n\n")
        self.key_points_area.see(tk.END)
        self._trim_widget(self.key_points_area, self.MAX_KEY_POINTS_CHARS)

    def get_recent_captions(self):
        with self.caption_lock:
            joined = self.full_transcription_text
            total_chars = self.caption_total_chars
        return joined, total_chars

    def get_extract_interval(self):
        try:
            return float(self.extract_interval_var.get())
        except (TypeError, ValueError):
            return 20.0

    def setup_config_autosave(self):
        """Auto-save config when extractor-related UI values change."""
        watched_vars = [
            self.whisper_device_var,
            self.whisper_local_only_var,
            self.dark_mode_var,
            self.extract_enabled_var,
            self.ollama_url_var,
            self.ollama_model_var,
            self.extract_prompt_var,
            self.extract_interval_var,
            self.extract_timeout_var,
            self.diarize_enabled_var,
            self.hf_token_var,
            self.diarize_interval_var,
            self.diarize_max_speakers_var,
            self.diarize_device_var,
        ]
        for var in watched_vars:
            var.trace_add("write", self.on_config_ui_changed)

    def on_config_ui_changed(self, *args):
        if self._suspend_config_autosave:
            return
        self.apply_extractor_ui_state()
        self.apply_diarize_ui_state()
        self.schedule_config_save()

    def schedule_config_save(self):
        if self._suspend_config_autosave:
            return
        if self._config_save_after_id is not None:
            self.root.after_cancel(self._config_save_after_id)
        self._config_save_after_id = self.root.after(400, self.flush_config_save)

    def flush_config_save(self):
        self._config_save_after_id = None
        self.save_config()

    def on_extractor_enable_toggled(self):
        self.apply_extractor_ui_state()
        self.ensure_key_point_extractor_state()
        self.schedule_config_save()

    def apply_extractor_ui_state(self):
        enabled = self.extract_enabled_var.get()
        state = "normal" if enabled else "disabled"
        self.ollama_url_entry.config(state=state)
        self.ollama_model_entry.config(state=state)
        self.extract_interval_entry.config(state=state)
        self.extract_timeout_entry.config(state=state)
        self.extract_prompt_entry.config(state=state)

        if enabled:
            self.text_frame.grid_configure(row=0, column=0, columnspan=1, padx=(0, 5), sticky=tk.NSEW)
            self.points_frame.grid()
            self.output_row.columnconfigure(0, weight=3)
            self.output_row.columnconfigure(1, weight=2)
        else:
            self.points_frame.grid_remove()
            self.text_frame.grid_configure(row=0, column=0, columnspan=2, padx=(0, 0), sticky=tk.NSEW)
            self.output_row.columnconfigure(0, weight=1)
            self.output_row.columnconfigure(1, weight=0)

    def ensure_key_point_extractor_state(self):
        if not self.is_recording:
            return

        if self.extract_enabled_var.get():
            if self.key_point_extractor is None:
                self.key_point_extractor = KeyPointExtractor(
                    interval_getter=self.get_extract_interval,
                    text_provider=self.get_recent_captions,
                    extract_callback=self.extract_brief_key_points,
                    output_callback=self.append_key_points,
                )
                self.key_point_extractor.start()
        elif self.key_point_extractor is not None:
            self.key_point_extractor.stop()
            self.key_point_extractor = None

    def on_diarize_enable_toggled(self):
        self.apply_diarize_ui_state()
        self.ensure_diarizer_state()
        self.schedule_config_save()

    def apply_diarize_ui_state(self):
        enabled = self.diarize_enabled_var.get()
        state = "normal" if enabled else "disabled"
        self.hf_token_entry.config(state=state)
        self.diarize_interval_entry.config(state=state)
        self.diarize_max_speakers_entry.config(state=state)
        self.diarize_device_combo.config(state=state)

        if enabled:
            try:
                self.speaker_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
            except tk.TclError:
                pass
        else:
            self.speaker_frame.pack_forget()

    def get_diarization_interval(self):
        try:
            return max(10.0, float(self.diarize_interval_var.get()))
        except (TypeError, ValueError):
            return 30.0

    def get_diarization_max_speakers(self):
        try:
            val = int(self.diarize_max_speakers_var.get())
            return val if val > 0 else None
        except (TypeError, ValueError):
            return None

    def ensure_diarizer_state(self):
        if not self.is_recording:
            return

        if not self.diarize_enabled_var.get():
            if self.speaker_diarizer is not None:
                self.speaker_diarizer.stop()
                self.speaker_diarizer = None
            return

        if self.speaker_diarizer is not None:
            return  # already running

        hf_token = self.hf_token_var.get().strip()
        if not hf_token:
            self.append_diarization_results("[Diarization] HF Token not configured. Please enter your HuggingFace token.")
            return
        if not HAS_PYANNOTE:
            self.append_diarization_results("[Diarization] pyannote.audio not installed. Run: pip install pyannote.audio torch")
            return

        self.speaker_diarizer = SpeakerDiarizer(
            hf_token=hf_token,
            audio_buffer=self.audio_buffer,
            segment_store=self.segment_store,
            diarization_callback=self.append_diarization_results,
            interval=self.get_diarization_interval(),
            device=self.diarize_device_var.get(),
            max_speakers=self.get_diarization_max_speakers(),
        )
        self.speaker_diarizer.start()

    def append_diarization_results(self, results):
        """Called from SpeakerDiarizer thread with diarization updates."""
        if isinstance(results, str):
            # Status/error message
            self.root.after(0, self._append_diarization_message, results)
        elif isinstance(results, tuple):
            # (labeled_segments, label_updates)
            labeled, updates = results
            self.root.after(0, self._rebuild_speaker_text_area, labeled, updates)

    def _append_diarization_message(self, message):
        self.status_var.set(message)

    def _rebuild_speaker_text_area(self, labeled_segments, updates):
        """Rebuild the speaker-labeled text area content with color-coded speaker tags."""
        # Scroll position preservation
        try:
            scroll_pos = self.speaker_text_area.yview()
        except Exception:
            scroll_pos = (0.0, 1.0)

        self.speaker_text_area.delete(1.0, tk.END)

        for seg in labeled_segments:
            if seg.speaker:
                speaker_num = int(seg.speaker.split("_")[-1]) if seg.speaker.startswith("SPEAKER_") else 0
                color_idx = speaker_num % len(self._speaker_colors)
                tag = f"speaker_{color_idx}"
                label = f"SPEAKER {speaker_num}"
            else:
                tag = "speaker_default"
                label = "UNKNOWN"

            timestamp = f"{seg.start:.1f}s"
            line = f"[{timestamp}] [{label}] {seg.text.strip()}\n"
            self.speaker_text_area.insert(tk.END, line, tag)

        self.speaker_text_area.see(tk.END)
        self._trim_widget(self.speaker_text_area, self.MAX_SPEAKER_CHARS)

    @staticmethod
    def _as_bool(value, default=False):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        if value is None:
            return default
        return bool(value)

    def load_config(self):
        """Load persisted UI config from file."""
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.dark_mode_var.set(self._as_bool(data.get("dark_mode"), self.dark_mode_var.get()))
            self.whisper_device_var.set(str(data.get("whisper_device", self.whisper_device_var.get())))
            self.whisper_local_only_var.set(self._as_bool(data.get("whisper_local_files_only"), self.whisper_local_only_var.get()))
            self.extract_enabled_var.set(self._as_bool(data.get("extract_enabled"), self.extract_enabled_var.get()))
            self.ollama_url_var.set(str(data.get("ollama_url", self.ollama_url_var.get())))
            self.ollama_model_var.set(str(data.get("ollama_model", self.ollama_model_var.get())))
            self.extract_prompt_var.set(str(data.get("extract_prompt", self.extract_prompt_var.get())))
            self.extract_interval_var.set(str(data.get("extract_interval", self.extract_interval_var.get())))
            self.extract_timeout_var.set(str(data.get("extract_timeout", self.extract_timeout_var.get())))
            self.diarize_enabled_var.set(self._as_bool(data.get("diarization_enabled"), self.diarize_enabled_var.get()))
            self.hf_token_var.set(str(data.get("hf_token", self.hf_token_var.get())))
            self.diarize_interval_var.set(str(data.get("diarization_interval", self.diarize_interval_var.get())))
            self.diarize_max_speakers_var.set(str(data.get("max_speakers", self.diarize_max_speakers_var.get())))
            self.diarize_device_var.set(str(data.get("diarization_device", self.diarize_device_var.get())))
        except Exception as e:
            self.status_var.set(f"Config load failed: {e}")

    def save_config(self):
        """Save current UI config to file."""
        data = {
            "dark_mode": self.dark_mode_var.get(),
            "whisper_device": self.whisper_device_var.get().strip(),
            "whisper_local_files_only": self.whisper_local_only_var.get(),
            "extract_enabled": self.extract_enabled_var.get(),
            "ollama_url": self.ollama_url_var.get().strip(),
            "ollama_model": self.ollama_model_var.get().strip(),
            "extract_prompt": self.extract_prompt_var.get().strip(),
            "extract_interval": self.extract_interval_var.get().strip(),
            "extract_timeout": self.extract_timeout_var.get().strip(),
            "diarization_enabled": self.diarize_enabled_var.get(),
            "hf_token": self.hf_token_var.get().strip(),
            "diarization_interval": self.diarize_interval_var.get().strip(),
            "max_speakers": self.diarize_max_speakers_var.get().strip(),
            "diarization_device": self.diarize_device_var.get().strip(),
        }
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.status_var.set(f"Config save failed: {e}")

    def get_extract_timeout(self):
        try:
            return max(10.0, float(self.extract_timeout_var.get()))
        except (TypeError, ValueError):
            return 90.0

    def extract_brief_key_points(self, captions_text):
        """Call Ollama to extract brief key points from recent live captions."""
        prompt = self.extract_prompt_var.get().strip() or "Extract brief key points from the captions."
        model = self.ollama_model_var.get().strip()
        base_url = self.ollama_url_var.get().strip().rstrip("/")
        if not model or not base_url:
            return "[Key point extraction skipped] Configure Ollama URL and model."

        timeout_seconds = self.get_extract_timeout()
        prefers_chat_api = model.lower().startswith("qwen")

        def clean_ollama_text(text):
            if not isinstance(text, str):
                return ""
            cleaned = text.strip()
            if "</think>" in cleaned:
                cleaned = cleaned.split("</think>", 1)[1].strip()
            return cleaned

        def parse_ollama_text(data):
            """Handle response shape differences across Ollama endpoints/models.

            Ollama's numeric `context` field is token history, not decodable output text.
            """
            # Flexible extractor that walks common response shapes used by Ollama
            def extract(item):
                if not item:
                    return ""
                # strings are direct
                if isinstance(item, str):
                    return item
                # lists: join extracted parts
                if isinstance(item, list):
                    parts = [extract(x) for x in item]
                    return "\n".join([p for p in parts if p])
                # dicts: look for common fields
                if isinstance(item, dict):
                    # common direct text fields
                    for key in ("response", "text", "output_text", "output", "body"):
                        if key in item and item[key]:
                            return extract(item[key])
                    # nested message/content style
                    if "message" in item:
                        return extract(item["message"])
                    if "content" in item:
                        return extract(item["content"])
                    # OpenAI-like choices
                    if "choices" in item and isinstance(item["choices"], list):
                        return extract(item["choices"]) 
                    # Ollama may return `outputs` or `results`
                    if "outputs" in item and isinstance(item["outputs"], list):
                        return extract(item["outputs"]) 
                    if "results" in item and isinstance(item["results"], list):
                        return extract(item["results"]) 
                # Fallback: nothing extractable
                return ""

            # Try a set of likely top-level candidates first, then fall back to scanning values
            candidates = []
            if isinstance(data, str):
                candidates.append(data)
            if isinstance(data, dict):
                for k in ("response", "text", "output_text", "output", "message", "content", "results", "choices", "outputs"):
                    if k in data:
                        candidates.append(data[k])
                # also include the whole dict as last resort
                candidates.append(data)

            for item in candidates:
                out = extract(item)
                cleaned = clean_ollama_text(out)
                if cleaned:
                    return cleaned

            # As a final fallback, scan all string values in the dict
            if isinstance(data, dict):
                for v in data.values():
                    out = extract(v)
                    cleaned = clean_ollama_text(out)
                    if cleaned:
                        return cleaned

            return ""

        def build_generate_payload(captions_excerpt, disable_think=False):
            full_prompt = f"{prompt}\n\nLive captions:\n{captions_excerpt}\n"
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                # Keep generations concise to improve response time.
                "options": {"num_predict": 200},
            }
            if disable_think:
                payload["think"] = False
            return payload

        def build_chat_payload(captions_excerpt):
            return {
                "model": model,
                "stream": False,
                "think": False,
                "options": {"num_predict": 200},
                "messages": [
                    {
                        "role": "system",
                        "content": "Extract concise meeting key points. Do not include chain-of-thought.",
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nLive captions:\n{captions_excerpt}\n",
                    },
                ],
            }

        def request_once(endpoint, payload):
            req = urllib.request.Request(
                url=f"{base_url}{endpoint}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
                data = json.loads(body)
            return parse_ollama_text(data)

        def request_with_fallbacks(captions_excerpt, disable_think_first=False):
            request_plan = []
            if prefers_chat_api:
                request_plan.extend(
                    [
                        ("/api/chat", build_chat_payload(captions_excerpt)),
                        ("/api/generate", build_generate_payload(captions_excerpt, disable_think=True)),
                    ]
                )
            else:
                request_plan.extend(
                    [
                        ("/api/generate", build_generate_payload(captions_excerpt, disable_think=disable_think_first)),
                        ("/api/generate", build_generate_payload(captions_excerpt, disable_think=True)),
                        ("/api/chat", build_chat_payload(captions_excerpt)),
                    ]
                )

            seen = set()
            for endpoint, payload in request_plan:
                payload_key = json.dumps(payload, sort_keys=True)
                request_key = (endpoint, payload_key)
                if request_key in seen:
                    continue
                seen.add(request_key)

                result = request_once(endpoint, payload)
                if result:
                    return result
            return ""

        try:
            result = request_with_fallbacks(captions_text)
            if result:
                return result

            return "[Key point extraction skipped] Ollama returned no displayable text. The `context` array is token history, not the answer. Try `think=false`, `/api/chat`, or a different model."
        except socket.timeout:
            # Retry once with a smaller context so slow models can still return.
            fallback_text = captions_text[-8000:]
            if fallback_text and fallback_text != captions_text:
                try:
                    result = request_with_fallbacks(fallback_text, disable_think_first=True)
                    if result:
                        return result
                except Exception:
                    pass
            return f"[Key point extraction skipped] Request timed out after {int(timeout_seconds)}s. Increase Timeout(s) or use a faster model."
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", "")
            if isinstance(reason, socket.timeout):
                return f"[Key point extraction skipped] Request timed out after {int(timeout_seconds)}s. Increase Timeout(s) or use a faster model."
            return f"[Key point extraction skipped] Ollama request failed: {e}"
        except Exception as e:
            return f"[Key point extraction skipped] Unexpected error: {e}"
    
    def clear_text(self):
        self.text_area.delete(1.0, tk.END)
        self.key_points_area.delete(1.0, tk.END)
        self.speaker_text_area.delete(1.0, tk.END)
        if self.segment_store is not None:
            self.segment_store.clear()
        with self.caption_lock:
            self.full_transcription_text = ""
            self.caption_total_chars = 0
        self.status_var.set("Text cleared.")
    
    def open_records_folder(self):
        """Open the records folder in file explorer."""
        folder = os.path.abspath(self.records_folder)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.run(["open", folder])
        else:
            subprocess.run(["xdg-open", folder])
    
    def on_closing(self):
        if self._config_save_after_id is not None:
            self.root.after_cancel(self._config_save_after_id)
            self._config_save_after_id = None
        with self._pending_lock:
            if self._flush_after_id is not None:
                try:
                    self.root.after_cancel(self._flush_after_id)
                except Exception:
                    pass
                self._flush_after_id = None
        self.save_config()
        self.stop_recording()
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()


# ------------------------------
# Entry Point
# ------------------------------
if __name__ == "__main__":
    app = MeetingRecorderApp()
    app.run()