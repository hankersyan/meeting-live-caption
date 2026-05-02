"""
Meeting Live Caption — FunASR Edition

Real-time speech recognition with speaker diarization,
supporting mic, speaker (WASAPI loopback), and file input.

Usage:
    python app.py
"""

import sys
import os
import json
import queue
import threading
import time
import urllib.request
import urllib.error
import socket
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np

# GUI
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

# Audio input & ASR engine
from audio_input import (
    AudioInputBase, MicInput, SpeakerInput, FileInput,
    list_mic_devices, list_speaker_devices, pyaudio, PYAUDIOWP_AVAILABLE,
)
from asr_engine import (
    FunASRTranscriber, TranscriptionResult, MODEL_CONFIG, FUNASR_AVAILABLE,
)


# ------------------------------
# Speaker Color Palette
# ------------------------------

SPEAKER_COLORS = [
    "#2196F3",  # Blue
    "#4CAF50",  # Green
    "#FF9800",  # Orange
    "#F44336",  # Red
    "#9C27B0",  # Purple
    "#00BCD4",  # Cyan
    "#795548",  # Brown
    "#607D8B",  # Blue Grey
]

SPEAKER_DARK_COLORS = [
    "#64B5F6",  # Light Blue
    "#81C784",  # Light Green
    "#FFB74D",  # Light Orange
    "#E57373",  # Light Red
    "#CE93D8",  # Light Purple
    "#4DD0E1",  # Light Cyan
    "#A1887F",  # Light Brown
    "#90A4AE",  # Light Blue Grey
]


# ------------------------------
# Key Point Extractor (from original main.py)
# ------------------------------

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
# Main Application
# ------------------------------

class MeetingCaptionApp:
    # Input mode constants
    INPUT_MIC = "mic"
    INPUT_SPEAKER = "speaker"
    INPUT_FILE = "file"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Meeting Live Caption — FunASR")
        self.root.geometry("920x700")
        self.root.minsize(800, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.records_folder = "records"
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_funasr.json")

        # State
        self.audio_input: Optional[AudioInputBase] = None
        self.transcriber: Optional[FunASRTranscriber] = None
        self.key_point_extractor: Optional[KeyPointExtractor] = None
        self.is_recording = False
        self.mic_devices: List = []
        self.speaker_devices: List = []

        # Caption history for key-point extraction
        self.caption_lock = threading.Lock()
        self.full_transcription_text = ""
        self.caption_total_chars = 0

        # Buffered text updates
        self.MAX_CAPTION_HISTORY_CHARS = 200000
        self.MAX_TEXT_AREA_CHARS = 8000
        self.MAX_KEY_POINTS_CHARS = 3000
        self.FLUSH_INTERVAL_MS = 150

        self._pending_lock = threading.Lock()
        self._pending_text_buffer: list = []
        self._flush_after_id = None

        # Config autosave
        self._suspend_config_autosave = True
        self._config_save_after_id = None

        # Speaker tag tracking for tkinter text tags
        self._speaker_tags_created: Dict[str, bool] = {}

        # Style
        self.style = ttk.Style()

        # Build UI
        self._build_ui()
        self._load_config()
        self._setup_config_autosave()
        self._suspend_config_autosave = False
        self._apply_theme()
        self._apply_extractor_ui_state()
        self._on_input_mode_changed()
        self._refresh_devices()

    # ----------------------------------------------------------
    # UI Construction
    # ----------------------------------------------------------

    def _build_ui(self):
        # --- 1. Input Source & Device Selection ---
        self.input_frame = ttk.LabelFrame(self.root, text="Input Source", padding="8")
        self.input_frame.pack(fill=tk.X, padx=10, pady=(10, 2))

        # Input mode radio buttons
        self.input_mode_var = tk.StringVar(value=self.INPUT_SPEAKER)
        modes = [
            ("Microphone", self.INPUT_MIC),
            ("Speaker (WASAPI)", self.INPUT_SPEAKER),
            ("File", self.INPUT_FILE),
        ]
        for i, (label, mode) in enumerate(modes):
            rb = ttk.Radiobutton(
                self.input_frame, text=label, variable=self.input_mode_var,
                value=mode, command=self._on_input_mode_changed,
            )
            rb.grid(row=0, column=i, sticky=tk.W, padx=(0, 15))

        # Device dropdown
        ttk.Label(self.input_frame, text="Device:").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(
            self.input_frame, textvariable=self.device_var,
            state="readonly", width=55,
        )
        self.device_combo.grid(row=1, column=1, columnspan=2, sticky=tk.W+tk.E, padx=(5, 0), pady=(6, 0))
        self.device_combo.bind("<<ComboboxSelected>>", self._on_device_selected)

        ttk.Button(self.input_frame, text="Refresh", command=self._refresh_devices).grid(
            row=1, column=3, padx=(8, 0), pady=(6, 0))

        # File path (visible only in file mode)
        self.file_frame = ttk.Frame(self.input_frame)
        self.file_frame.grid(row=2, column=0, columnspan=4, sticky=tk.W+tk.E, pady=(6, 0))
        ttk.Label(self.file_frame, text="File:").pack(side=tk.LEFT)
        self.file_path_var = tk.StringVar()
        self.file_path_entry = ttk.Entry(self.file_frame, textvariable=self.file_path_var, width=60)
        self.file_path_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        ttk.Button(self.file_frame, text="Browse...", command=self._browse_file).pack(side=tk.LEFT)

        # Simulate realtime checkbox
        self.simulate_realtime_var = tk.BooleanVar(value=True)
        self.simulate_realtime_check = ttk.Checkbutton(
            self.file_frame, text="Simulate realtime", variable=self.simulate_realtime_var,
        )
        self.simulate_realtime_check.pack(side=tk.LEFT, padx=(10, 0))

        # --- 2. ASR Engine Configuration ---
        self.asr_frame = ttk.LabelFrame(self.root, text="ASR Engine (FunASR)", padding="8")
        self.asr_frame.pack(fill=tk.X, padx=10, pady=(2, 2))

        # Model selector
        ttk.Label(self.asr_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.model_var = tk.StringVar(value="iic/SenseVoiceSmall")
        model_names = list(MODEL_CONFIG.keys())
        self.model_combo = ttk.Combobox(
            self.asr_frame, textvariable=self.model_var,
            values=model_names, state="readonly", width=22,
        )
        self.model_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 15))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_changed)

        # Model description
        self.model_desc_var = tk.StringVar()
        self.model_desc_label = ttk.Label(
            self.asr_frame, textvariable=self.model_desc_var,
            foreground="#5b6675", font=("Microsoft YaHei", 9),
        )
        self.model_desc_label.grid(row=0, column=2, sticky=tk.W, padx=(0, 15))

        # Language selector
        ttk.Label(self.asr_frame, text="Language:").grid(row=0, column=3, sticky=tk.W, padx=(0, 5))
        self.lang_var = tk.StringVar(value="en")
        self.lang_combo = ttk.Combobox(
            self.asr_frame, textvariable=self.lang_var,
            state="readonly", width=8,
        )
        self.lang_combo.grid(row=0, column=4, sticky=tk.W, padx=(0, 15))

        # Speaker diarization info (always enabled)
        ttk.Label(
            self.asr_frame, text="Speaker Diarization: cam++ (always on)",
            foreground="#4CAF50", font=("Microsoft YaHei", 9, "bold"),
        ).grid(row=0, column=5, sticky=tk.W, padx=(0, 15))

        # Processing mode info (updated dynamically)
        self.mode_info_var = tk.StringVar(value="Two-pass: VAD+ASR+cam++")
        self.mode_info_label = ttk.Label(
            self.asr_frame, textvariable=self.mode_info_var,
            foreground="#5b6675", font=("Microsoft YaHei", 8),
        )
        self.mode_info_label.grid(row=1, column=0, columnspan=6, sticky=tk.W, pady=(4, 0))

        # Device (CPU/CUDA)
        ttk.Label(self.asr_frame, text="Device:").grid(row=0, column=6, sticky=tk.W, padx=(0, 5))
        self.device_asr_var = tk.StringVar(value="cpu")
        self.device_asr_combo = ttk.Combobox(
            self.asr_frame, textvariable=self.device_asr_var,
            values=["cpu", "cuda"], state="readonly", width=6,
        )
        self.device_asr_combo.grid(row=0, column=7, sticky=tk.W)

        # --- 3. Save Options ---
        self.save_frame = ttk.Frame(self.root, padding="5 5 5 5")
        self.save_frame.pack(fill=tk.X, padx=10, pady=(2, 2))
        self.save_audio_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.save_frame, text="Save audio & text to 'records' folder",
            variable=self.save_audio_var,
        ).pack(side=tk.LEFT)
        self.file_label_var = tk.StringVar(value="No recording yet")
        ttk.Label(self.save_frame, textvariable=self.file_label_var, foreground="#0a66c2").pack(
            side=tk.LEFT, padx=(20, 0))
        ttk.Button(self.save_frame, text="Open Records Folder", command=self._open_records_folder).pack(
            side=tk.RIGHT)

        # --- 4. Key Point Extraction (Ollama) ---
        self.extract_cfg = ttk.LabelFrame(self.root, text="Key Point Extraction (Ollama)", padding="8")
        self.extract_cfg.pack(fill=tk.X, padx=10, pady=(2, 2))

        self.extract_enabled_var = tk.BooleanVar(value=True)
        self.extract_enable_check = ttk.Checkbutton(
            self.extract_cfg, text="Enable", variable=self.extract_enabled_var,
            command=self._on_extractor_enable_toggled,
        )
        self.extract_enable_check.grid(row=0, column=0, sticky=tk.W, padx=(0, 8))

        ttk.Label(self.extract_cfg, text="URL:").grid(row=0, column=1, sticky=tk.W)
        self.ollama_url_var = tk.StringVar(value="http://localhost:11434")
        self.ollama_url_entry = ttk.Entry(self.extract_cfg, textvariable=self.ollama_url_var, width=26)
        self.ollama_url_entry.grid(row=0, column=2, sticky=tk.W, padx=(5, 10))

        ttk.Label(self.extract_cfg, text="Model:").grid(row=0, column=3, sticky=tk.W)
        self.ollama_model_var = tk.StringVar(value="LiquidAI/lfm2.5-1.2b-instruct")
        self.ollama_model_entry = ttk.Entry(self.extract_cfg, textvariable=self.ollama_model_var, width=16)
        self.ollama_model_entry.grid(row=0, column=4, sticky=tk.W, padx=(5, 10))

        ttk.Label(self.extract_cfg, text="Refresh(s):").grid(row=0, column=5, sticky=tk.W)
        self.extract_interval_var = tk.StringVar(value="20")
        self.extract_interval_entry = ttk.Entry(self.extract_cfg, textvariable=self.extract_interval_var, width=6)
        self.extract_interval_entry.grid(row=0, column=6, sticky=tk.W, padx=(5, 0))

        ttk.Label(self.extract_cfg, text="Timeout(s):").grid(row=0, column=7, sticky=tk.W, padx=(10, 0))
        self.extract_timeout_var = tk.StringVar(value="90")
        self.extract_timeout_entry = ttk.Entry(self.extract_cfg, textvariable=self.extract_timeout_var, width=6)
        self.extract_timeout_entry.grid(row=0, column=8, sticky=tk.W, padx=(5, 0))

        ttk.Label(self.extract_cfg, text="Prompt:").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        self.extract_prompt_var = tk.StringVar(
            value="Extract 3-5 brief key points from the live meeting captions. Use concise bullet points and avoid repetition."
        )
        self.extract_prompt_entry = ttk.Entry(self.extract_cfg, textvariable=self.extract_prompt_var, width=120)
        self.extract_prompt_entry.grid(row=1, column=1, columnspan=6, sticky=tk.W+tk.E, padx=(5, 0), pady=(6, 0))
        self.extract_cfg.columnconfigure(2, weight=1)
        self.extract_cfg.columnconfigure(4, weight=1)

        # --- 5. Buttons ---
        self.button_frame = ttk.Frame(self.root, padding="10 5 10 5")
        self.button_frame.pack(fill=tk.X)
        self.start_btn = ttk.Button(self.button_frame, text="▶ Start Recording", command=self._start_recording)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.stop_btn = ttk.Button(self.button_frame, text="■ Stop Recording", command=self._stop_recording, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.clear_btn = ttk.Button(self.button_frame, text="Clear Text", command=self._clear_text)
        self.clear_btn.pack(side=tk.LEFT)
        self.dark_mode_var = tk.BooleanVar(value=False)
        self.dark_mode_check = ttk.Checkbutton(
            self.button_frame, text="Dark Mode", variable=self.dark_mode_var,
            command=self._on_theme_toggled,
        )
        self.dark_mode_check.pack(side=tk.RIGHT)

        # --- 6. Status Bar ---
        self.status_var = tk.StringVar(value="Ready. Click Refresh to list devices.")
        self.status_label = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding="5",
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=(2, 5))

        # --- 7. Output Area ---
        self.output_row = ttk.Frame(self.root, padding="5 5 10 10")
        self.output_row.pack(fill=tk.BOTH, expand=True)
        self.output_row.columnconfigure(0, weight=3)
        self.output_row.columnconfigure(1, weight=2)
        self.output_row.rowconfigure(0, weight=1)

        self.text_frame = ttk.LabelFrame(self.output_row, text="Live Transcription", padding="8")
        self.text_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 5), pady=(0, 5))
        self.text_area = scrolledtext.ScrolledText(
            self.text_frame, wrap=tk.WORD,
            font=("Microsoft YaHei", 11), spacing1=2, spacing3=2,
        )
        self.text_area.pack(fill=tk.BOTH, expand=True)

        # Key points output
        self.points_frame = ttk.LabelFrame(self.output_row, text="Brief Key Points", padding="8")
        self.points_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(5, 0), pady=(0, 5))
        self.key_points_area = scrolledtext.ScrolledText(
            self.points_frame, wrap=tk.WORD, font=("Microsoft YaHei", 10), height=8,
        )
        self.key_points_area.pack(fill=tk.BOTH, expand=True)

    # ----------------------------------------------------------
    # Input Mode & Device Management
    # ----------------------------------------------------------

    def _on_input_mode_changed(self, event=None):
        mode = self.input_mode_var.get()
        show_device = mode in (self.INPUT_MIC, self.INPUT_SPEAKER)
        show_file = mode == self.INPUT_FILE

        # Toggle device combo visibility
        if show_device:
            self.device_combo.config(state="readonly")
            self._refresh_devices()
        else:
            self.device_combo.config(state="disabled")

        # Toggle file frame
        if show_file:
            self.file_frame.grid()
        else:
            self.file_frame.grid_remove()

        # Toggle simulate-realtime checkbox
        if show_file:
            self.simulate_realtime_check.pack(side=tk.LEFT, padx=(10, 0))
        else:
            self.simulate_realtime_check.pack_forget()

    def _refresh_devices(self):
        mode = self.input_mode_var.get()
        if mode == self.INPUT_MIC:
            devices = list_mic_devices()
            self.mic_devices = devices
        elif mode == self.INPUT_SPEAKER:
            devices = list_speaker_devices()
            self.speaker_devices = devices
        else:
            return

        device_list = [f"[{idx}] {name} ({host})" for idx, name, host in devices]
        if device_list:
            self.device_combo["values"] = device_list
            self.device_var.set(device_list[0])
            self.status_var.set(f"Found {len(device_list)} device(s).")
        else:
            self.device_combo["values"] = []
            self.device_var.set("")
            if mode == self.INPUT_SPEAKER and not PYAUDIOWP_AVAILABLE:
                self.status_var.set("No WASAPI loopback support. Install pyaudiowpatch.")
            else:
                self.status_var.set("No devices found.")

    def _on_device_selected(self, event=None):
        selection = self.device_combo.current()
        mode = self.input_mode_var.get()
        devices = self.mic_devices if mode == self.INPUT_MIC else self.speaker_devices
        if selection >= 0 and selection < len(devices):
            idx, name, host = devices[selection]
            self.status_var.set(f"Selected: [{idx}] {name}")

    def _browse_file(self):
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.mp4 *.flac *.ogg *.m4a"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.file_path_var.set(path)

    # ----------------------------------------------------------
    # Model Config
    # ----------------------------------------------------------

    def _on_model_changed(self, event=None):
        model_name = self.model_var.get()
        cfg = MODEL_CONFIG.get(model_name, {})

        self.model_desc_var.set(cfg.get("description", ""))

        # Update language options
        languages = cfg.get("languages", ["auto"])
        self.lang_combo["values"] = languages
        if self.lang_var.get() not in languages:
            self.lang_var.set(languages[0])

        # Update processing mode info
        if cfg.get("timestamp_support", False):
            self.mode_info_var.set("Native pipeline: ASR+VAD+Punc+cam++ (timestamp-aware)")
        else:
            self.mode_info_var.set("Two-pass pipeline: VAD → ASR + cam++ embedding → cluster")

    # ----------------------------------------------------------
    # Recording
    # ----------------------------------------------------------

    def _start_recording(self):
        if self.is_recording:
            return

        if not FUNASR_AVAILABLE:
            messagebox.showerror("Error", "FunASR is not installed.\nRun: pip install funasr")
            return

        mode = self.input_mode_var.get()
        save_audio = self.save_audio_var.get()

        # Create audio input
        try:
            if mode == self.INPUT_MIC:
                selection = self.device_combo.current()
                if selection < 0 or selection >= len(self.mic_devices):
                    messagebox.showwarning("No Device", "Please select a microphone device.")
                    return
                device_index = self.mic_devices[selection][0]
                self.audio_input = MicInput(
                    device_index=device_index, chunk_duration=3.0, save_audio=save_audio,
                )
            elif mode == self.INPUT_SPEAKER:
                selection = self.device_combo.current()
                if selection < 0 or selection >= len(self.speaker_devices):
                    messagebox.showwarning("No Device", "Please select a speaker device.")
                    return
                device_index = self.speaker_devices[selection][0]
                self.audio_input = SpeakerInput(
                    device_index=device_index, chunk_duration=3.0, save_audio=save_audio,
                )
            elif mode == self.INPUT_FILE:
                file_path = self.file_path_var.get().strip()
                if not file_path or not os.path.isfile(file_path):
                    messagebox.showwarning("No File", "Please select a valid audio file.")
                    return
                self.audio_input = FileInput(
                    file_path=file_path, chunk_duration=5.0,
                    simulate_realtime=self.simulate_realtime_var.get(),
                    save_audio=save_audio,
                )
            else:
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create audio input: {e}")
            return

        # Save config before starting
        self._save_config()

        # UI state
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._set_controls_enabled(False)

        # Start audio input
        self.audio_input.start(records_folder=self.records_folder)

        # Reset caption state
        with self.caption_lock:
            self.full_transcription_text = ""
            self.caption_total_chars = 0
        self.key_points_area.delete(1.0, tk.END)
        self._speaker_tags_created.clear()

        # Create transcriber (diarization is always enabled)
        model_name = self.model_var.get()

        self.transcriber = FunASRTranscriber(
            audio_input=self.audio_input,
            model_name=model_name,
            language=self.lang_var.get(),
            device=self.device_asr_var.get(),
            records_folder=self.records_folder,
        )
        self.transcriber.set_text_callback(self._append_transcription)
        self.transcriber.start()

        self.is_recording = True
        self._ensure_key_point_extractor_state()

        if save_audio:
            base = self.audio_input.get_base_filename()
            self.file_label_var.set(f"Saving to: {self.records_folder}\\{base}.[wav,txt]")
        else:
            self.file_label_var.set("Audio/Text saving disabled")
        self.status_var.set("Recording and transcribing...")

    def _stop_recording(self):
        if not self.is_recording:
            return
        self.status_var.set("Stopping...")

        # Flush pending text
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
        if self.audio_input:
            wav_path = self.audio_input.get_wav_path()
            self.audio_input.stop()
            self.audio_input = None
            if wav_path:
                self.file_label_var.set(f"Saved: {wav_path} (and .txt)")

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._set_controls_enabled(True)
        self.is_recording = False
        self.status_var.set("Stopped. Files saved in 'records' folder.")

    def _set_controls_enabled(self, enabled: bool):
        state = "readonly" if enabled else "disabled"
        normal_state = "normal" if enabled else "disabled"

        self.device_combo.config(state=state)
        self.model_combo.config(state=state)
        self.lang_combo.config(state=state)
        self.device_asr_combo.config(state=state)
        self.file_path_entry.config(state=normal_state)

    # ----------------------------------------------------------
    # Transcription Display with Speaker Colors
    # ----------------------------------------------------------

    def _append_transcription(self, result: TranscriptionResult):
        """Callback from FunASRTranscriber — thread-safe."""
        with self.caption_lock:
            text = result.text
            self.full_transcription_text += text
            self.caption_total_chars += len(text)
            if len(self.full_transcription_text) > self.MAX_CAPTION_HISTORY_CHARS:
                self.full_transcription_text = self.full_transcription_text[-self.MAX_CAPTION_HISTORY_CHARS:]

        with self._pending_lock:
            self._pending_text_buffer.append(result)
            if self._flush_after_id is None:
                self._flush_after_id = self.root.after(self.FLUSH_INTERVAL_MS, self._flush_pending_text)

    def _flush_pending_text(self):
        with self._pending_lock:
            self._flush_after_id = None
            buffer = self._pending_text_buffer
            self._pending_text_buffer = []

        if buffer:
            self._append_text_impl(buffer)

    def _append_text_impl(self, results):
        """Insert transcription results with speaker-colored labels."""
        dark = self.dark_mode_var.get()
        colors = SPEAKER_DARK_COLORS if dark else SPEAKER_COLORS

        for result in results:
            if isinstance(result, str):
                # Legacy string (shouldn't happen but safe fallback)
                self.text_area.insert(tk.END, result)
                continue

            speaker = result.speaker
            if speaker:
                # Create tag for this speaker if needed
                if speaker not in self._speaker_tags_created:
                    spk_idx = 0
                    if speaker.startswith("SPEAKER_"):
                        try:
                            spk_idx = int(speaker.split("_")[1])
                        except ValueError:
                            pass
                    color = colors[spk_idx % len(colors)]
                    self.text_area.tag_configure(speaker, foreground=color, font=("Microsoft YaHei", 11, "bold"))
                    self._speaker_tags_created[speaker] = True

                # Insert label
                label = f"[{speaker.replace('SPEAKER_', 'SPK ')}] "
                self.text_area.insert(tk.END, label, speaker)
                # Insert text in normal style
                self.text_area.insert(tk.END, result.text)
            else:
                self.text_area.insert(tk.END, result.text)

        self.text_area.see(tk.END)
        self._trim_widget(self.text_area, self.MAX_TEXT_AREA_CHARS)

    def _trim_widget(self, widget, max_chars):
        content = widget.get(1.0, tk.END)
        if len(content) > max_chars:
            trim_chars = len(content) - max_chars
            end_trim = widget.index(f"1.0 + {trim_chars} chars")
            widget.delete(1.0, end_trim)

    # ----------------------------------------------------------
    # Key Points
    # ----------------------------------------------------------

    def _append_key_points(self, text):
        self.root.after(0, self._append_key_points_impl, text)

    def _append_key_points_impl(self, text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.key_points_area.insert(tk.END, f"[{timestamp}]\n{text.strip()}\n\n")
        self.key_points_area.see(tk.END)
        self._trim_widget(self.key_points_area, self.MAX_KEY_POINTS_CHARS)

    def _get_recent_captions(self):
        with self.caption_lock:
            joined = self.full_transcription_text
            total_chars = self.caption_total_chars
        return joined, total_chars

    def _get_extract_interval(self):
        try:
            return float(self.extract_interval_var.get())
        except (TypeError, ValueError):
            return 20.0

    def _ensure_key_point_extractor_state(self):
        if not self.is_recording:
            return
        if self.extract_enabled_var.get():
            if self.key_point_extractor is None:
                self.key_point_extractor = KeyPointExtractor(
                    interval_getter=self._get_extract_interval,
                    text_provider=self._get_recent_captions,
                    extract_callback=self._extract_brief_key_points,
                    output_callback=self._append_key_points,
                )
                self.key_point_extractor.start()
        elif self.key_point_extractor is not None:
            self.key_point_extractor.stop()
            self.key_point_extractor = None

    def _get_extract_timeout(self):
        try:
            return max(10.0, float(self.extract_timeout_var.get()))
        except (TypeError, ValueError):
            return 90.0

    def _extract_brief_key_points(self, captions_text):
        """Call Ollama to extract key points."""
        prompt = self.extract_prompt_var.get().strip() or "Extract brief key points from the captions."
        model = self.ollama_model_var.get().strip()
        base_url = self.ollama_url_var.get().strip().rstrip("/")
        if not model or not base_url:
            return "[Key point extraction skipped] Configure Ollama URL and model."

        timeout_seconds = self._get_extract_timeout()

        try:
            payload = {
                "model": model,
                "prompt": f"{prompt}\n\nLive captions:\n{captions_text}\n",
                "stream": False,
                "options": {"num_predict": 200},
            }
            req = urllib.request.Request(
                url=f"{base_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
                data = json.loads(body)

            # Extract text from response
            if isinstance(data, dict):
                for key in ("response", "text", "output"):
                    if key in data and data[key]:
                        text = str(data[key]).strip()
                        if text:
                            return text
            return ""
        except socket.timeout:
            return f"[Key point extraction skipped] Timed out after {int(timeout_seconds)}s."
        except Exception as e:
            return f"[Key point extraction skipped] Error: {e}"

    # ----------------------------------------------------------
    # Theme
    # ----------------------------------------------------------

    def _on_theme_toggled(self):
        self._apply_theme()
        self._schedule_config_save()

    def _apply_theme(self):
        dark = self.dark_mode_var.get()
        self.style.theme_use("clam")

        if dark:
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
        self.style.configure("TRadiobutton", background=palette["bg"], foreground=palette["text"])
        self.style.map("TRadiobutton", foreground=[("disabled", palette["muted"])])
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
            fieldbackground=[("readonly", palette["panel"]), ("disabled", palette["panel_alt"])],
            selectbackground=[("readonly", palette["panel"])],
            selectforeground=[("readonly", palette["text"])],
        )

        self.status_label.configure(background=palette["panel_alt"], foreground=palette["text"])

        text_widgets = [self.text_area, self.key_points_area]
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

        for frame in (self.text_frame, self.points_frame, self.extract_cfg,
                      self.input_frame, self.asr_frame):
            frame.configure(style="TLabelframe")

        # Re-apply speaker tag colors for the current theme
        colors = SPEAKER_DARK_COLORS if dark else SPEAKER_COLORS
        for speaker in self._speaker_tags_created:
            spk_idx = 0
            if speaker.startswith("SPEAKER_"):
                try:
                    spk_idx = int(speaker.split("_")[1])
                except ValueError:
                    pass
            color = colors[spk_idx % len(colors)]
            self.text_area.tag_configure(speaker, foreground=color, font=("Microsoft YaHei", 11, "bold"))

    def _apply_extractor_ui_state(self):
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

    def _on_extractor_enable_toggled(self):
        self._apply_extractor_ui_state()
        self._ensure_key_point_extractor_state()
        self._schedule_config_save()

    # ----------------------------------------------------------
    # Config Persistence
    # ----------------------------------------------------------

    def _setup_config_autosave(self):
        watched_vars = [
            self.input_mode_var,
            self.model_var,
            self.lang_var,
            self.device_asr_var,
            self.dark_mode_var,
            self.extract_enabled_var,
            self.ollama_url_var,
            self.ollama_model_var,
            self.extract_prompt_var,
            self.extract_interval_var,
            self.extract_timeout_var,
            self.simulate_realtime_var,
            self.save_audio_var,
        ]
        for var in watched_vars:
            var.trace_add("write", self._on_config_ui_changed)

    def _on_config_ui_changed(self, *args):
        if self._suspend_config_autosave:
            return
        self._apply_extractor_ui_state()
        self._schedule_config_save()

    def _schedule_config_save(self):
        if self._suspend_config_autosave:
            return
        if self._config_save_after_id is not None:
            self.root.after_cancel(self._config_save_after_id)
        self._config_save_after_id = self.root.after(400, self._flush_config_save)

    def _flush_config_save(self):
        self._config_save_after_id = None
        self._save_config()

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

    def _load_config(self):
        if not os.path.exists(self.config_path):
            # Trigger model change to update UI
            self._on_model_changed()
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.input_mode_var.set(str(data.get("input_mode", self.input_mode_var.get())))
            self.model_var.set(str(data.get("model", self.model_var.get())))
            self.lang_var.set(str(data.get("language", self.lang_var.get())))
            self.device_asr_var.set(str(data.get("device", self.device_asr_var.get())))
            self.dark_mode_var.set(self._as_bool(data.get("dark_mode"), self.dark_mode_var.get()))
            self.save_audio_var.set(self._as_bool(data.get("save_audio"), self.save_audio_var.get()))
            self.extract_enabled_var.set(self._as_bool(data.get("extract_enabled"), self.extract_enabled_var.get()))
            self.ollama_url_var.set(str(data.get("ollama_url", self.ollama_url_var.get())))
            self.ollama_model_var.set(str(data.get("ollama_model", self.ollama_model_var.get())))
            self.extract_prompt_var.set(str(data.get("extract_prompt", self.extract_prompt_var.get())))
            self.extract_interval_var.set(str(data.get("extract_interval", self.extract_interval_var.get())))
            self.extract_timeout_var.set(str(data.get("extract_timeout", self.extract_timeout_var.get())))
            self.simulate_realtime_var.set(self._as_bool(data.get("simulate_realtime"), self.simulate_realtime_var.get()))
            self.file_path_var.set(str(data.get("file_path", "")))
        except Exception as e:
            self.status_var.set(f"Config load failed: {e}")

        # Update UI state after loading
        self._on_model_changed()

    def _save_config(self):
        data = {
            "input_mode": self.input_mode_var.get(),
            "model": self.model_var.get().strip(),
            "language": self.lang_var.get().strip(),
            "device": self.device_asr_var.get().strip(),
            "dark_mode": self.dark_mode_var.get(),
            "save_audio": self.save_audio_var.get(),
            "extract_enabled": self.extract_enabled_var.get(),
            "ollama_url": self.ollama_url_var.get().strip(),
            "ollama_model": self.ollama_model_var.get().strip(),
            "extract_prompt": self.extract_prompt_var.get().strip(),
            "extract_interval": self.extract_interval_var.get().strip(),
            "extract_timeout": self.extract_timeout_var.get().strip(),
            "simulate_realtime": self.simulate_realtime_var.get(),
            "file_path": self.file_path_var.get().strip(),
        }
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.status_var.set(f"Config save failed: {e}")

    # ----------------------------------------------------------
    # Misc
    # ----------------------------------------------------------

    def _clear_text(self):
        self.text_area.delete(1.0, tk.END)
        self.key_points_area.delete(1.0, tk.END)
        with self.caption_lock:
            self.full_transcription_text = ""
            self.caption_total_chars = 0
        self._speaker_tags_created.clear()
        self.status_var.set("Text cleared.")

    def _open_records_folder(self):
        folder = os.path.abspath(self.records_folder)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            import subprocess
            subprocess.run(["open", folder])
        else:
            import subprocess
            subprocess.run(["xdg-open", folder])

    def _on_closing(self):
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
        self._save_config()
        self._stop_recording()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ------------------------------
# Entry Point
# ------------------------------

if __name__ == "__main__":
    app = MeetingCaptionApp()
    app.run()
