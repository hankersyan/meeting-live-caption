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
from datetime import datetime
from collections import deque

import numpy as np
import pyaudiowpatch as pyaudio

# GUI
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess

# Whisper
from faster_whisper import WhisperModel


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
    
    def __init__(self, recorder: AudioRecorder, model_size="base", language="en", records_folder="records"):
        self.recorder = recorder
        self.model_size = model_size
        self.language = language
        self.records_folder = records_folder
        self.is_running = False
        self.thread = None
        
        self.text_callback = None
        self.model = None
        self.text_file = None
        self.text_path = None
        
    def set_text_callback(self, callback):
        self.text_callback = callback
        
    def start(self):
        if self.is_running:
            return
        self.is_running = True
        
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
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                num_workers=1,
            )
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            if self.text_callback:
                self.text_callback(f"[Error] Failed to load model: {e}\n")
            return
        
        overlap_samples = int(self.recorder.sample_rate * 0.5)
        prev_chunk = np.array([], dtype=np.int16)
        
        while self.is_running:
            try:
                try:
                    audio_chunk = self.recorder.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
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
                for segment in segments:
                    full_text += segment.text
                
                if full_text.strip():
                    full_text = full_text.strip() + " "
                    # Update UI
                    if self.text_callback:
                        self.text_callback(full_text)
                    # Save to file
                    self._save_text(full_text)
                
                prev_chunk = audio_chunk
                
            except Exception as e:
                print(f"Transcription error: {e}")
                time.sleep(0.1)


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
        
        self.p = None
        self.recorder = None
        self.transcriber = None
        self.is_recording = False
        self.available_devices = []
        
        self.setup_ui()
        self.refresh_devices()
        
    def setup_ui(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Row 0: Device selection
        ttk.Label(control_frame, text="Audio Device:").grid(row=0, column=0, sticky=tk.W, padx=(0,5))
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(control_frame, textvariable=self.device_var, state="readonly", width=60)
        self.device_combo.grid(row=0, column=1, sticky=tk.W, padx=(0,10))
        self.device_combo.bind("<<ComboboxSelected>>", self.on_device_selected)
        ttk.Button(control_frame, text="Refresh", command=self.refresh_devices).grid(row=0, column=2, padx=(0,5))
        
        # Model & Language
        ttk.Label(control_frame, text="Model:").grid(row=0, column=3, sticky=tk.W, padx=(10,5))
        self.model_var = tk.StringVar(value="base")
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var,
                                   values=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                                   state="readonly", width=10)
        model_combo.grid(row=0, column=4, sticky=tk.W, padx=(0,5))
        
        ttk.Label(control_frame, text="Language:").grid(row=0, column=5, sticky=tk.W, padx=(10,5))
        self.lang_var = tk.StringVar(value="en")
        lang_combo = ttk.Combobox(control_frame, textvariable=self.lang_var,
                                  values=["en", "zh", "es", "fr", "de", "ja", "ko", "auto"],
                                  state="readonly", width=5)
        lang_combo.grid(row=0, column=6, sticky=tk.W)
        
        # Row 1: Save options
        save_frame = ttk.Frame(self.root, padding="5")
        save_frame.pack(fill=tk.X, padx=10)
        self.save_audio_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(save_frame, text="Save audio & text to 'records' folder", variable=self.save_audio_var).pack(side=tk.LEFT)
        self.file_label_var = tk.StringVar(value="No recording yet")
        ttk.Label(save_frame, textvariable=self.file_label_var, foreground="blue").pack(side=tk.LEFT, padx=(20,0))
        ttk.Button(save_frame, text="Open Records Folder", command=self.open_records_folder).pack(side=tk.RIGHT)
        
        # Buttons
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(fill=tk.X)
        self.start_btn = ttk.Button(button_frame, text="Start Recording", command=self.start_recording)
        self.start_btn.pack(side=tk.LEFT, padx=(0,5))
        self.stop_btn = ttk.Button(button_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0,5))
        self.clear_btn = ttk.Button(button_frame, text="Clear Text", command=self.clear_text)
        self.clear_btn.pack(side=tk.LEFT)
        
        # Status
        self.status_var = tk.StringVar(value="Ready. Click Refresh to list devices.")
        status_label = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding="5")
        status_label.pack(fill=tk.X, padx=10, pady=(0,5))
        
        # Transcription text area
        text_frame = ttk.LabelFrame(self.root, text="Live Transcription", padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.text_area = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, font=("Arial", 11))
        self.text_area.pack(fill=tk.BOTH, expand=True)
        
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
        
        device_index = self.available_devices[selection][0]
        save_audio = self.save_audio_var.get()
        
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.device_combo.config(state=tk.DISABLED)
        
        self.recorder = AudioRecorder(device_index, sample_rate=16000, chunk_duration=2.0, save_audio=save_audio)
        self.recorder.start(records_folder=self.records_folder)
        
        self.transcriber = Transcriber(self.recorder, model_size=self.model_var.get(),
                                       language=self.lang_var.get(), records_folder=self.records_folder)
        self.transcriber.set_text_callback(self.append_text)
        self.transcriber.start()
        
        self.is_recording = True
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
        if self.transcriber:
            self.transcriber.stop()
            self.transcriber = None
        if self.recorder:
            wav_path = self.recorder.get_wav_path()
            self.recorder.stop()
            self.recorder = None
            if wav_path:
                self.file_label_var.set(f"Saved: {wav_path} (and .txt)")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.device_combo.config(state="readonly")
        self.is_recording = False
        self.status_var.set("Stopped. Files saved in 'records' folder.")
    
    def append_text(self, text):
        self.root.after(0, self._append_text_impl, text)
    
    def _append_text_impl(self, text):
        self.text_area.insert(tk.END, text)
        self.text_area.see(tk.END)
    
    def clear_text(self):
        self.text_area.delete(1.0, tk.END)
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