"""
Unified Audio Input Module
Supports three input sources: Microphone, Speaker (WASAPI loopback), and File.
All sources output 16kHz mono int16 numpy arrays to a shared audio_queue.
"""

import os
import queue
import threading
import time
import wave
import tempfile
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np

# pyaudiowpatch for WASAPI loopback; pyaudio for standard mic
try:
    import pyaudiowpatch as pyaudio
    PYAUDIOWP_AVAILABLE = True
except ImportError:
    try:
        import pyaudio
        PYAUDIOWP_AVAILABLE = False
    except ImportError:
        pyaudio = None
        PYAUDIOWP_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None
    SOUNDFILE_AVAILABLE = False


# ------------------------------
# Base Class
# ------------------------------

class AudioInputBase(ABC):
    """Abstract base class for audio input sources.

    All subclasses push 16kHz mono int16 numpy arrays into audio_queue.
    """

    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 3.0, save_audio: bool = True):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.save_audio = save_audio

        self.audio_queue: queue.Queue = queue.Queue()
        self.is_recording = False

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # File writing
        self._wave_file = None
        self._base_filename: Optional[str] = None
        self._wave_fullpath: Optional[str] = None
        self._write_thread: Optional[threading.Thread] = None
        self._write_queue: queue.Queue = queue.Queue()

    def start(self, records_folder: str = "records"):
        """Start capturing audio."""
        if self.is_recording:
            return
        self.is_recording = True
        self._stop_event.clear()
        self.audio_queue = queue.Queue()
        self._write_queue = queue.Queue()

        if self.save_audio:
            os.makedirs(records_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._base_filename = f"meeting_{timestamp}"
            self._wave_fullpath = os.path.join(records_folder, self._base_filename + ".wav")

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        if self.save_audio:
            self._write_thread = threading.Thread(target=self._file_writer_loop, daemon=True)
            self._write_thread.start()

    def stop(self):
        """Stop capturing and clean up."""
        self.is_recording = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._write_thread:
            self._write_queue.put(None)
            self._write_thread.join(timeout=3.0)
            self._write_thread = None
        if self._wave_file:
            self._wave_file.close()
            self._wave_file = None

    def get_base_filename(self) -> Optional[str]:
        return self._base_filename

    def get_wav_path(self) -> Optional[str]:
        return self._wave_fullpath

    @abstractmethod
    def _capture_loop(self):
        """Main capture loop — subclass must implement."""
        ...

    def _init_wave_file(self, channels: int, sample_rate: int):
        """Open WAV file for writing with given params."""
        if self.save_audio and self._wave_fullpath:
            self._wave_file = wave.open(self._wave_fullpath, 'wb')
            self._wave_file.setnchannels(channels)
            self._wave_file.setsampwidth(2)
            self._wave_file.setframerate(sample_rate)

    def _write_audio_data(self, raw_bytes: bytes):
        """Queue raw audio bytes for file writing."""
        if self.save_audio:
            self._write_queue.put(raw_bytes)

    def _file_writer_loop(self):
        """Write audio data to WAV file without blocking the capture thread."""
        while True:
            item = self._write_queue.get()
            if item is None:
                break
            try:
                if self._wave_file:
                    self._wave_file.writeframes(item)
            except Exception as e:
                print(f"[AudioInput] File write error: {e}")

    @staticmethod
    def _resample_to_16k(mono_int16: np.ndarray, source_rate: int, target_rate: int = 16000) -> np.ndarray:
        """Simple linear interpolation resampling to target_rate."""
        if source_rate == target_rate:
            return mono_int16
        ratio = target_rate / source_rate
        indices = np.arange(0, len(mono_int16), 1 / ratio)
        indices = indices[indices < len(mono_int16)]
        return np.interp(indices, np.arange(len(mono_int16)), mono_int16).astype(np.int16)

    @staticmethod
    def _to_mono(raw_int16: np.ndarray, channels: int) -> np.ndarray:
        """Convert multi-channel audio to mono by averaging."""
        if channels == 2:
            return raw_int16.reshape(-1, 2).mean(axis=1).astype(np.int16)
        return raw_int16


# ------------------------------
# Microphone Input
# ------------------------------

class MicInput(AudioInputBase):
    """Captures audio from a standard microphone input device."""

    def __init__(self, device_index: int, sample_rate: int = 16000, chunk_duration: float = 3.0, save_audio: bool = True):
        super().__init__(sample_rate, chunk_duration, save_audio)
        self.device_index = device_index

    def _capture_loop(self):
        p = None
        stream = None
        try:
            p = pyaudio.PyAudio()
            device_info = p.get_device_info_by_index(self.device_index)
            channels = min(device_info["maxInputChannels"], 2)
            if channels <= 0:
                raise ValueError(f"Device {self.device_index} has no input channels.")
            source_rate = int(device_info["defaultSampleRate"])

            stream = p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=source_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=1024,
            )

            self._init_wave_file(channels, source_rate)

            buffer = np.array([], dtype=np.int16)

            while not self._stop_event.is_set():
                try:
                    data = stream.read(1024, exception_on_overflow=False)
                except Exception:
                    time.sleep(0.01)
                    continue

                raw_chunk = np.frombuffer(data, dtype=np.int16)
                self._write_audio_data(data)

                mono = self._to_mono(raw_chunk, channels)
                resampled = self._resample_to_16k(mono, source_rate, self.sample_rate)
                buffer = np.concatenate([buffer, resampled])

                while len(buffer) >= self.chunk_size:
                    chunk = buffer[:self.chunk_size].copy()
                    buffer = buffer[self.chunk_size:]
                    self.audio_queue.put(chunk)

        except Exception as e:
            print(f"[MicInput] Capture error: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            if p:
                p.terminate()
            if self.save_audio:
                self._write_queue.put(None)


# ------------------------------
# Speaker (WASAPI Loopback) Input
# ------------------------------

class SpeakerInput(AudioInputBase):
    """Captures audio from WASAPI loopback (system audio / speaker output)."""

    def __init__(self, device_index: int, sample_rate: int = 16000, chunk_duration: float = 3.0, save_audio: bool = True):
        super().__init__(sample_rate, chunk_duration, save_audio)
        self.device_index = device_index

    def _capture_loop(self):
        p = None
        stream = None
        try:
            p = pyaudio.PyAudio()
            device_info = p.get_device_info_by_index(self.device_index)
            channels = device_info["maxInputChannels"]
            if channels <= 0:
                raise ValueError(f"Device {self.device_index} has no input channels.")
            source_rate = int(device_info["defaultSampleRate"])

            stream = p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=source_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=1024,
            )

            self._init_wave_file(channels, source_rate)

            buffer = np.array([], dtype=np.int16)

            while not self._stop_event.is_set():
                try:
                    data = stream.read(1024, exception_on_overflow=False)
                except Exception:
                    time.sleep(0.01)
                    continue

                raw_chunk = np.frombuffer(data, dtype=np.int16)
                self._write_audio_data(data)

                mono = self._to_mono(raw_chunk, channels)
                resampled = self._resample_to_16k(mono, source_rate, self.sample_rate)
                buffer = np.concatenate([buffer, resampled])

                while len(buffer) >= self.chunk_size:
                    chunk = buffer[:self.chunk_size].copy()
                    buffer = buffer[self.chunk_size:]
                    self.audio_queue.put(chunk)

        except Exception as e:
            print(f"[SpeakerInput] Capture error: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            if p:
                p.terminate()
            if self.save_audio:
                self._write_queue.put(None)


# ------------------------------
# File Input
# ------------------------------

class FileInput(AudioInputBase):
    """Reads audio from a local file (WAV, MP3, MP4, etc.) and feeds chunks into the queue.

    If simulate_realtime is True, chunks are pushed at roughly real-time speed.
    Otherwise, all chunks are pushed as fast as possible.
    """

    def __init__(self, file_path: str, sample_rate: int = 16000, chunk_duration: float = 5.0,
                 simulate_realtime: bool = True, save_audio: bool = True):
        super().__init__(sample_rate, chunk_duration, save_audio)
        self.file_path = file_path
        self.simulate_realtime = simulate_realtime
        self._temp_wav: Optional[str] = None

    def _ensure_wav_16k_mono(self) -> str:
        """Convert the input file to 16kHz mono WAV if needed, return the WAV path."""
        ext = os.path.splitext(self.file_path)[1].lower()

        # If already a WAV, try reading directly
        if ext == ".wav" and SOUNDFILE_AVAILABLE:
            try:
                info = sf.info(self.file_path)
                if info.samplerate == 16000 and info.channels == 1:
                    return self.file_path
            except Exception:
                pass

        # Otherwise convert via ffmpeg
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        cmd = [
            "ffmpeg", "-y",
            "-i", self.file_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")

        self._temp_wav = tmp_path
        return tmp_path

    def _capture_loop(self):
        wav_path = None
        try:
            wav_path = self._ensure_wav_16k_mono()

            if SOUNDFILE_AVAILABLE:
                speech, sr = sf.read(wav_path, dtype='int16')
            else:
                # Fallback: use scipy
                from scipy.io import wavfile as sp_wavfile
                sr, speech = sp_wavfile.read(wav_path)
                if speech.ndim == 2:
                    speech = speech.mean(axis=1).astype(np.int16)

            if sr != self.sample_rate:
                speech = self._resample_to_16k(speech, sr, self.sample_rate)

            # Ensure 1D
            if speech.ndim > 1:
                speech = speech.mean(axis=1).astype(np.int16)

            # Ensure int16
            if speech.dtype != np.int16:
                speech = (speech * 32767).astype(np.int16) if speech.dtype in (np.float32, np.float64) else speech.astype(np.int16)

            self._init_wave_file(1, self.sample_rate)

            total_samples = len(speech)
            offset = 0

            while offset < total_samples and not self._stop_event.is_set():
                end = min(offset + self.chunk_size, total_samples)
                chunk = speech[offset:end].copy()

                # Write original bytes to WAV
                self._write_audio_data(chunk.tobytes())

                # Pad last chunk if too short
                if len(chunk) < self.chunk_size:
                    padded = np.zeros(self.chunk_size, dtype=np.int16)
                    padded[:len(chunk)] = chunk
                    chunk = padded

                self.audio_queue.put(chunk)

                if self.simulate_realtime:
                    # Sleep for the chunk duration
                    self._stop_event.wait(self.chunk_duration)

                offset = end

        except Exception as e:
            print(f"[FileInput] Capture error: {e}")
        finally:
            if self.save_audio:
                self._write_queue.put(None)
            # Clean up temp WAV
            if self._temp_wav and os.path.exists(self._temp_wav):
                try:
                    os.remove(self._temp_wav)
                except Exception:
                    pass
                self._temp_wav = None


# ------------------------------
# Device Discovery Helpers
# ------------------------------

def list_mic_devices() -> List[Tuple[int, str, str]]:
    """List available microphone input devices. Returns [(index, name, host_api)]."""
    if pyaudio is None:
        return []
    devices = []
    try:
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                host_api = p.get_host_api_info_by_index(info["hostApi"])
                # Exclude loopback devices (they show up as input devices on Windows)
                is_loopback = "loopback" in info["name"].lower()
                if not is_loopback:
                    devices.append((i, info["name"], host_api["name"]))
        p.terminate()
    except Exception as e:
        print(f"[AudioInput] Error listing mic devices: {e}")
    return devices


def list_speaker_devices() -> List[Tuple[int, str, str]]:
    """List available WASAPI loopback devices. Returns [(index, name, host_api)]."""
    if not PYAUDIOWP_AVAILABLE:
        return []
    devices = []
    try:
        p = pyaudio.PyAudio()
        for device in p.get_loopback_device_info_generator():
            name = device["name"]
            idx = device["index"]
            host_api = p.get_host_api_info_by_index(device["hostApi"])
            devices.append((idx, name, host_api["name"]))
        p.terminate()
    except Exception as e:
        print(f"[AudioInput] Error listing speaker devices: {e}")
    return devices
