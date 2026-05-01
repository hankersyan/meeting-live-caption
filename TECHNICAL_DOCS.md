# Meeting Live Caption - Technical Documentation

## Overview

This document describes the technical implementation of the Meeting Live Caption application, which provides real-time audio recording and transcription using Whisper AI.

## Architecture

The application follows a multi-threaded design with five main components:

### 1. AudioRecorder Class

Handles WASAPI loopback recording with the following responsibilities:
- Captures audio from the selected loopback device
- Resamples audio to 16kHz for Whisper compatibility
- Manages audio chunks for transcription
- Saves audio data to WAV files in the 'records' folder
- Uses threading for non-blocking audio capture

Key features:
- WASAPI loopback recording for system audio capture
- Dynamic resampling from source rate to 16kHz
- Chunk-based processing (2-second chunks by default)
- Separate file writing thread to prevent blocking
- Mono conversion for multi-channel audio

### 2. Transcriber Class

Performs real-time transcription using Whisper with these functions:
- Retrieves audio chunks from the AudioRecorder queue
- Applies Whisper model for speech-to-text conversion
- Handles overlapping audio for continuity
- Saves transcribed text to TXT files
- Updates GUI with transcribed text

Key features:
- Faster Whisper model integration
- VAD (Voice Activity Detection) filtering
- Overlap handling to maintain context
- Temperature and beam search settings for quality
- Multi-language support

### 3. MeetingRecorderApp Class

Provides the GUI interface with these capabilities:
- Audio device selection and management
- Model and language selection
- Start/stop recording controls
- Live transcription display
- File management and folder access
- Threading coordination

### 4. KeyPointExtractor Class

Performs periodic LLM-based summarization of recent captions using Ollama:
- Pulls recent live caption text from in-memory history
- Calls Ollama `/api/generate` with configurable URL, model, and prompt
- Runs at a configurable refresh interval
- Updates GUI with timestamped brief key-point summaries
- Uses a dedicated thread and stop event for clean shutdown

### 5. SpeakerDiarizer Class

Performs periodic speaker diarization using pyannote.audio:
- Collects accumulated audio via AudioBuffer (ring buffer, max 120s)
- Periodically runs pyannote speaker-diarization-3.1 pipeline on collected audio
- Maps diarization results (speaker + time intervals) to transcription segments via LabelMapper
- Updates the speaker-labeled transcription area with color-coded speaker tags
- Requires HuggingFace access token for model access
- Gracefully degrades if pyannote.audio is not installed

### Supporting Data Structures

- **AudioBuffer**: Thread-safe ring buffer that accumulates mono int16 audio (max 120s), supplies audio snapshots to the diarizer
- **SegmentStore**: Thread-safe store for `TranscriptionSegment(start, end, text, speaker)` objects produced by the Transcriber
- **TranscriptionSegment**: Dataclass with absolute timestamps, transcribed text, and optional speaker label
- **map_speakers_to_segments()**: Maps diarization (start, end, speaker) tuples onto transcription segments using timestamp overlap analysis

## Audio Processing Pipeline

```
System Audio → WASAPI Loopback → AudioRecorder → Resample → Queue → Transcriber → Whisper → Text Output
                                                                       ↓                       ↓
                                                                AudioBuffer            SegmentStore
                                                                       ↓                       ↓
                                                             SpeakerDiarizer ────→ LabelMapper ──→ Speaker-labeled Text
                                                                       ↓
                                                                pyannote.audio pipeline

                                            ↓
                                        WAV File

Additional periodic paths:
- KeyPointExtractor → Ollama → Key Points
- SpeakerDiarizer → pyannote → Speaker-labeled Transcription
```

The pipeline ensures low-latency processing through:
- Small audio chunks (2 seconds)
- Overlapping segments for context
- Asynchronous file writing
- Non-blocking UI updates

## Key Technologies Used

- **PyAudioWPA** - Cross-platform audio I/O with WASAPI support
- **Faster Whisper** - High-performance Whisper implementation
- **NumPy** - Audio signal processing
- **Tkinter** - GUI interface
- **Threading** - Concurrent processing
- **pyannote.audio** - Speaker diarization (optional)
- **PyTorch** - Deep learning framework for pyannote (optional)

## Configuration Parameters

### AudioRecorder
- `sample_rate`: Target sample rate (default: 16000 Hz)
- `chunk_duration`: Duration of audio chunks (default: 2.0 seconds)
- `save_audio`: Whether to save audio files (default: True)

### Transcriber
- `model_size`: Whisper model size ("tiny", "base", "small", "medium", "large-v2", "large-v3")
- `language`: Target language ("en", "zh", "es", etc. or "auto")
- `beam_size`: Beam search parameter (default: 5)
- `best_of`: Best-of parameter for beam search (default: 5)
- `segment_store`: SegmentStore instance for storing timestamped segments (optional)
- `audio_buffer`: AudioBuffer instance for accumulating audio (optional)

### KeyPointExtractor / Ollama
- `ollama_url`: Base URL for Ollama API (default: `http://localhost:11434`)
- `ollama_model`: Model name used for extraction (default: `LiquidAI/lfm2.5-1.2b-instruct`)
- `extract_prompt`: Prompt template used to extract key points
- `extract_interval`: Refresh interval in seconds (default: 20)

### SpeakerDiarizer / pyannote
- `diarization_enabled`: Enable/disable speaker diarization (default: false)
- `hf_token`: HuggingFace access token for pyannote model access
- `diarization_interval`: Seconds between diarization runs (default: 30)
- `max_speakers`: Maximum number of speakers to detect (0 = auto)
- `diarization_device`: Inference device ("cpu" or "cuda")

## File Management

All output files are stored in the 'records' subdirectory with timestamped filenames:
- Audio files: `meeting_YYYYMMDD_HHMMSS.wav`
- Text files: `meeting_YYYYMMDD_HHMMSS.txt`
- Speaker-labeled text: `meeting_YYYYMMDD_HHMMSS_labeled.txt` (only when diarization is enabled)

Each session creates a set of files with the same timestamp.

## Error Handling

The application implements comprehensive error handling:
- Device connection failures
- Model loading errors
- File I/O errors
- Audio processing exceptions
- Threading synchronization issues
- Missing diarization dependencies (graceful degradation)
- Diarization pipeline load failures (continues transcription without labels)

## Performance Considerations

- CPU-intensive transcription process
- Memory usage scales with recording duration
- Disk I/O optimization through separate writing thread
- Adjustable model sizes for performance vs. quality trade-off
- Diarization runs on a separate periodic schedule to avoid impacting transcription
- Audio buffer uses ring buffer (max 120s, ~3.8MB) to limit memory usage

## Platform Compatibility

Currently designed for Windows due to WASAPI dependency. The core architecture could be adapted for other platforms using different audio backends (PortAudio, PulseAudio, etc.).

## Dependencies

- `faster-whisper`: Fast Whisper implementation
- `pyaudiowpatch`: PyAudio with WASAPI support
- `numpy`: Numerical computations
- Standard library: threading, queue, wave, tkinter, etc.
- `pyannote.audio` (optional): Speaker diarization pipeline
- `torch` (optional): PyTorch backend for pyannote