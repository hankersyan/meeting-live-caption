# Meeting Live Caption - Technical Documentation (FunASR Edition)

## Overview

This document describes the technical implementation of the Meeting Live Caption application using FunASR for real-time speech recognition and speaker diarization.

## Architecture

The application follows a modular multi-threaded design with five main components:

### 1. AudioInputBase & Subclasses (`audio_input.py`)

Abstract base class for audio input with three concrete implementations:

- **MicInput**: Captures audio from a standard microphone via pyaudio. Lists non-loopback input devices.
- **SpeakerInput**: Captures system audio via WASAPI loopback using pyaudiowpatch. Lists loopback devices only.
- **FileInput**: Reads audio files (WAV/MP3/MP4) via soundfile + ffmpeg. Supports real-time simulation mode.

All subclasses:
- Output 16kHz mono int16 numpy arrays to `audio_queue`
- Support WAV file saving with separate write thread
- Handle resampling from source rate to 16kHz
- Use `stop_event` for clean shutdown

Key features:
- WASAPI loopback recording for system audio capture
- Dynamic resampling via linear interpolation
- Configurable chunk duration (default: 3s for Mic/Speaker, 5s for File)
- Separate file writing thread to prevent blocking

### 2. FunASRTranscriber (`asr_engine.py`)

Processes audio chunks from queue using FunASR AutoModel API:

- **Non-streaming mode**: Uses `paraformer-zh` + `fsmn-vad` + `ct-punc` + `cam++` pipeline
  - Produces `sentences` field with per-sentence speaker labels
  - Delay ~3-5 seconds, acceptable for meetings
  - Best speaker diarization quality
- **Streaming mode**: Uses `paraformer-zh-streaming` with chunk-based processing
  - 600ms display granularity, 300ms lookahead
  - No speaker diarization support
  - Near-real-time latency

Key features:
- `SpeakerRegistry` for cross-chunk speaker consistency (cosine similarity matching)
- `TranscriptionResult` dataclass with text, speaker, timestamps
- `transcribe_file()` standalone function for file transcription
- Model config metadata (MODEL_CONFIG) for UI integration

### 3. MeetingCaptionApp (`app.py`)

Main GUI application with tkinter:

- Input source selector (Mic/Speaker/File)
- FunASR model selector with auto-configuration
- Speaker diarization toggle
- Speaker color-coded transcription display using tkinter Text tags
- Key point extraction via Ollama (preserved from original)
- Dark/light theme with modern styling
- Config persistence to `config_funasr.json`

### 4. KeyPointExtractor (`app.py`)

Periodic LLM-based summarization of recent captions via Ollama:
- Pulls recent caption text from in-memory history
- Calls Ollama `/api/generate` with configurable URL, model, and prompt
- Runs at configurable refresh interval
- Updates GUI with timestamped summaries

### 5. SpeakerRegistry (`asr_engine.py`)

Maintains cross-chunk speaker identity consistency:
- Stores speaker embedding vectors
- Matches new embeddings via cosine similarity (default threshold: 0.65)
- Uses exponential moving average to update stored embeddings
- Thread-safe with lock protection

## Audio Processing Pipeline

```
Audio Input (Mic/Speaker/File)
    → 16kHz mono int16 numpy array
    → audio_queue
    → FunASRTranscriber
        → Non-streaming: AutoModel(paraformer-zh + fsmn-vad + ct-punc + cam++)
            → TranscriptionResult(text, speaker, timestamps)
        → Streaming: AutoModel(paraformer-zh-streaming)
            → TranscriptionResult(text, no speaker)
    → GUI callback (speaker color-coded display)
    → Text file save
    → KeyPointExtractor → Ollama → Key Points
```

## FunASR Model Integration

### AutoModel API

```python
# Non-streaming with speaker diarization
model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++",
    device="cpu",
)
result = model.generate(input=audio_float32, batch_size_s=300)
# result[0]["text"] = full text
# result[0]["sentences"] = [{"text": ..., "start": ms, "end": ms, "spk": int}]
```

### Streaming API

```python
model = AutoModel(model="paraformer-zh-streaming", device="cpu")
cache = {}
for chunk in audio_chunks:
    res = model.generate(
        input=chunk, cache=cache, is_final=False,
        chunk_size=[0, 10, 5],
        encoder_chunk_look_back=4,
        decoder_chunk_look_back=1,
    )
```

## Key Technologies

- **FunASR**: Speech recognition, VAD, punctuation, speaker diarization
- **cam++**: Speaker verification/embedding model (7.2M params)
- **PyAudioWPA**: WASAPI loopback audio capture
- **NumPy**: Audio signal processing
- **Tkinter**: GUI framework
- **Ollama**: Key point extraction

## Configuration Parameters

### AudioInputBase
- `sample_rate`: Target sample rate (default: 16000 Hz)
- `chunk_duration`: Duration of audio chunks (default: 3.0s)
- `save_audio`: Whether to save audio files (default: True)

### FunASRTranscriber
- `model_name`: FunASR model (paraformer-zh, SenseVoiceSmall, etc.)
- `language`: Target language (auto, zh, en, ja, ko, yue)
- `enable_diarization`: Enable speaker diarization (default: True)
- `device`: cpu or cuda

### KeyPointExtractor
- `ollama_url`: Ollama API URL
- `ollama_model`: Model name for extraction
- `extract_interval`: Refresh interval in seconds
- `extract_prompt`: Custom extraction prompt

## Performance Considerations

- FunASR non-streaming processes 5s chunk in ~0.5-2s on CPU
- cam++ speaker model is lightweight (7.2M params)
- Streaming mode provides ~600ms latency
- Speaker embedding comparison is negligible (vector dot product)
- Memory usage scales with recording duration (capped at MAX_CAPTION_HISTORY_CHARS)

## Dependencies

- `funasr`: FunASR speech recognition toolkit
- `modelscope`: Model download and management
- `pyaudiowpatch`: WASAPI loopback audio capture
- `numpy`: Numerical computations
- `soundfile`: Audio file reading
- Standard library: threading, queue, wave, tkinter, etc.
