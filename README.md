# Meeting Live Caption

Real-time Meeting Audio and Text Recorder with **FunASR** speech recognition and speaker diarization. Records from microphone, system audio, or file, transcribes with speaker labels, and saves both audio (WAV) and transcription (TXT).

## Features

- **Three Input Sources**: Microphone, Speaker (WASAPI loopback), Audio File (WAV/MP3/MP4)
- **FunASR Speech Recognition**: Non-streaming (with speaker diarization) and streaming (low latency) modes
- **Speaker Diarization**: Automatic speaker identification via cam++ model, color-coded display
- **Multiple Models**: paraformer-zh, SenseVoiceSmall, paraformer-zh-streaming, Fun-ASR-Nano
- **Key Point Extraction**: Periodic AI summaries via Ollama
- **Dark Mode**: Comfortable viewing in low-light environments
- **File Saving**: Simultaneous audio and text output with timestamps

## Requirements

- Python 3.8+
- Windows OS (for WASAPI loopback support)
- [ffmpeg](https://ffmpeg.org/) (for MP3/MP4 file input)

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install funasr modelscope numpy soundfile pyaudiowpatch
   ```
3. Run the application:
   ```bash
   python app.py
   ```

### Optional (for legacy faster-whisper version)
```bash
pip install faster-whisper
python main.py  # Original whisper-based version
```

### Optional (for key-point extraction)
- [Ollama](https://ollama.com/) running locally or remotely
- A pulled model (example: `ollama pull LiquidAI/lfm2.5-1.2b-instruct`)

## Usage

1. Launch: `python app.py`
2. Select input source: Microphone / Speaker / File
3. Choose FunASR model and enable speaker diarization if desired
4. Click "Start Recording"
5. View live transcription with speaker color labels
6. Click "Stop Recording" when finished
7. Find saved files in the `records/` folder

### CLI Transcription

```bash
# Transcribe a file with speaker diarization
python asr_speaker_diarization.py meeting.wav

# Simple transcription
python asr.py recording.mp3 paraformer-zh cpu
```

## FunASR Models

| Model | Type | Languages | Speaker Diarization | Description |
|-------|------|-----------|-------------------|-------------|
| paraformer-zh | Non-streaming | Chinese | Yes | Best for Chinese meetings |
| SenseVoiceSmall | Non-streaming | zh, en, ja, ko, yue | Yes | Multilingual support |
| paraformer-zh-streaming | Streaming | Chinese | No | Low latency (~600ms) |
| Fun-ASR-Nano-2512 | Non-streaming | 31 languages | Yes | Multilingual, real-time capable |

## File Structure

```
├── app.py                        # FunASR GUI application (entry point)
├── audio_input.py                # Audio input: Mic, Speaker, File
├── asr_engine.py                 # FunASR engine + speaker diarization
├── asr.py                        # CLI file transcription
├── asr_speaker_diarization.py    # CLI speaker diarization
├── main.py                       # Legacy faster-whisper version
├── config_funasr.json            # FunASR app configuration
└── records/                      # Output folder
    ├── meeting_YYYYMMDD_HHMMSS.wav
    └── meeting_YYYYMMDD_HHMMSS.txt
```

## Architecture

```
Audio Input (Mic/Speaker/File)
    → audio_queue (16kHz mono int16)
    → FunASRTranscriber
        → AutoModel (ASR + VAD + Punc + cam++ Speaker)
        → SpeakerRegistry (cross-chunk consistency)
    → GUI (speaker color-coded display)
    → Text File
    → KeyPointExtractor → Ollama → Key Points
```

## License

This project is open-source. See the LICENSE file for details.
