# Meeting Live Caption - Usage Guide (FunASR Edition)

## Getting Started

### Prerequisites

- Windows operating system (for WASAPI loopback)
- Python 3.8 or higher
- [ffmpeg](https://ffmpeg.org/) installed and in PATH (for MP3/MP4 input)
- At least 4GB free disk space for model downloads

### Installation

1. **Install Python Dependencies**
   ```bash
   pip install funasr modelscope numpy soundfile pyaudiowpatch
   ```

2. **First-Time Setup**
   - FunASR models download automatically on first use
   - Model sizes vary: ~220MB (paraformer-zh), ~234MB (SenseVoiceSmall)
   - Internet connection required for initial model download

3. **Launch the Application**
   ```bash
   python app.py
   ```

## Interface Walkthrough

### 1. Input Source Selection

- **Microphone**: Record from a microphone. Select from available mic devices.
- **Speaker (WASAPI)**: Record system audio (Teams, Zoom, browser, etc.). Select a WASAPI loopback device.
- **File**: Transcribe a local audio file (WAV, MP3, MP4, FLAC, etc.). Browse or type the file path.

### 2. ASR Engine Configuration

- **Model**: Choose the FunASR model:
  - `paraformer-zh`: Best for Chinese meetings, supports speaker diarization
  - `SenseVoiceSmall`: Multilingual (Chinese, English, Japanese, Korean, Cantonese)
  - `paraformer-zh-streaming`: Streaming mode, low latency (~600ms), no speaker diarization
  - `Fun-ASR-Nano-2512`: 31 languages, real-time capable
- **Language**: Set based on model (auto-detected by default)
- **Speaker Diarization**: Enable cam++ for automatic speaker identification
- **Device**: CPU or CUDA (GPU)

### 3. Save Options

- **Save audio & text**: Saves WAV + TXT to `records/` folder
- **Open Records Folder**: Quick access to output files

### 4. Key Point Extraction (Ollama)

- **Enable**: Turn periodic extraction on/off
- **URL**: Ollama API URL (default: `http://localhost:11434`)
- **Model**: Ollama model name (e.g., `LiquidAI/lfm2.5-1.2b-instruct`)
- **Refresh(s)**: Interval between extractions
- **Timeout(s)**: Maximum wait time
- **Prompt**: Instruction for key point generation

### 5. Output Area

- **Left panel**: Live transcription with speaker color labels
  - Format: `[SPK 0] Transcribed text`
  - Each speaker has a unique color
- **Right panel**: Timestamped key point summaries

## Step-by-Step Usage

### Recording from System Audio (Speaker)

1. Click **Refresh** to populate WASAPI loopback devices
2. Select your playback device from the dropdown
3. Choose model (e.g., `paraformer-zh` with speaker diarization)
4. Click **Start Recording**
5. Play your meeting audio — transcription appears in real-time
6. Click **Stop Recording** when done

### Recording from Microphone

1. Select **Microphone** input mode
2. Click **Refresh** and select your mic device
3. Choose model and click **Start Recording**
4. Speak — transcription appears with speaker labels

### Transcribing a File

1. Select **File** input mode
2. Click **Browse...** to select an audio/video file
3. Optionally check **Simulate realtime** for paced output
4. Click **Start Recording**
5. Transcription proceeds through the file

### CLI Transcription

```bash
# Transcribe with speaker diarization
python asr_speaker_diarization.py meeting.wav

# Specify model and device
python asr_speaker_diarization.py meeting.mp4 SenseVoiceSmall cuda

# Simple transcription
python asr.py recording.mp3
```

## Model Selection Guide

| Scenario | Recommended Model | Reason |
|----------|------------------|--------|
| Chinese meeting with multiple speakers | paraformer-zh | Best Chinese ASR + speaker diarization |
| Multilingual meeting | SenseVoiceSmall | Supports zh, en, ja, ko, yue |
| Need lowest latency | paraformer-zh-streaming | ~600ms streaming response |
| Many languages | Fun-ASR-Nano-2512 | 31 languages + 7 Chinese dialects |

## Speaker Diarization

- Uses FunASR's **cam++** speaker verification model
- Automatically identifies and labels different speakers
- Each speaker is color-coded in the transcription display
- Supports up to 8 distinct speaker colors
- Works best with non-streaming models
- Only available when the diarization checkbox is enabled

## Output Files

Files are saved in the `records/` folder:
- `meeting_YYYYMMDD_HHMMSS.wav` — Audio recording
- `meeting_YYYYMMDD_HHMMSS.txt` — Transcription text

## Configuration

Settings are automatically saved to `config_funasr.json` and restored on next launch. This includes:
- Input mode, model, language, device selection
- Speaker diarization toggle
- Dark mode preference
- Ollama configuration
- File path (for file input mode)

## Troubleshooting

### FunASR Not Installed
```
pip install funasr modelscope
```

### No WASAPI Devices Found
- Ensure you have active audio playback devices
- Install pyaudiowpatch: `pip install pyaudiowpatch`

### Model Download Fails
- Check internet connection
- Set model mirror: `export MODELSCOPE_CACHE=/path/to/cache`

### MP3/MP4 Input Not Working
- Install ffmpeg and ensure it's in your PATH
- `ffmpeg -version` should work from the command line

### Poor Transcription Quality
- Try a different model (SenseVoiceSmall for multilingual)
- Ensure correct language is selected
- Use CUDA for larger models

### Speaker Diarization Inaccurate
- Use non-streaming models for best results
- Ensure microphone/audio quality is good
- More distinct speakers are easier to separate

## Legacy Version

The original faster-whisper version is still available:
```bash
pip install faster-whisper
python main.py
```
