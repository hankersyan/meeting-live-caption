# Meeting Live Caption - Usage Guide

## Getting Started

### Prerequisites

Before using the application, ensure you have:
- Windows operating system (required for WASAPI loopback)
- Python 3.7 or higher installed
- At least 2GB of free disk space for model downloads
- A working audio system with at least one playback device

### Installation Steps

1. **Download the Application**
   - Clone the repository or download the source files
   - Ensure you have the `main.py` file and any dependencies

2. **Install Python Dependencies**
   ```bash
   pip install faster-whisper pyaudiowpatch numpy
   ```

3. **First-Time Setup**
   - The first time you run the application, Whisper models will be downloaded automatically
   - Download size varies by model (50MB for tiny, up to 3GB for large models)
   - Internet connection required for initial model download

4. **Launch the Application**
   ```bash
   python main.py
   ```

## Interface Walkthrough

### Main Window Components

1. **Control Panel (Top Section)**
   - **Audio Device Dropdown**: Select the playback device to record from
   - **Refresh Button**: Reload available audio devices
   - **Model Selection**: Choose Whisper model size
   - **Language Selection**: Set transcription language

2. **Options Panel**
   - **Save Audio & Text Checkbox**: Enable/disable file saving
   - **File Label**: Shows current file status
   - **Open Records Folder Button**: Opens the output directory

3. **Key Point Extraction (Ollama)**
   - **Enable**: Turn periodic extraction on/off
   - **URL**: Ollama API base URL (default: `http://localhost:11434`)
   - **Model**: Ollama model name to use for extraction
   - **Refresh(s)**: Interval in seconds between extraction runs
   - **Prompt**: Instruction used to generate brief key points

4. **Action Buttons**
   - **Start Recording**: Begin audio capture and transcription
   - **Stop Recording**: End recording session
   - **Clear Text**: Clear the transcription display

5. **Status Bar**
   - Shows current application state and messages

6. **Transcription Display**
   - Scrollable text area showing live transcription

7. **Brief Key Points Display**
   - Scrollable text area showing timestamped key-point summaries

## Detailed Usage Instructions

### Step 1: Select Audio Device
1. Click "Refresh" to populate the device list
2. Select the audio device you want to record (typically your default playback device)
3. The status bar will confirm your selection

### Step 2: Configure Settings
1. **Model Selection**: Choose based on your needs:
   - Tiny: Fastest, lower accuracy
   - Base: Good balance of speed and quality
   - Small: Better quality, slower
   - Medium/Large: Highest quality, slowest
2. **Language Selection**: 
   - Choose specific language for better accuracy
   - Select "auto" for automatic language detection

### Step 3: Configure Output
1. Check "Save audio & text" if you want to store recordings
2. Files will be saved in the 'records' folder with timestamped names
3. Click "Open Records Folder" to access saved files

### Step 4: Configure Key Point Extraction
1. Enable key-point extraction if you want live summaries
2. Set Ollama URL (local default: `http://localhost:11434`)
3. Set model name (example: `LiquidAI/lfm2.5-1.2b-instruct`)
4. Set refresh interval in seconds (example: `20`)
5. Customize the extraction prompt if needed

### Step 5: Start Recording
1. Click "Start Recording"
2. The application will begin capturing audio and transcribing speech
3. Watch the status bar for progress indicators
4. Live transcription appears in the text area
5. Brief key points are updated periodically in the key-points area

### Step 6: Stop and Review
1. Click "Stop Recording" when finished
2. Check the 'records' folder for saved files
3. Review both audio (WAV) and text (TXT) outputs

## Advanced Usage Tips

### Model Selection Guidelines
- **For meetings with clear speech**: Use 'base' or 'small'
- **For multiple speakers or noisy environments**: Use 'medium' or 'large'
- **For real-time applications**: Use 'tiny' or 'base'
- **For high accuracy**: Use 'large-v2' or 'large-v3'

### Language Settings
- Specific language codes improve accuracy
- Supported languages include: en, zh, es, fr, de, ja, ko, and many others
- Auto-detection works well but may be slower

### Performance Optimization
- Close other CPU-intensive applications
- Use smaller models for older systems
- Ensure adequate RAM (8GB+ recommended)
- SSD storage improves file I/O performance

## File Organization

The application creates the following in the 'records' folder:
```
records/
├── meeting_20231201_143022.wav    # Audio recording
├── meeting_20231201_143022.txt    # Transcription
└── meeting_20231201_154511.wav    # Next recording...
```

Each recording session creates a paired WAV and TXT file with identical timestamps.

## Troubleshooting Common Issues

### No Audio Devices Found
- Check that your system has active audio playback devices
- Ensure audio drivers are up to date
- Restart the application after connecting new audio devices

### Poor Transcription Quality
- Try a larger Whisper model
- Ensure the correct language is selected
- Reduce background noise during recording
- Speak clearly and at moderate pace

### Slow Performance
- Close other applications to free up system resources
- Use a smaller Whisper model
- Check that your system meets minimum requirements

### Missing Output Files
- Verify that "Save audio & text" checkbox is enabled
- Check that the 'records' folder has write permissions
- Ensure sufficient disk space is available

### Application Freezes
- Force close the application and restart
- Update audio drivers
- Check for conflicting audio applications

## Best Practices

### For Meetings
- Position microphones appropriately if using microphone input
- Test the setup before important meetings
- Keep the application window visible during recording
- Plan for adequate storage space for longer sessions

### For Presentations
- Use higher quality models for important presentations
- Test the audio setup beforehand
- Have backup recording methods available

### File Management
- Regularly archive old recordings
- Organize recordings by date or topic
- Back up important transcriptions separately

## Keyboard Shortcuts

While the GUI doesn't currently have keyboard shortcuts, you can add them by modifying the source code to bind key events to the start/stop/clear functions.