"""
ASR with Speaker Diarization using FunASR + cam++

Reads an audio/video file, transcribes with FunASR, runs speaker
diarization via cam++ model, and prints colorized results.

Usage:
    python asr_speaker_diarization.py path/to/your_audio.wav [model_name] [device]

Examples:
    python asr_speaker_diarization.py meeting.wav
    python asr_speaker_diarization.py meeting.mp4 paraformer-zh cpu
    python asr_speaker_diarization.py meeting.mp3 SenseVoiceSmall cuda

Requirements:
    pip install funasr modelscope numpy soundfile
    ffmpeg must be installed for MP3/MP4 input
"""

import sys
import os

from asr_engine import transcribe_file, FUNASR_AVAILABLE, MODEL_CONFIG


# ANSI color codes for up to 8 speakers
SPEAKER_ANSI_COLORS = [
    "\033[94m",  # Blue
    "\033[92m",  # Green
    "\033[93m",  # Yellow
    "\033[91m",  # Red
    "\033[95m",  # Magenta
    "\033[96m",  # Cyan
    "\033[97m",  # White
    "\033[90m",  # Gray
]
RESET = "\033[0m"


def print_results(results):
    """Print speaker-labeled transcription to console with color."""
    print("=" * 60)
    print("Speaker-labeled Transcription (FunASR + cam++)")
    print("=" * 60)

    for tr in results:
        if tr.speaker:
            spk_idx = 0
            if tr.speaker.startswith("SPEAKER_"):
                try:
                    spk_idx = int(tr.speaker.split("_")[1])
                except ValueError:
                    pass
            color = SPEAKER_ANSI_COLORS[spk_idx % len(SPEAKER_ANSI_COLORS)]
            label = f"SPK {spk_idx}"
            timestamp = f"{tr.start_ms/1000:6.1f}s - {tr.end_ms/1000:6.1f}s"
            print(f"{color}[{timestamp}] [{label}]{RESET} {tr.text}")
        else:
            print(tr.text)

    # Per-speaker summary
    print("\n" + "=" * 60)
    speaker_texts = {}
    for tr in results:
        speaker = tr.speaker or "UNKNOWN"
        speaker_texts.setdefault(speaker, []).append(tr.text)

    for speaker, texts in speaker_texts.items():
        full = " ".join(texts)
        print(f"\n{speaker}:")
        print(f"  {full}")


def main():
    if not FUNASR_AVAILABLE:
        print("Error: FunASR is not installed. Run: pip install funasr")
        sys.exit(1)

    if len(sys.argv) < 2:
        print(__doc__)
        print(f"\nAvailable models: {', '.join(MODEL_CONFIG.keys())}")
        print("Device options: cpu, cuda")
        sys.exit(1)

    file_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "iic/SenseVoiceSmall"
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"

    if not os.path.isfile(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Input: {file_path}")
    print(f"Model: {model_name} | Device: {device}")
    print(f"Speaker diarization: cam++")
    print()

    # Transcribe with speaker diarization
    results = transcribe_file(
        file_path=file_path,
        model_name=model_name,
        language="en",
        device=device,
    )

    if not results:
        print("No transcription results.")
        sys.exit(0)

    # Print results
    print_results(results)

    # Save to file
    output_dir = os.path.dirname(os.path.abspath(file_path))
    base = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base}_labeled.txt")

    transcribe_file(
        file_path=file_path,
        model_name=model_name,
        language="en",
        device=device,
        output_path=output_path,
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
