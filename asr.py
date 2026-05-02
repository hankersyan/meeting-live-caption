"""
FunASR-based Audio File Transcription

Transcribe an audio file using FunASR AutoModel.

Usage:
    python asr.py <audio_file_path> [model_name] [device]

Examples:
    python asr.py recording.wav
    python asr.py recording.mp3 paraformer-zh cpu
    python asr.py recording.wav SenseVoiceSmall cuda
"""

import sys
import os

from asr_engine import transcribe_file, FUNASR_AVAILABLE, MODEL_CONFIG


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

    print(f"Transcribing: {file_path}")
    print(f"Model: {model_name} | Device: {device}")
    print("-" * 50)

    results = transcribe_file(
        file_path=file_path,
        model_name=model_name,
        language="en",
        device=device,
    )

    for tr in results:
        if tr.speaker:
            print(f"[{tr.start_ms/1000:6.1f}s - {tr.end_ms/1000:6.1f}s] [{tr.speaker}] {tr.text}")
        else:
            print(tr.text)

    # Save output
    output_file = file_path.rsplit('.', 1)[0] + '.txt'
    transcribe_file(
        file_path=file_path,
        model_name=model_name,
        language="en",
        device=device,
        output_path=output_file,
    )
    print(f"\nTranscription saved to: {output_file}")


if __name__ == "__main__":
    main()
