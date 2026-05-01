"""
Simple Speaker Diarization Sample
- Reads an MP4 file
- Extracts audio to WAV (16kHz mono)
- Transcribes with faster-whisper
- Runs pyannote.audio speaker diarization
- Maps speaker labels to transcribed segments
- Prints colorized results

Usage:
    python speaker_diarization_sample.py path/to/your_video.mp4

Requirements:
    pip install faster-whisper pyannote.audio torch numpy
    ffmpeg must be installed and available in PATH
"""

import sys
import os
import subprocess
import tempfile
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Suppress noisy torchcodec warning from pyannote.audio
import warnings
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline


# ------------------------------
# Data Structures
# ------------------------------

@dataclass
class TranscriptionSegment:
    """A single transcribed segment with timing and optional speaker label."""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


# ------------------------------
# Step 1: Extract audio from MP4
# ------------------------------

def extract_audio(mp4_path: str, output_wav: str, target_sr: int = 16000) -> str:
    """Use ffmpeg to extract mono 16kHz audio from an MP4 file."""
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"Input file not found: {mp4_path}")

    cmd = [
        "ffmpeg", "-y",
        "-i", mp4_path,
        "-vn",                       # no video
        "-acodec", "pcm_s16le",      # PCM 16-bit
        "-ar", str(target_sr),       # sample rate
        "-ac", "1",                  # mono
        output_wav,
    ]
    print(f"Extracting audio from: {mp4_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    print(f"Audio extracted to: {output_wav}")
    return output_wav


# ------------------------------
# Step 2: Transcribe with Whisper
# ------------------------------

def transcribe(wav_path: str, model_size: str = "base", language: str = None, device: str = "cpu") -> List[TranscriptionSegment]:
    """Run faster-whisper transcription on a WAV file."""
    print(f"\nLoading Whisper model '{model_size}' (device={device})...")
    model = WhisperModel(model_size, device=device, compute_type="int8", cpu_threads=4)

    print("Transcribing...")
    segments, info = model.transcribe(
        wav_path,
        language=language,
        beam_size=5,
        best_of=5,
        temperature=0.0,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    results: List[TranscriptionSegment] = []
    for seg in segments:
        results.append(TranscriptionSegment(
            start=round(seg.start, 2),
            end=round(seg.end, 2),
            text=seg.text.strip(),
        ))

    print(f"  Detected language: {info.language} (probability={info.language_probability:.2f})")
    print(f"  Transcribed {len(results)} segments\n")
    return results


# ------------------------------
# Step 3: Speaker Diarization
# ------------------------------

def diarize(wav_path: str, hf_token: str, device: str = "cpu", num_speakers: Optional[int] = None) -> List[Tuple[float, float, str]]:
    """Run pyannote.audio speaker diarization on a WAV file.

    Uses in-memory waveform loading to avoid pyannote's built-in AudioDecoder bug.
    """
    import torch
    from scipy.io import wavfile

    print(f"Loading pyannote speaker-diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    if device == "cuda":
        pipeline.to(torch.device("cuda"))

    # Read WAV file into memory and pass as waveform dict
    # (bypasses pyannote's broken AudioDecoder)
    print(f"Reading WAV file: {wav_path}")
    sample_rate, waveform = wavfile.read(wav_path)

    # Convert to float32 torch tensor, shape (1, num_samples)
    if waveform.dtype == np.int16:
        waveform_float = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype == np.int32:
        waveform_float = waveform.astype(np.float32) / 2147483648.0
    else:
        waveform_float = waveform.astype(np.float32)

    # Ensure mono: if stereo, average channels
    if waveform_float.ndim == 2:
        waveform_float = waveform_float.mean(axis=1)

    waveform_torch = torch.from_numpy(waveform_float).unsqueeze(0)  # (1, T)

    print(f"Running diarization (audio: {waveform_torch.shape[-1] / sample_rate:.1f}s, {sample_rate}Hz)...")
    output = pipeline({"waveform": waveform_torch, "sample_rate": sample_rate})

    # DiarizeOutput is a dataclass with fields:
    #   speaker_diarization: Annotation
    #   exclusive_speaker_diarization: Annotation
    #   speaker_embeddings: np.ndarray | None
    annotation = output.speaker_diarization

    # --- DEBUG: inspect annotation ---
    print(f"\n  [DEBUG] annotation type: {type(annotation)}")
    print(f"  [DEBUG] labels: {annotation.labels()}")

    results: List[Tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        results.append((round(turn.start, 2), round(turn.end, 2), speaker))

    if not results:
        print("  [DEBUG] yield_label returned nothing, trying without yield_label...")
        for turn, track in annotation.itertracks(yield_label=False):
            labels = annotation.get_labels(track)
            speaker = labels[0] if labels else "SPEAKER_UNKNOWN"
            results.append((round(turn.start, 2), round(turn.end, 2), speaker))

    print(f"\n  [DEBUG] Raw diarization segments ({len(results)}):")
    for r in results[:10]:  # show first 10
        print(f"    [{r[0]:6.1f}s - {r[1]:6.1f}s] {r[2]}")
    if len(results) > 10:
        print(f"    ... and {len(results) - 10} more")

    unique_speakers = set(s for _, _, s in results)
    print(f"  Detected {len(unique_speakers)} speaker(s): {sorted(unique_speakers)}")
    print(f"  Total diarization segments: {len(results)}\n")
    return results


# ------------------------------
# Step 4: Map speakers to transcription
# ------------------------------

def map_speakers(
    segments: List[TranscriptionSegment],
    diarization_results: List[Tuple[float, float, str]],
    time_eps: float = 0.3,
) -> List[TranscriptionSegment]:
    """Map speaker labels to transcription segments based on timestamp overlap."""
    labeled = [TranscriptionSegment(s.start, s.end, s.text, None) for s in segments]

    for seg in labeled:
        best_speaker = None
        max_overlap = 0.0
        for d_start, d_end, speaker in diarization_results:
            overlap = max(0.0, min(seg.end, d_end) - max(seg.start, d_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = speaker

        if max_overlap >= time_eps and best_speaker is not None:
            seg.speaker = best_speaker

    return labeled


# ------------------------------
# Step 5: Output utilities
# ------------------------------

def print_results(segments: List[TranscriptionSegment]):
    """Print speaker-labeled transcription to console with color."""
    # ANSI color codes for up to 8 speakers
    colors = [
        "\033[94m",  # blue
        "\033[92m",  # green
        "\033[93m",  # yellow
        "\033[91m",  # red
        "\033[95m",  # magenta
        "\033[96m",  # cyan
        "\033[97m",  # white
        "\033[90m",  # gray
    ]
    reset = "\033[0m"

    print("=" * 60)
    print("Speaker-labeled Transcription")
    print("=" * 60)

    for seg in segments:
        if seg.speaker:
            speaker_num = int(seg.speaker.split("_")[-1]) if seg.speaker.startswith("SPEAKER_") else 0
            color = colors[speaker_num % len(colors)]
            label = f"SPEAKER {speaker_num}"
        else:
            color = "\033[90m"
            label = "UNKNOWN"

        timestamp = f"{seg.start:6.1f}s - {seg.end:6.1f}s"
        print(f"{color}[{timestamp}] [{label}]{reset} {seg.text}")

    # Also print a summary
    print("\n" + "=" * 60)
    speaker_texts: dict = {}
    for seg in segments:
        speaker = seg.speaker or "UNKNOWN"
        if speaker not in speaker_texts:
            speaker_texts[speaker] = []
        speaker_texts[speaker].append(seg.text)

    for speaker, texts in speaker_texts.items():
        full = " ".join(texts)
        print(f"\n{speaker}:")
        print(f"  {full}")


def save_results(segments: List[TranscriptionSegment], output_path: str):
    """Save speaker-labeled transcription to a text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Speaker-labeled Transcription\n")
        f.write("=" * 50 + "\n\n")
        for seg in segments:
            speaker = seg.speaker if seg.speaker else "UNKNOWN"
            f.write(f"[{seg.start:6.1f}s - {seg.end:6.1f}s] [{speaker}] {seg.text}\n")

        # Summary per speaker
        f.write("\n\n" + "=" * 50 + "\nPer-Speaker Summary\n" + "=" * 50 + "\n")
        speaker_texts: dict = {}
        for seg in segments:
            s = seg.speaker or "UNKNOWN"
            speaker_texts.setdefault(s, []).append(seg.text)
        for speaker, texts in speaker_texts.items():
            f.write(f"\n{speaker}:\n  {' '.join(texts)}\n")

    print(f"\nResults saved to: {output_path}")


# ------------------------------
# Main
# ------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mp4_path = sys.argv[1]
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("=" * 60)
        print("WARNING: No HF_TOKEN environment variable found.")
        print("pyannote.audio requires a HuggingFace token.")
        print("Set it with:  set HF_TOKEN=hf_xxxxx  (Windows)")
        print("         or:  export HF_TOKEN=hf_xxxxx  (Linux/Mac)")
        print("=" * 60)

    # Optional: override via command-line
    if len(sys.argv) >= 3:
        hf_token = sys.argv[2]

    # --- Step 1: Extract audio ---
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        extract_audio(mp4_path, wav_path)

        # --- Step 2: Transcribe ---
        segments = transcribe(wav_path, model_size="base", device="cpu")

        # Print raw transcription first
        print("Raw Transcription (no speaker labels):")
        for seg in segments:
            print(f"  [{seg.start:6.1f}s - {seg.end:6.1f}s] {seg.text}")
        print()

        # --- Step 3: Diarization ---
        if hf_token:
            diarization_results = diarize(wav_path, hf_token, device="cpu")

            # --- Step 4: Map speakers to transcription ---
            labeled = map_speakers(segments, diarization_results)

            # --- DEBUG: mapping diagnostics ---
            labeled_count = sum(1 for s in labeled if s.speaker is not None)
            print(f"\n  [DEBUG] Mapping: {labeled_count}/{len(labeled)} segments got speaker labels")
            if labeled_count == 0 and diarization_results:
                # Try again with a more permissive overlap threshold
                print("  [DEBUG] Retrying with time_eps=0.1 ...")
                labeled = map_speakers(segments, diarization_results, time_eps=0.1)
                labeled_count = sum(1 for s in labeled if s.speaker is not None)
                print(f"  [DEBUG] After retry: {labeled_count}/{len(labeled)} segments labeled")
                if labeled_count == 0:
                    print("  [DEBUG] Still no labels. Assigning speakers by nearest diarization segment...")
                    # Fallback: assign each segment the speaker from the closest diarization segment
                    for seg in labeled:
                        best_speaker = None
                        best_gap = float("inf")
                        seg_mid = (seg.start + seg.end) / 2
                        for d_start, d_end, speaker in diarization_results:
                            d_mid = (d_start + d_end) / 2
                            gap = abs(seg_mid - d_mid)
                            if gap < best_gap:
                                best_gap = gap
                                best_speaker = speaker
                        if seg_mid >= d_start and seg_mid <= d_end:
                            seg.speaker = best_speaker  # only assign if the segment midpoint falls within a diarization window
                            labeled_count += 1
                    print(f"  [DEBUG] After fallback: {sum(1 for s in labeled if s.speaker is not None)}/{len(labeled)} segments labeled")

            # --- Step 5: Output ---
            print_results(labeled)

            # Save to file
            output_dir = os.path.dirname(os.path.abspath(mp4_path))
            base = os.path.splitext(os.path.basename(mp4_path))[0]
            output_path = os.path.join(output_dir, f"{base}_labeled.txt")
            save_results(labeled, output_path)
        else:
            print("Skipping diarization (no HF_TOKEN).")

    finally:
        # Clean up temp WAV
        if os.path.exists(wav_path):
            os.remove(wav_path)
            print(f"\nCleaned up temp file: {wav_path}")


if __name__ == "__main__":
    main()
