import sys
from faster_whisper import WhisperModel

def transcribe_audio(file_path, model_size="large-v3", device="cuda", compute_type="float16"):
    """
    Transcribe an audio file using Whisper model
    
    Args:
        file_path (str): Path to the audio file to transcribe
        model_size (str): Size of the Whisper model to use
        device (str): Device to run the model on ("cuda" or "cpu")
        compute_type (str): Compute type ("float16", "int8", etc.)
    """
    # Initialize the model
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Perform transcription
    segments, info = model.transcribe(file_path, beam_size=5)
    segments = list(segments)

    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    # Print the transcribed text
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    
    # Create output text file name
    output_file = file_path.rsplit('.', 1)[0] + '.txt'
    
    # Write transcription to text file
    with open(output_file, 'w', encoding='utf-8') as f:
        for segment in segments:
            f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")
    
    print(f"\nTranscription saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_wav_txt.py <audio_file_path> [model_size] [device] [compute_type]")
        print("Example: python convert_wav_txt.py recording.wav large-v3 cuda float16")
        print("Example: python convert_wav_txt.py recording.wav medium cpu int8")
        print("\nAvailable model sizes: tiny, base, small, medium, large-v1, large-v2, large-v3")
        print("Device options: cpu, cuda")
        print("Compute types: float16, int8, int8_float16")
        sys.exit(1)
    
    file_path = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "large-v3"
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    compute_type = sys.argv[4] if len(sys.argv) > 4 else "float16"
    
    transcribe_audio(file_path, model_size, device, compute_type)