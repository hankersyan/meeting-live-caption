import queue
import sys
import wave
import tempfile
import os
import time
import threading

import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
import torch
from pyannote.audio import Pipeline

# ---------- Configuration ----------
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000          # 0.5 second chunks
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
DIARIZATION_WINDOW = 5.0   # seconds – analyse last N seconds every DIARIZATION_INTERVAL
DIARIZATION_INTERVAL = 2.0 # how often we run diarization (seconds)
HF_AUTH_TOKEN = ""  # required for pyannote

# ---------- Load models ----------
print("Loading Vosk model...")
vosk_model = Model(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
recognizer.SetWords(True)   # enable word timestamps

print("Loading diarization pipeline...")
diarization_pipe = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_AUTH_TOKEN
)
if torch.cuda.is_available():
    diarization_pipe.to(torch.device("cuda"))

# ---------- Global state ----------
audio_buffer = np.empty((0,), dtype=np.int16)
buffer_lock = threading.Lock()
word_results = []   # list of (word, start_time, end_time, speaker)
diarization_runs = 0
stop_event = threading.Event()

# ---------- Audio callback ----------
def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for each audio block."""
    global audio_buffer
    if status:
        print("Audio status:", status, file=sys.stderr)
    with buffer_lock:
        audio_buffer = np.append(audio_buffer, indata[:, 0])  # mono channel

# ---------- Diarization thread ----------
def diarization_worker():
    global word_results, audio_buffer, diarization_runs
    while not stop_event.is_set():
        time.sleep(DIARIZATION_INTERVAL)

        # Copy the required window of audio
        with buffer_lock:
            window_samples = int(DIARIZATION_WINDOW * SAMPLE_RATE)
            if len(audio_buffer) < window_samples:
                continue
            # take the last DIARIZATION_WINDOW seconds
            window = audio_buffer[-window_samples:].copy()
            # timestamp of the beginning of this window
            window_start_time = (len(audio_buffer) - window_samples) / SAMPLE_RATE

        # Write to a temporary WAV file (pyannote expects a file)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)   # 16-bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(window.tobytes())

        print(f"[Diarization] processing WAV {wav_path}")
        # Run diarization
        diarization = diarization_pipe(wav_path)
        os.unlink(wav_path)  # clean up

        # Update word_results with speaker labels
        # We will annotate all words that fall inside the window
        new_labels = []  # (start, end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # shift turn times to absolute stream time
            abs_start = window_start_time + turn.start
            abs_end = window_start_time + turn.end
            new_labels.append((abs_start, abs_end, speaker))

        # Lock and update word_results
        with buffer_lock:
            for w in word_results:
                # Only re-label words that fall inside our window
                if w[1] >= window_start_time and w[2] <= window_start_time + DIARIZATION_WINDOW:
                    # find speaker at word midpoint
                    midpoint = (w[1] + w[2]) / 2
                    for s_start, s_end, spk in new_labels:
                        if s_start <= midpoint <= s_end:
                            w[3] = spk  # tuple is immutable? We'll use list or mutable containers
                            break

        diarization_runs += 1
        # print just for demonstration
        print(f"\n[Diarization #{diarization_runs}] found {len(new_labels)} speaker segments.")

# ---------- Main recognition loop ----------
def main():
    global word_results

    print("Starting microphone stream...")
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype='int16',
        callback=audio_callback
    )

    # Start diarization thread
    diar_thread = threading.Thread(target=diarization_worker, daemon=True)
    diar_thread.start()

    # Buffer to hold incomplete audio for Vosk
    vosk_queue = queue.Queue()

    def vosk_feeder():
        """Move received audio chunks into Vosk."""
        while not stop_event.is_set():
            data = vosk_queue.get()
            if data is None:
                break
            if recognizer.AcceptWaveform(data):
                res = recognizer.Result()
                # Process final result
                process_result(eval(res))
            else:
                # Partial result (optional)
                partial = recognizer.PartialResult()
                # Only print partial without speaker (not yet labeled)
                # print(eval(partial).get('partial', ''))

    feeder_thread = threading.Thread(target=vosk_feeder, daemon=True)
    feeder_thread.start()

    # Read from the audio buffer and push to Vosk queue
    stream.start()
    read_offset = 0
    try:
        while True:
            time.sleep(0.1)  # small sleep to prevent busy-wait
            with buffer_lock:
                if len(audio_buffer) - read_offset >= BLOCK_SIZE:
                    chunk = audio_buffer[read_offset:read_offset+BLOCK_SIZE]
                    read_offset += BLOCK_SIZE
                    vosk_queue.put(chunk.tobytes())
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        stream.stop()
        stream.close()
        vosk_queue.put(None)
        feeder_thread.join()
        diar_thread.join()
        # Print final labeled transcript
        print("\n=== Final transcript with speakers ===")
        for w in word_results:
            word, start, end, speaker = w
            print(f"[{speaker or '?'}] {word} ({start:.2f}-{end:.2f})")

def process_result(result_dict):
    """Vosk final result containing 'result' key with word list."""
    global word_results
    words = result_dict.get('result', [])
    for w in words:
        word = w['word']
        start = w['start']
        end = w['end']
        # Append a mutable list so we can later assign speaker
        word_results.append([word, start, end, None])  # speaker unknown

if __name__ == "__main__":
    main()