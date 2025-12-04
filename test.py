from faster_whisper import WhisperModel
import subprocess
import numpy as np
import sys

path = "/Users/iw/Documents/NTU/1141/whisper/whisper-rt-full/whisper-realtime/台大法學院 36.m4a"

model = WhisperModel("medium")

p = subprocess.Popen(
    ["ffmpeg", "-i", path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL
)

block = 16000 * 3   # 三秒

while True:
    audio = p.stdout.read(block*2)
    if not audio:
        break

    samples = np.frombuffer(audio, np.int16).astype(np.float32) / 32768.0

    segments, info = model.transcribe(samples, language=None)
    for seg in segments:
        sys.stdout.write(seg.text+"\n")
        sys.stdout.flush()