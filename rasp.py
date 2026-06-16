import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import sounddevice as sd
import librosa
import numpy as np
import os
import soundfile as sf
from datetime import datetime
from collections import deque

DEVICE = torch.device('cpu')
MODEL_PATH = "models/close_beeband_mobilenet_model.pth"
SAVE_DIR = "recorded_audio"
os.makedirs(SAVE_DIR, exist_ok=True)
LABEL_MAP = {0: 'Active', 1: 'Queenless', 2: 'Infested'}
SR = 16000
CHUNK_S = 5
WINDOW_S = 30
WINDOW_FRAMES = WINDOW_S * SR
RMS_THRESHOLD = 0.003
TARGET_RMS = 0.01
CONFIDENCE_THRESHOLD = 0.60

print("Loading model...")
model = timm.create_model('mobilenetv4_hybrid_large', pretrained=False, num_classes=0)
model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, 3))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully!")

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1)

def preprocess(audio_chunk):
    audio_chunk = audio_chunk.flatten()
    rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
    if rms > 0:
        audio_chunk = audio_chunk * (TARGET_RMS / rms)
    mel = librosa.feature.melspectrogram(y=audio_chunk, sr=SR, n_fft=1024, hop_length=512, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=1.0)
    tensor = torch.tensor(mel_db, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    resized = F.interpolate(tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
    rgb = resized.repeat(3, 1, 1)
    return (rgb - MEAN) / STD, rms

audio_buffer = deque(maxlen=WINDOW_FRAMES)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_buffer.extend(indata.flatten())

    if len(audio_buffer) < WINDOW_FRAMES:
        return

    chunk = np.array(list(audio_buffer)[:WINDOW_FRAMES])

    input_tensor, rms = preprocess(chunk)
    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).squeeze(0)
        pred = torch.argmax(probs).item()
        confidence = probs[pred].item()

    status = "LOW_AUDIO" if rms < RMS_THRESHOLD else "OK"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(SAVE_DIR, f"rec_{timestamp}.wav")
    sf.write(filename, list(audio_buffer)[:CHUNK_S * SR], SR)

    probs_str = " | ".join(f"{LABEL_MAP[i]}: {probs[i]:.3f}" for i in range(3))

    if confidence < CONFIDENCE_THRESHOLD:
        top2 = np.argsort(probs.numpy())[-2:][::-1]
        guess = f"{LABEL_MAP[pred]}?"
        print(f"[{timestamp}] {guess} ({confidence:.1%}) | rms={rms:.4f} [{status}] | {probs_str}  <-- LOW CONFIDENCE (maybe {LABEL_MAP[top2[1]]}?)")
    else:
        print(f"[{timestamp}] {LABEL_MAP[pred]} ({confidence:.1%}) | rms={rms:.4f} [{status}] | {probs_str}")

print(f"Listening... (classifying every {CHUNK_S}s on a rolling {WINDOW_S}s window)")
with sd.InputStream(callback=audio_callback, channels=1, samplerate=SR, blocksize=SR * CHUNK_S):
    while True:
        sd.sleep(100)
