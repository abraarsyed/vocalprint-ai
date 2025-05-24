"""
Accent & Fluency Detection CLI Tool
-----------------------------------
Detects English accents from audio extracted from public video URLs
and calculates transcript, fluency score, and top-3 accent confidence scores.

Model Used: dima806/english_accents_classification
"""

import os
import subprocess
import requests
import whisper
import torchaudio
import logging
import argparse
import tempfile
import shutil
import json
import torch
import yt_dlp
import soundfile as sf

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from scipy.special import softmax

# ---------------- Logging Setup ----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("AccentCLI")

# ---------------- Load Accent Classifier ----------------
MODEL_ID = "dima806/english_accents_classification"
accent_model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
labels = list(accent_model.config.id2label.values())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accent_model.to(device)

# ---------------- Helper Functions ----------------

def download_video(url, output_dir):
    logger.info(f"Downloading video from: {url}")
    try:
        if any(x in url for x in ["youtube.com", "youtu.be", "loom.com"]):
            ydl_opts = {
                'format': 'bestvideo+bestaudio/best',
                'merge_output_format': 'mp4',
                'outtmpl': os.path.join(output_dir, 'input_video.%(ext)s'),
                'quiet': True,
                'no_warnings': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        else:
            response = requests.get(url, stream=True, timeout=20)
            response.raise_for_status()
            filepath = os.path.join(output_dir, "input_video.mp4")
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        logger.error(f"Failed to download video: {e}")
        raise

def extract_audio(video_path, audio_path):
    logger.info("Extracting audio from video...")
    subprocess.run([
        'ffmpeg', '-y', '-i', video_path,
        '-ss', '00:00:15', '-t', '00:00:30',
        '-ar', '16000', '-ac', '1',
        '-loglevel', 'error', audio_path
    ], check=True)

def transcribe(audio_path, whisper_model):
    logger.info("Transcribing with Whisper...")
    result = whisper_model.transcribe(audio_path)
    return result["text"], result["segments"], result["language"]

def classify_accent(audio_path):
    logger.info("Running accent classification...")
    waveform, sample_rate = sf.read(audio_path)
    inputs = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = accent_model(**inputs).logits
    probs = softmax(logits[0].cpu().numpy())
    top_indices = probs.argsort()[::-1][:3]
    top_accents = [{"accent": labels[i], "confidence": round(float(probs[i]) * 100, 2)} for i in top_indices]
    return top_accents[0]["accent"], top_accents[0]["confidence"], top_accents

def compute_fluency(segments):
    if not segments:
        return 0
    total_time = segments[-1]['end']
    speaking_time = sum(seg['end'] - seg['start'] for seg in segments)
    return int(min(speaking_time / total_time * 100, 100))

# ---------------- Main CLI Logic ----------------

def main():
    parser = argparse.ArgumentParser(description="Accent & Fluency Detection CLI")
    parser.add_argument('--url', required=True, help='Public video URL (YouTube, Loom, MP4)')
    parser.add_argument('--output', help='Output path for JSON result')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to run Whisper')
    parser.add_argument('--keep', action='store_true', help='Keep temporary files')
    args = parser.parse_args()

    whisper_device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    whisper_model = whisper.load_model("small", device=whisper_device)

    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "audio.wav")

    try:
        download_video(args.url, temp_dir)
        video_file = next((f for f in os.listdir(temp_dir) if f.endswith(".mp4")), None)
        if not video_file:
            raise FileNotFoundError("No .mp4 file found in temp dir")
        extract_audio(os.path.join(temp_dir, video_file), audio_path)
        transcript, segments, language = transcribe(audio_path, whisper_model)
        top_accent, confidence, top3 = classify_accent(audio_path)
        fluency = compute_fluency(segments)

        result = {
            "accent": top_accent,
            "accent_confidence": confidence,
            "top_3_predictions": top3,
            "fluency_score": fluency,
            "language_detected_by_whisper": language,
            "transcript_sample": transcript[:300]
        }

        print(json.dumps(result, indent=2))
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved output to {args.output}")
    except Exception as e:
        logger.error(f"FAILED: {e}")
    finally:
        if args.keep:
            logger.info(f"Temporary files kept in: {temp_dir}")
        else:
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()