import os
import subprocess
import requests
import torch
import yt_dlp
import soundfile as sf

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from scipy.special import softmax

from core.logger import logger


MODEL_ID = "dima806/english_accents_classification"
accent_model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
labels = list(accent_model.config.id2label.values())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accent_model.to(device)


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
