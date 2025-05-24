# web/app.py
# Gradio-based web UI for VocalPrint AI (Refactored to use shared CLI logic)

import gradio as gr
import os
import tempfile
import whisper
import torch
import json
import sys

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.processor import (
    download_video,
    extract_audio,
    transcribe,
    classify_accent,
    compute_fluency
)

# Load Whisper model once
whisper_model = whisper.load_model("small")

def process_video(url):
    try:
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "video.mp4")
        audio_path = os.path.join(temp_dir, "audio.wav")

        download_video(url, temp_dir)
        video_file = next((f for f in os.listdir(temp_dir) if f.endswith(".mp4")), None)
        if not video_file:
            raise FileNotFoundError("No .mp4 file found")

        extract_audio(os.path.join(temp_dir, video_file), audio_path)
        transcript, segments, language = transcribe(audio_path, whisper_model)
        top_accent, confidence, top3 = classify_accent(audio_path)
        fluency = compute_fluency(segments)

        # Format the top3 for the dataframe display
        top3_formatted = [[item["accent"], f"{item['confidence']}%"] for item in top3]

        return (
            top_accent,
            f"{confidence}%",
            fluency,
            language,
            transcript[:500],
            top3_formatted
        )
    except Exception as e:
        return ("Error", "-", "-", "-", str(e), [])

iface = gr.Interface(
    fn=process_video,
    inputs=gr.Textbox(label="Public Video URL (YouTube, Loom, MP4)", placeholder="https://..."),
    outputs=[
        gr.Textbox(label="Detected Accent"),
        gr.Textbox(label="Confidence (%)"),
        gr.Textbox(label="Fluency Score (0–100)"),
        gr.Textbox(label="Language Detected by Whisper"),
        gr.Textbox(label="Transcript Sample (first 500 chars)"),
        gr.Dataframe(headers=["Accent", "Confidence"], label="Top 3 Accent Predictions")
    ],
    title="VocalPrint AI",
    description="Analyze English speech from a public video link to detect accent, fluency, and transcription.",
    allow_flagging="never",
    theme="default"
)

if __name__ == "__main__":
    iface.launch()
