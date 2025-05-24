import argparse
import json
import os
import tempfile
import shutil

import whisper
import torch

from core.processor import (
    download_video,
    extract_audio,
    transcribe,
    classify_accent,
    compute_fluency
)
from core.logger import logger


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