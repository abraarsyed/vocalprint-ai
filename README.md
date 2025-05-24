# VocalPrint AI

VocalPrint AI is a full-stack tool that detects spoken English accents, scores fluency, and transcribes speech from public video/audio sources.

The current version is CLI-based. A web interface and deployment support will follow in the next phase.

---

## Overview

This tool allows you to:

- Detect English accents such as Indian, British, American, Australian, Canadian
- Get top-3 accent predictions with confidence percentages
- Score English speaking fluency based on speaking time
- Transcribe speech using OpenAI Whisper
- Support input from public YouTube, Loom, or direct MP4 links
- Output results in structured JSON

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the CLI tool

```bash
python3 accent-detection-cli.py \            
  --url "https://www.youtube.com/watch?v=W2Jzkl8J2nM" \ 
  --device cpu
```

### 3. Sample output

```bash
{
  "accent": "canada",
  "accent_confidence": 86.0,
  "top_3_predictions": [
    {
      "accent": "canada",
      "confidence": 86.0
    },
    {
      "accent": "us",
      "confidence": 13.56
    },
    {
      "accent": "england",
      "confidence": 0.21
    }
  ],
  "fluency_score": 100,
  "language_detected_by_whisper": "en",
  "transcript_sample": " you're a mass of competing short term interests. And so the question is then, well, which short term interest should win out? And the answer to that is none of them. They need to be organized into a hierarchy that makes them functional across time and across individuals. So like a two year old is v"
}
```
