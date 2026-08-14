#!/usr/bin/env python3
"""Generate 1000 downloads milestone images using Nano Banana 2 (Gemini Flash Image)."""

import os
from pathlib import Path
from google import genai
from google.genai import types

# Load .env manually
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = "gemini-3.1-flash-image-preview"
OUTPUT_DIR = Path("social_posts/1000_milestone")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_image(prompt: str, filename: str, aspect_ratio: str = "1:1"):
    print(f"Generating {filename}...")
    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size="2K",
            ),
        ),
    )
    for part in response.parts:
        if part.inline_data is not None:
            image = part.as_image()
            path = OUTPUT_DIR / filename
            image.save(str(path))
            print(f"  Saved: {path}")
            return
    print(f"  WARNING: No image generated for {filename}")


STYLE_BASE = (
    "Professional podcast promotional graphic. Dark navy/black background with subtle "
    "warm amber and gold accent lighting, evoking a late-night radio studio atmosphere. "
    "Clean modern typography. No photorealistic people. Subtle microphone and radio wave "
    "design elements. Polished, minimal, high contrast."
)

images = [
    {
        "filename": "main_milestone_square.png",
        "aspect_ratio": "1:1",
        "prompt": (
            f"{STYLE_BASE} "
            "Large bold glowing text '1,000' as the hero element in the center, with "
            "'DOWNLOADS' directly below it in a thinner font. Below that in smaller text: "
            "'27 episodes · 200+ callers · 1 month'. "
            "At the top: 'LUKE AT THE ROOST' in elegant lettering. "
            "Subtle golden microphone icon above the title. "
            "At the bottom: 'lukeattheroost.com' in small clean text. "
            "The overall feel is celebratory but classy, like a late-night milestone announcement."
        ),
    },
    {
        "filename": "main_milestone_twitter.png",
        "aspect_ratio": "16:9",
        "prompt": (
            f"{STYLE_BASE} "
            "Wide banner format. Left side has 'LUKE AT THE ROOST' title with a subtle "
            "microphone graphic. Right side has large bold glowing '1,000 DOWNLOADS' text "
            "with '27 episodes · 200+ callers · 1 month' below. "
            "Warm amber glow connecting the two sides. "
            "Bottom right corner: 'lukeattheroost.com'. "
            "Designed as a Twitter/X post image."
        ),
    },
    {
        "filename": "carousel_1_downloads.png",
        "aspect_ratio": "1:1",
        "prompt": (
            f"{STYLE_BASE} "
            "Instagram carousel slide 1. Giant bold text '1,000' taking up most of the frame, "
            "with 'DOWNLOADS' below. Subtle radio wave ripples emanating from the numbers. "
            "'LUKE AT THE ROOST' at the top in small elegant text. "
            "Small '1/5' page indicator dots at the bottom."
        ),
    },
    {
        "filename": "carousel_2_episodes.png",
        "aspect_ratio": "1:1",
        "prompt": (
            f"{STYLE_BASE} "
            "Instagram carousel slide 2. Large bold text '27' in the center with "
            "'EPISODES' below. A subtle audio waveform timeline graphic running horizontally "
            "behind the number. 'LUKE AT THE ROOST' at the top in small elegant text. "
            "Small '2/5' page indicator dots at the bottom."
        ),
    },
    {
        "filename": "carousel_3_callers.png",
        "aspect_ratio": "1:1",
        "prompt": (
            f"{STYLE_BASE} "
            "Instagram carousel slide 3. Large bold text '200+' in the center with "
            "'CALLERS' below. Subtle vintage telephone handset icon above the number. "
            "'LUKE AT THE ROOST' at the top in small elegant text. "
            "Small '3/5' page indicator dots at the bottom."
        ),
    },
    {
        "filename": "carousel_4_regulars.png",
        "aspect_ratio": "1:1",
        "prompt": (
            f"{STYLE_BASE} "
            "Instagram carousel slide 4. Large bold text '13' in the center with "
            "'RETURNING REGULARS' below. Subtle connected dots/nodes graphic suggesting "
            "a network of recurring characters. "
            "'LUKE AT THE ROOST' at the top in small elegant text. "
            "Small '4/5' page indicator dots at the bottom."
        ),
    },
    {
        "filename": "carousel_5_thankyou.png",
        "aspect_ratio": "1:1",
        "prompt": (
            f"{STYLE_BASE} "
            "Instagram carousel slide 5. Warm, heartfelt tone. Large elegant text "
            "'THANK YOU' in the center with a soft golden glow. Below: "
            "'lukeattheroost.com' and '208-439-LUKE' in clean small text. "
            "'LUKE AT THE ROOST' at the top in small elegant text. "
            "Small '5/5' page indicator dots at the bottom. "
            "Slightly warmer color temperature than the other slides to feel like a closing moment."
        ),
    },
]

if __name__ == "__main__":
    for img in images:
        try:
            generate_image(img["prompt"], img["filename"], img["aspect_ratio"])
        except Exception as e:
            print(f"  ERROR generating {img['filename']}: {e}")
    print(f"\nDone! Images saved to {OUTPUT_DIR}/")
