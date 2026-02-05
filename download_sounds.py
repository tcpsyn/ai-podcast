#!/usr/bin/env python3
"""
Download free sound effects for the radio show soundboard.
Uses sounds from freesound.org and other free sources.
"""

import os
import urllib.request
import ssl
from pathlib import Path

# Bypass SSL issues
ssl._create_default_https_context = ssl._create_unverified_context

SOUNDS_DIR = Path(__file__).parent / "sounds"
SOUNDS_DIR.mkdir(exist_ok=True)

# Free sound effect URLs (public domain / CC0)
# These are from various free sources
SOUND_URLS = {
    # Using pixabay free sounds (no attribution required)
    'rimshot.wav': 'https://cdn.pixabay.com/audio/2022/03/15/audio_7a569d6dde.mp3',
    'laugh.wav': 'https://cdn.pixabay.com/audio/2024/02/14/audio_70fa4b1f7c.mp3',
    'sad_trombone.wav': 'https://cdn.pixabay.com/audio/2022/03/15/audio_cce0f1f0f1.mp3',
    'cheer.wav': 'https://cdn.pixabay.com/audio/2021/08/04/audio_0625c1539c.mp3',
    'boo.wav': 'https://cdn.pixabay.com/audio/2022/10/30/audio_f2a4d3d7db.mp3',
    'drumroll.wav': 'https://cdn.pixabay.com/audio/2022/03/24/audio_52a6ef9129.mp3',
    'crickets.wav': 'https://cdn.pixabay.com/audio/2022/03/09/audio_691875e05c.mp3',
    'phone_ring.wav': 'https://cdn.pixabay.com/audio/2022/03/15/audio_0f66b49312.mp3',
}

def download_sound(name, url):
    """Download a sound file"""
    output_path = SOUNDS_DIR / name

    if output_path.exists():
        print(f"  ✓ {name} (already exists)")
        return True

    try:
        print(f"  Downloading {name}...")

        # Download the file
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()

        # If it's an MP3, we need to convert it
        if url.endswith('.mp3'):
            temp_mp3 = SOUNDS_DIR / f"temp_{name}.mp3"
            with open(temp_mp3, 'wb') as f:
                f.write(data)

            # Try to convert with ffmpeg
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-y', '-i', str(temp_mp3),
                '-ar', '24000', '-ac', '1',
                str(output_path)
            ], capture_output=True)

            temp_mp3.unlink()  # Remove temp file

            if result.returncode == 0:
                print(f"  ✓ {name}")
                return True
            else:
                print(f"  ✗ {name} (ffmpeg conversion failed)")
                return False
        else:
            with open(output_path, 'wb') as f:
                f.write(data)
            print(f"  ✓ {name}")
            return True

    except Exception as e:
        print(f"  ✗ {name} ({e})")
        return False

def main():
    print("Downloading sound effects for radio show soundboard...")
    print(f"Saving to: {SOUNDS_DIR}\n")

    # Check for ffmpeg
    import subprocess
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except:
        print("WARNING: ffmpeg not found. Install it with: brew install ffmpeg")
        print("Some sounds may not download correctly.\n")

    success = 0
    for name, url in SOUND_URLS.items():
        if download_sound(name, url):
            success += 1

    print(f"\nDownloaded {success}/{len(SOUND_URLS)} sounds.")
    print("\nTo add more sounds:")
    print("  1. Find free .wav files online")
    print("  2. Name them according to the SOUNDBOARD mapping in radio_show.py")
    print("  3. Place them in the sounds/ directory")
    print("\nRecommended free sound sources:")
    print("  - freesound.org")
    print("  - pixabay.com/sound-effects")
    print("  - zapsplat.com")
    print("  - soundbible.com")

if __name__ == "__main__":
    main()
