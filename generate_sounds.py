#!/usr/bin/env python3
"""
Generate sound effects using ElevenLabs Sound Effects API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SOUNDS_DIR = Path(__file__).parent / "sounds"
SOUNDS_DIR.mkdir(exist_ok=True)

# Sound effects to generate with descriptions
SOUND_EFFECTS = {
    'airhorn.wav': 'loud air horn blast, sports event',
    'boo.wav': 'crowd booing, disappointed audience',
    'crickets.wav': 'crickets chirping, awkward silence',
    'drumroll.wav': 'drum roll, building suspense',
    'buzzer.wav': 'game show wrong answer buzzer',
    'laugh.wav': 'audience laughing, sitcom laugh track',
    'rimshot.wav': 'ba dum tss, drum rimshot comedy',
    'sad_trombone.wav': 'sad trombone, wah wah wah failure sound',
    'phone_ring.wav': 'old telephone ringing',
    'cheer.wav': 'crowd cheering and applauding',
    'scratch.wav': 'vinyl record scratch',
    'wow.wav': 'crowd saying wow, impressed reaction',
    'fart.wav': 'comedic fart sound effect',
    'victory.wav': 'victory fanfare, triumphant horns',
    'uh_oh.wav': 'uh oh, something went wrong sound',
}

def generate_sound(name, description):
    """Generate a sound effect using ElevenLabs"""
    from elevenlabs.client import ElevenLabs
    import soundfile as sf
    import numpy as np

    output_path = SOUNDS_DIR / name

    if output_path.exists():
        print(f"  ✓ {name} (already exists)")
        return True

    try:
        print(f"  Generating {name}: '{description}'...")

        client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

        # Generate sound effect
        result = client.text_to_sound_effects.convert(
            text=description,
            duration_seconds=2.0,
        )

        # Collect audio data
        audio_data = b''.join(result)

        # Save as mp3 first, then convert
        temp_mp3 = SOUNDS_DIR / f"temp_{name}.mp3"
        with open(temp_mp3, 'wb') as f:
            f.write(audio_data)

        # Convert to wav with ffmpeg
        import subprocess
        subprocess.run([
            'ffmpeg', '-y', '-i', str(temp_mp3),
            '-ar', '24000', '-ac', '1',
            str(output_path)
        ], capture_output=True, check=True)

        temp_mp3.unlink()
        print(f"  ✓ {name}")
        return True

    except Exception as e:
        print(f"  ✗ {name} ({e})")
        return False

def main():
    print("Generating sound effects with ElevenLabs...")
    print(f"Saving to: {SOUNDS_DIR}")
    print("(This uses your ElevenLabs credits)\n")

    # Check for ffmpeg
    import subprocess
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except:
        print("ERROR: ffmpeg required. Install with: brew install ffmpeg")
        return

    success = 0
    for name, description in SOUND_EFFECTS.items():
        if generate_sound(name, description):
            success += 1

    print(f"\nGenerated {success}/{len(SOUND_EFFECTS)} sounds.")

if __name__ == "__main__":
    main()
