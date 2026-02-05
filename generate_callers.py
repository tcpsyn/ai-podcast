import os
os.environ["SUNO_USE_SMALL_MODELS"] = "False"

from bark import generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from scipy.signal import butter, filtfilt
import numpy as np

def phone_filter(audio, sample_rate=24000):
    """Apply telephone bandpass filter (300Hz - 3400Hz)"""
    low = 300 / (sample_rate / 2)
    high = 3400 / (sample_rate / 2)
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, audio)

    # Add slight compression and normalize
    filtered = np.tanh(filtered * 1.5) * 0.9
    return filtered.astype(np.float32)

# Define your callers
CALLERS = [
    {
        "name": "caller1_mike",
        "voice": "v2/en_speaker_6",
        "text": """Hey, thanks for taking my call!
        So I've been thinking about this a lot and...
        I know it sounds crazy, but hear me out."""
    },
    {
        "name": "caller2_sarah",
        "voice": "v2/en_speaker_9",
        "text": """Hi! Oh my gosh, I can't believe I got through.
        Okay so... this is kind of a long story,
        but basically I had this experience last week that blew my mind."""
    },
    {
        "name": "caller3_dave",
        "voice": "v2/en_speaker_1",
        "text": """Yeah, hey. First time caller, long time listener.
        Look, I gotta be honest with you here,
        I think you're missing something important."""
    },
    {
        "name": "caller4_jenny",
        "voice": "v2/en_speaker_3",
        "text": """Okay okay, so get this...
        I was literally just talking about this with my friend yesterday!
        And she said, and I quote, well, I can't say that on air."""
    },
]

def main():
    print("Loading models...")
    preload_models()

    os.makedirs("output", exist_ok=True)

    for caller in CALLERS:
        print(f"\nGenerating: {caller['name']}")

        # Generate raw audio
        audio = generate_audio(caller["text"], history_prompt=caller["voice"])

        # Save clean version
        write_wav(f"output/{caller['name']}_clean.wav", 24000, audio)

        # Apply phone filter and save
        phone_audio = phone_filter(audio)
        write_wav(f"output/{caller['name']}_phone.wav", 24000, phone_audio)

        print(f"  Saved: output/{caller['name']}_clean.wav")
        print(f"  Saved: output/{caller['name']}_phone.wav")

    print("\nDone! Check the output/ folder.")

if __name__ == "__main__":
    main()
