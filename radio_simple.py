#!/usr/bin/env python3
"""
Simplified Radio Show - for debugging
"""

import os
import sys
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import butter, filtfilt
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 24000

CALLERS = {
    "1": ("Big Tony", "IKne3meq5aSn9XLyUdCD", "You are Big Tony, a loud Italian guy from Staten Island. Swear naturally, be opinionated. Keep it to 2 sentences."),
    "2": ("Drunk Diane", "FGY2WhTYpPnrIDTdsKH5", "You are Drunk Diane, tipsy woman at a bar. Ramble a bit, be funny. Keep it to 2 sentences."),
    "3": ("Stoner Phil", "bIHbv24MWmeRgasZH58o", "You are Stoner Phil, super chill stoner dude. Speak slow, be spacey but profound. Keep it to 2 sentences."),
}

def phone_filter(audio):
    b, a = butter(4, [300/(SAMPLE_RATE/2), 3400/(SAMPLE_RATE/2)], btype='band')
    return (np.tanh(filtfilt(b, a, audio.flatten()) * 1.5) * 0.8).astype(np.float32)

class SimpleRadio:
    def __init__(self):
        print("Loading Whisper...")
        self.whisper = WhisperModel("base", device="cpu", compute_type="int8")

        print("Connecting to ElevenLabs...")
        from elevenlabs.client import ElevenLabs
        self.tts = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

        print("Connecting to Ollama...")
        import ollama
        self.ollama = ollama

        self.caller = CALLERS["1"]
        self.history = []
        print("\nReady!\n")

    def record(self):
        print("  [Recording - press Enter to stop]")
        chunks = []
        recording = True

        def callback(indata, frames, time, status):
            if recording:
                chunks.append(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
            input()  # Wait for Enter

        recording = False
        return np.vstack(chunks) if chunks else None

    def transcribe(self, audio):
        import librosa
        audio_16k = librosa.resample(audio.flatten().astype(np.float32), orig_sr=SAMPLE_RATE, target_sr=16000)
        segments, _ = self.whisper.transcribe(audio_16k)
        return " ".join([s.text for s in segments]).strip()

    def respond(self, text):
        self.history.append({"role": "user", "content": text})

        response = self.ollama.chat(
            model="llama3.2:latest",
            messages=[{"role": "system", "content": self.caller[2]}] + self.history[-6:],
            options={"temperature": 0.9}
        )

        reply = response["message"]["content"]
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def speak(self, text):
        print("  [Generating voice...]")
        audio_gen = self.tts.text_to_speech.convert(
            voice_id=self.caller[1],
            text=text,
            model_id="eleven_turbo_v2_5",
            output_format="pcm_24000"
        )

        audio_bytes = b"".join(audio_gen)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        filtered = phone_filter(audio)

        print("  [Playing...]")
        sd.play(filtered, SAMPLE_RATE)
        sd.wait()

    def run(self):
        print("=" * 50)
        print("  SIMPLE RADIO - Type commands:")
        print("  1/2/3 = switch caller")
        print("  r     = record & respond")
        print("  t     = type message (skip recording)")
        print("  q     = quit")
        print("=" * 50)
        print(f"\nCaller: {self.caller[0]}\n")

        while True:
            cmd = input("> ").strip().lower()

            if cmd == 'q':
                break
            elif cmd in '123':
                self.caller = CALLERS[cmd]
                self.history = []
                print(f"\n📞 Switched to: {self.caller[0]}\n")
            elif cmd == 'r':
                audio = self.record()
                if audio is not None:
                    print("  [Transcribing...]")
                    text = self.transcribe(audio)
                    print(f"\n  YOU: {text}\n")
                    if text:
                        print("  [Thinking...]")
                        reply = self.respond(text)
                        print(f"\n  📞 {self.caller[0].upper()}: {reply}\n")
                        self.speak(reply)
            elif cmd == 't':
                text = input("  Type message: ")
                if text:
                    print("  [Thinking...]")
                    reply = self.respond(text)
                    print(f"\n  📞 {self.caller[0].upper()}: {reply}\n")
                    self.speak(reply)
            else:
                print("  Commands: r=record, t=type, 1/2/3=caller, q=quit")

if __name__ == "__main__":
    radio = SimpleRadio()
    radio.run()
