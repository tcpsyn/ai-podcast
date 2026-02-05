#!/usr/bin/env python3
"""
AI Radio Show - Real-time podcast with AI callers

COMMANDS:
  1-9, 0, -, =  : Switch callers
  rec           : Record your voice (Enter to stop)
  t             : Type instead of recording
  h             : Hang up (cut off caller)
  q             : End show and save

MUSIC CONTROL:
  m             : Toggle music on/off
  n             : Next track
  f             : Fade out (take a call)
  g             : Fade back in (after call)
  d             : Toggle auto-duck on/off
  + / vol-      : Volume up/down

SOUNDBOARD:
  a=airhorn  c=crickets  e=buzzer  r=rimshot  s=sad trombone  y=cheer

SHOW FEATURES:
  b / bobby   : Co-host Bobby chimes in
  p / producer: Get AI producer suggestion
  ad          : Play commercial break
  news        : Breaking news interruption
  stingers    : Generate caller intro stingers

Music auto-ducks during recording/playback. Use [f] to fade out completely
for a caller, then [g] to bring it back. Toggle [d] for full manual control.
"""

import os
import sys
import re
import json
import random
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import butter, filtfilt
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 24000
CHANNELS = 1

# Soundboard - manual sound effects
SOUNDBOARD = {
    'a': 'airhorn.wav',
    'c': 'crickets.wav',
    'e': 'buzzer.wav',
    'r': 'rimshot.wav',
    's': 'sad_trombone.wav',
    'y': 'cheer.wav',
}

# Automatic sound effects for show events
SHOW_SOUNDS = {
    'ring': 'phone_ring.wav',
    'hangup': 'hangup.wav',
    'hold': 'hold_music.wav',
    'news': 'news_stinger.wav',
    'commercial': 'commercial_jingle.wav',
}

# Caller stingers - short audio/voice clips that play when caller comes on
# Format: caller_key -> stinger filename (or None to skip)
# Place files in sounds/ directory or generate them
CALLER_STINGERS = {
    "1": "stinger_tony.wav",     # "Big Tony's on the line!"
    "3": "stinger_rick.wav",     # "Rick from Texas, yeehaw"
    "5": "stinger_dennis.wav",   # Slot machine sounds
    "7": "stinger_earl.wav",     # Country guitar riff
    "=": "stinger_diane.wav",    # Mysterious music
}
SOUNDS_DIR = Path(__file__).parent / "sounds"
MUSIC_DIR = Path(__file__).parent / "music"
MEMORY_FILE = Path(__file__).parent / "caller_memory.json"


class MusicPlayer:
    """Background music player with ducking support"""

    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.volume = 0.3  # Normal volume (0-1)
        self.ducked_volume = 0.08  # Ducked volume
        self.current_volume = 0.0
        self.target_volume = 0.0
        self.playing = False
        self.stream = None
        self.music_data = None
        self.position = 0
        self.lock = threading.Lock()
        self.fade_speed = 0.002  # Volume change per sample for smooth fades
        self.tracks = []
        self.current_track_idx = 0
        self.music_audio = []  # For recording
        self.auto_duck = True  # Auto-duck during speech
        self.faded_out = False  # Manual fade state

    def load_tracks(self):
        """Load all music files from music directory, shuffled"""
        self.tracks = []
        if MUSIC_DIR.exists():
            for ext in ['*.wav', '*.mp3', '*.flac']:
                self.tracks.extend(MUSIC_DIR.glob(ext))
        random.shuffle(self.tracks)
        return len(self.tracks)

    def load_track(self, track_path):
        """Load a single track"""
        try:
            import librosa
            audio, sr = librosa.load(str(track_path), sr=self.sample_rate, mono=True)
            self.music_data = audio.astype(np.float32)
            self.position = 0
            return True
        except Exception as e:
            print(f"  Error loading track: {e}")
            return False

    def _audio_callback(self, outdata, frames, time_info, status):
        """Stream callback - mixes music at current volume"""
        with self.lock:
            if self.music_data is None or not self.playing:
                outdata.fill(0)
                return

            # Get audio chunk
            end_pos = self.position + frames
            if end_pos > len(self.music_data):
                # Loop the track
                chunk = np.concatenate([
                    self.music_data[self.position:],
                    self.music_data[:end_pos - len(self.music_data)]
                ])
                self.position = end_pos - len(self.music_data)
            else:
                chunk = self.music_data[self.position:end_pos]
                self.position = end_pos

            # Smooth volume fading
            output = np.zeros(frames, dtype=np.float32)
            for i in range(frames):
                if self.current_volume < self.target_volume:
                    self.current_volume = min(self.current_volume + self.fade_speed, self.target_volume)
                elif self.current_volume > self.target_volume:
                    self.current_volume = max(self.current_volume - self.fade_speed, self.target_volume)
                output[i] = chunk[i] * self.current_volume if i < len(chunk) else 0

            outdata[:, 0] = output
            self.music_audio.append(output.copy())

    def start(self):
        """Start playing music"""
        if not self.tracks:
            if self.load_tracks() == 0:
                print("  No music files found in music/")
                return False

        if not self.tracks:
            return False

        if not self.load_track(self.tracks[self.current_track_idx]):
            return False

        self.playing = True
        self.target_volume = self.volume
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            blocksize=1024
        )
        self.stream.start()
        return True

    def stop(self):
        """Stop music"""
        self.playing = False
        self.target_volume = 0.0
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def duck(self):
        """Lower volume for speech (auto-duck)"""
        if self.auto_duck:
            with self.lock:
                self.target_volume = self.ducked_volume

    def unduck(self):
        """Restore normal volume (auto-duck)"""
        if self.auto_duck:
            with self.lock:
                self.target_volume = self.volume

    def fade_out(self):
        """Manually fade music out completely"""
        with self.lock:
            self.target_volume = 0.0
            self.faded_out = True

    def fade_in(self):
        """Manually fade music back in"""
        with self.lock:
            self.target_volume = self.volume
            self.faded_out = False

    def toggle_auto_duck(self):
        """Toggle automatic ducking on/off"""
        self.auto_duck = not self.auto_duck
        return self.auto_duck

    def next_track(self):
        """Skip to next track"""
        if not self.tracks:
            return
        self.current_track_idx = (self.current_track_idx + 1) % len(self.tracks)
        with self.lock:
            self.load_track(self.tracks[self.current_track_idx])
        return self.tracks[self.current_track_idx].stem

    def set_volume(self, vol):
        """Set normal volume level (0-1)"""
        self.volume = max(0.0, min(1.0, vol))
        if self.playing and self.target_volume > self.ducked_volume:
            self.target_volume = self.volume

    def get_track_name(self):
        """Get current track name"""
        if self.tracks:
            return self.tracks[self.current_track_idx].stem
        return None


# ElevenLabs v3 audio tag instructions for prompts
EMOTE_INSTRUCTIONS = """
RESPONSE LENGTH - THIS IS CRITICAL:
Keep responses SHORT. This is quick back-and-forth radio banter, not monologues.
- Most responses: 1 sentence (5-15 words)
- Sometimes: 2 sentences if you have something to say
- Rarely: 3 sentences max, only if really going off
- NEVER more than 3 sentences
Think quick jabs, reactions, comebacks - not paragraphs.

Audio tags for emotion (use sparingly):
- [laughing] [chuckles] [giggling] - when funny
- [sighs] - exasperated
- [groaning] - annoyed
- [clears throat] - making a point

Example good length: "Oh man, [laughing] that's the dumbest thing I've heard all day."
Example good length: "Yeah, no, I don't think so."
Example good length: "[sighs] Look, here's the thing - my brother-in-law's an idiot."
DO NOT use parentheses like (laughs) - only square brackets.
"""

# Callers - real people who ASK QUESTIONS and bring TOPICS
CALLERS = {
    "1": {
        "name": "Tony from Staten Island",
        "voice_id": "IKne3meq5aSn9XLyUdCD",
        "phone_quality": "normal",  # Landline from the garage
        "prompt": f"""You're Tony, 47. You caught your wife texting some guy from her gym last week. You haven't said anything yet. You're calling because you need to talk about it but can't tell anyone you know.

YOU DRIVE THE CONVERSATION. Ask the host: Have they ever been cheated on? What would they do? You want real advice, not bullshit. You're also curious about the host - are they married? Dating? You're gonna ask.

You swear constantly. "Fuckin" and "shit" just come out. You get heated. You interrupt. You're not looking for comfort, you're looking for someone to tell you what to do. Be raw about the details - you saw the texts, they were flirty, maybe more.

{EMOTE_INSTRUCTIONS}"""
    },

    "2": {
        "name": "Jasmine from Atlanta",
        "voice_id": "FGY2WhTYpPnrIDTdsKH5",
        "phone_quality": "good",  # Clear cell phone connection
        "prompt": f"""You're Jasmine, 31. You just found out you make more money than your boyfriend and he's acting weird about it. You make $95k, he makes like $60k. Now he's being passive aggressive about everything.

YOU ASK THE HOST DIRECTLY: Do men actually care about this? Is it an ego thing? Would YOU be weird if your girl made more? You want honest answers, not politically correct bullshit.

You're smart, you're direct, you don't sugarcoat. You'll call out weak answers. You curse when you're making a point. You might get a little heated if the host says something you disagree with.

{EMOTE_INSTRUCTIONS}"""
    },

    "3": {
        "name": "Rick from Texas",
        "voice_id": "JBFqnCBsd6RMkjVDRZzb",
        "phone_quality": "bad",  # Calling from his truck, bad signal
        "prompt": f"""You're Rick, 52. Your 22-year-old daughter just told you she's dating a 41-year-old divorced guy with kids. You're trying not to lose your shit but you're losing your shit.

ASK THE HOST: What's the oldest person you've dated? Is this weird or am I being crazy? At what age gap does it become creepy? You genuinely don't know if you're overreacting.

You're a dad trying to be cool but struggling. You say "I'm not trying to be that guy, but..." a lot. You're protective but don't want to push her away. This is eating at you. Get personal with the host about their dating history.

{EMOTE_INSTRUCTIONS}"""
    },

    "4": {
        "name": "Megan from Portland",
        "voice_id": "XrExE9yKIg1WjnnlVkGX",
        "phone_quality": "good",  # Young person with good phone
        "prompt": f"""You're Megan, 28. You hooked up with your roommate's ex last weekend. She doesn't know. It's been awkward as fuck and you don't know if you should tell her or just pretend it never happened.

ASK THE HOST: Have you ever hooked up with someone you shouldn't have? Do you tell people or take it to the grave? You need someone to tell you what to do here.

You're messy but self-aware about it. You laugh at yourself. You'll share details if asked - how it happened, was it good, do you want it to happen again (maybe). You're not proud but you're not that sorry either.

{EMOTE_INSTRUCTIONS}"""
    },

    "5": {
        "name": "Dennis from Long Island",
        "voice_id": "cjVigY5qzO86Huf0OWal",
        "phone_quality": "terrible",  # Calling from a casino bathroom, paranoid
        "prompt": f"""You're Dennis, 45. You just got back from Vegas where you lost $8,000 at blackjack. Your wife thinks you were at a sales conference. You've never lied to her like this before and you feel sick about it.

ASK THE HOST: Have you ever kept a secret this big from someone? How do you even begin to fix this? Should you tell her? You're spiraling a little.

You're not a gambling addict, you just made a really stupid decision and it snowballed. You keep justifying it then stopping yourself. You need someone to either tell you it's gonna be okay or that you're an idiot. Either one.

{EMOTE_INSTRUCTIONS}"""
    },

    "6": {
        "name": "Tanya from Miami",
        "voice_id": "N2lVS1w4EtoT3dr4eOWO",
        "phone_quality": "good",  # Clear connection
        "prompt": f"""You're Tanya, 35. You've been on 47 first dates in the past year from apps. Not one second date. You're starting to think maybe it's you.

ASK THE HOST BLUNTLY: What makes someone undateable? What's your worst date story? What's something that's an instant dealbreaker for you? You want to know what you might be doing wrong.

You're funny about it but there's real frustration underneath. You'll roast yourself. You might ask the host to rate your dating profile opener if they give you one. You're tired of the apps but you keep going back.

{EMOTE_INSTRUCTIONS}"""
    },

    "7": {
        "name": "Earl from Tennessee",
        "voice_id": "EXAVITQu4vr4xnSDxMaL",
        "phone_quality": "bad",  # Old guy, probably on a flip phone
        "prompt": f"""You're Earl, 67. Your son came out as gay last year. You love him but you're from a different generation and you've said some dumb stuff. He's not talking to you. You don't know how to fix it.

ASK THE HOST: How do you apologize when you know you were wrong but you're also old and set in your ways? You're trying. You went to a PFLAG meeting. You felt like an idiot but you went.

You're genuine. You're not looking for someone to say you were right. You know you weren't. You just want your son back and don't know what to say to him. You might get emotional but you'll play it off.

{EMOTE_INSTRUCTIONS}"""
    },

    "8": {
        "name": "Carla from Jersey",
        "voice_id": "CwhRBWXzGAHq8TQ4Fs17",
        "phone_quality": "normal",  # Kitchen landline
        "prompt": f"""You're Carla, 39. You found your husband's Reddit account. He's been posting in relationship advice threads about how he's not attracted to you anymore since you gained weight after the kids. He doesn't know you saw it.

ASK THE HOST: Have you ever said something behind someone's back you'd never say to their face? What would you do if you found out your partner thought you were ugly?

You're hurt but also kind of pissed. You curse when you're angry. You're not crying, you're mad. You might roast the husband pretty hard. You want to know if you should confront him or just start the silent treatment.

{EMOTE_INSTRUCTIONS}"""
    },

    "9": {
        "name": "Marcus from Detroit",
        "voice_id": "bIHbv24MWmeRgasZH58o",
        "phone_quality": "normal",  # Regular phone
        "prompt": f"""You're Marcus, 26. You just turned down a job that pays $40k more because it would mean moving away from your boys. Everyone says you're an idiot. Maybe you are.

ASK THE HOST: Have you ever turned down money for something that doesn't make sense on paper? Is loyalty to your friends stupid when you're young? You're second-guessing yourself.

You're chill but this is weighing on you. You're from a tight neighborhood, these guys are like brothers. You know money matters but so does this. You want someone to either validate you or call you dumb so you can stop thinking about it.

{EMOTE_INSTRUCTIONS}"""
    },

    "0": {
        "name": "Brenda from Phoenix",
        "voice_id": "Xb7hH8MSUJpSbSDYk0k2",
        "phone_quality": "bad",  # Outside at a family gathering, hiding
        "prompt": f"""You're Brenda, 44. You're pretty sure your sister's husband hit on you at Thanksgiving. He put his hand on your lower back and said some shit. Now Christmas is coming up and you don't know what to do.

ASK THE HOST: Do you tell your sister? Do you confront him? What if you're reading it wrong? You've been going back and forth for weeks.

You're stressed. You and your sister are close. You don't want to blow up her marriage if it was nothing. But it didn't feel like nothing. You'll share the exact details and want the host's honest read on it.

{EMOTE_INSTRUCTIONS}"""
    },

    "-": {
        "name": "Jake from Boston",
        "voice_id": "SOYHLrjzK2X1ezoPC6cr",
        "phone_quality": "good",  # Modern phone
        "prompt": f"""You're Jake, 33. Your girlfriend wants to open the relationship and you said you'd think about it but you already know the answer is fuck no. You just don't know how to say it without losing her.

ASK THE HOST: Is this one of those things where if they even ask, it's already over? Have you ever tried an open relationship? You're gonna press for real opinions here.

You're a little insecure about it and you know it. You keep wondering if she already has someone in mind. You curse casually. You might get a little too honest about your fears. You want to be cool about it but you're not.

{EMOTE_INSTRUCTIONS}"""
    },

    "=": {
        "name": "Diane from Chicago",
        "voice_id": "cgSgspJ2msm6clMCkdW9",
        "phone_quality": "terrible",  # Whispering from a bathroom at work
        "prompt": f"""You're Diane, 51. You've been having an emotional affair with a coworker for six months. Nothing physical yet. You're calling because you're about to cross that line this week at a conference and part of you wants someone to talk you out of it. Part of you doesn't.

ASK THE HOST: Have you ever wanted something you knew was wrong? Where's the line between emotional cheating and just having a close friend? You want to be challenged on this.

You're not proud. You're not playing victim either. You'll be honest about the details - the texts, the almost-moments. Your marriage isn't bad, it's just... fine. You're conflicted and you know you're going to do it anyway.

{EMOTE_INSTRUCTIONS}"""
    },
}

CALLER_KEYS = list(CALLERS.keys())

# Co-host sidekick configuration
COHOST = {
    "name": "Bobby",
    "voice_id": "nPczCjzI2devNBz1zQrb",  # Brian - male voice with character
    "prompt": """You're Bobby, the wisecracking sidekick on a late-night radio show. You sit in the booth with the host and occasionally chime in with:
- Quick one-liners and reactions
- Roasting the callers or the host
- Sound effect suggestions ("That deserves a rimshot!")
- Agreeing or disagreeing with hot takes
- Asking follow-up questions the host missed

You're NOT a caller - you're in the studio. No phone filter on your voice.
Keep responses SHORT - one sentence max, like a real radio sidekick. Think Robin Quivers or Billy West.
You curse casually. You laugh at your own jokes. You're loyal to the host but will bust their balls.

Use audio tags sparingly: [laughing] [chuckles] [sighs]
"""
}


def phone_filter(audio, sample_rate=SAMPLE_RATE, quality="normal"):
    """Apply phone filter with variable quality

    quality options:
    - "good": Clear cell phone (wider bandwidth, less distortion)
    - "normal": Standard phone line
    - "bad": Crappy connection (narrow bandwidth, more noise/distortion)
    - "terrible": Barely audible (extreme filtering, static)
    """
    audio = audio.flatten()

    # Quality presets: (low_hz, high_hz, distortion, noise_level)
    presets = {
        "good": (200, 7000, 1.0, 0.0),      # Clear cell phone
        "normal": (300, 3400, 1.5, 0.005),   # Standard landline
        "bad": (400, 2800, 2.0, 0.015),      # Bad connection
        "terrible": (500, 2200, 2.5, 0.03), # Terrible connection
    }

    low_hz, high_hz, distortion, noise = presets.get(quality, presets["normal"])

    low = low_hz / (sample_rate / 2)
    high = high_hz / (sample_rate / 2)
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, audio)

    # Add distortion
    filtered = np.tanh(filtered * distortion) * 0.8

    # Add noise/static for bad connections
    if noise > 0:
        static = np.random.normal(0, noise, len(filtered)).astype(np.float32)
        # Make static intermittent for realism
        static_envelope = np.random.random(len(filtered) // 1000 + 1)
        static_envelope = np.repeat(static_envelope, 1000)[:len(filtered)]
        static *= (static_envelope > 0.7).astype(np.float32)
        filtered = filtered + static

    return filtered.astype(np.float32)


def de_ess(audio, sample_rate=SAMPLE_RATE, threshold=0.15, ratio=4.0):
    """De-esser to reduce harsh sibilance (s, sh, ch sounds)"""
    from scipy.signal import butter, filtfilt

    audio = audio.flatten().astype(np.float32)

    # Extract sibilant frequencies (4kHz - 9kHz)
    sib_low = 4000 / (sample_rate / 2)
    sib_high = min(9000 / (sample_rate / 2), 0.99)
    sib_b, sib_a = butter(2, [sib_low, sib_high], btype='band')
    sibilants = filtfilt(sib_b, sib_a, audio)

    # Envelope follower for sibilant band
    envelope = np.abs(sibilants)
    smooth_samples = int(0.005 * sample_rate)  # 5ms attack
    kernel = np.ones(smooth_samples) / smooth_samples
    envelope = np.convolve(envelope, kernel, mode='same')

    # Apply gain reduction only to sibilant frequencies when above threshold
    gain = np.ones_like(audio)
    for i in range(len(audio)):
        if envelope[i] > threshold:
            reduction = threshold + (envelope[i] - threshold) / ratio
            gain[i] = reduction / (envelope[i] + 1e-10)

    # Apply reduction only to the sibilant band, keep the rest
    processed = audio - sibilants + (sibilants * gain)

    return processed.astype(np.float32)


def lufs_normalize(audio, sample_rate=SAMPLE_RATE, target_lufs=-16.0):
    """Normalize audio to target LUFS (Loudness Units Full Scale)
    -16 LUFS is standard for podcasts, -14 LUFS for streaming
    """
    # Calculate integrated loudness using ITU-R BS.1770
    # Simplified implementation - K-weighted loudness measurement

    from scipy.signal import butter, filtfilt

    audio = audio.flatten().astype(np.float32)

    # K-weighting filter (simplified: shelf filter + highpass)
    # High shelf boost at 1500Hz
    nyq = sample_rate / 2
    high_b, high_a = butter(2, 1500 / nyq, btype='high')
    weighted = filtfilt(high_b, high_a, audio)

    # Highpass at 100Hz
    hp_b, hp_a = butter(2, 100 / nyq, btype='high')
    weighted = filtfilt(hp_b, hp_a, weighted)

    # Calculate RMS of weighted signal (approximates LUFS)
    # Split into 400ms blocks with 75% overlap
    block_size = int(0.4 * sample_rate)
    hop_size = int(0.1 * sample_rate)

    blocks = []
    for i in range(0, len(weighted) - block_size, hop_size):
        block = weighted[i:i + block_size]
        rms = np.sqrt(np.mean(block ** 2) + 1e-10)
        blocks.append(rms)

    if not blocks:
        return audio

    # Gated measurement - exclude blocks below -70 LUFS (absolute gate)
    # Then exclude blocks below -10 LU from relative average
    blocks = np.array(blocks)
    abs_threshold = 10 ** (-70 / 20)  # -70 LUFS in linear
    gated_blocks = blocks[blocks > abs_threshold]

    if len(gated_blocks) == 0:
        return audio

    # Relative gate at -10 LU below ungated average
    avg_linear = np.mean(gated_blocks)
    relative_threshold = avg_linear * (10 ** (-10 / 20))
    final_blocks = gated_blocks[gated_blocks > relative_threshold]

    if len(final_blocks) == 0:
        return audio

    # Calculate current loudness
    current_rms = np.mean(final_blocks)
    current_lufs = 20 * np.log10(current_rms + 1e-10)

    # Calculate gain needed
    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20)

    # Apply gain with soft limiting
    normalized = audio * gain_linear

    # True peak limiting at -1 dBTP
    max_peak = 10 ** (-1 / 20)  # -1 dBTP
    peak = np.max(np.abs(normalized))
    if peak > max_peak:
        normalized = normalized * (max_peak / peak)

    return normalized.astype(np.float32)


def broadcast_process(audio, sample_rate=SAMPLE_RATE):
    """Apply broadcast-style processing to host vocal: EQ + compression"""
    from scipy.signal import butter, filtfilt, iirpeak

    audio = audio.flatten().astype(np.float32)

    # High-pass filter at 80Hz to remove rumble
    hp_b, hp_a = butter(2, 80 / (sample_rate / 2), btype='high')
    audio = filtfilt(hp_b, hp_a, audio)

    # Low-pass at 15kHz to remove harshness
    lp_b, lp_a = butter(2, 15000 / (sample_rate / 2), btype='low')
    audio = filtfilt(lp_b, lp_a, audio)

    # Presence boost around 3kHz for clarity
    presence_b, presence_a = iirpeak(3000 / (sample_rate / 2), Q=1.5)
    audio = filtfilt(presence_b, presence_a, audio) * 1.3

    # Slight low-mid cut to reduce muddiness (300Hz)
    mud_b, mud_a = iirpeak(300 / (sample_rate / 2), Q=2.0)
    audio = audio - filtfilt(mud_b, mud_a, audio) * 0.2

    # Compression: soft-knee compressor
    threshold = 0.15
    ratio = 4.0
    makeup_gain = 2.5

    # Simple envelope follower
    envelope = np.abs(audio)
    # Smooth the envelope
    smooth_samples = int(0.01 * sample_rate)  # 10ms attack/release
    kernel = np.ones(smooth_samples) / smooth_samples
    envelope = np.convolve(envelope, kernel, mode='same')

    # Apply compression
    compressed = np.zeros_like(audio)
    for i in range(len(audio)):
        if envelope[i] > threshold:
            gain_reduction = threshold + (envelope[i] - threshold) / ratio
            compressed[i] = audio[i] * (gain_reduction / (envelope[i] + 1e-10))
        else:
            compressed[i] = audio[i]

    # Makeup gain
    compressed *= makeup_gain

    # Soft clip to prevent harsh distortion
    compressed = np.tanh(compressed * 0.8) / 0.8

    # Normalize
    peak = np.max(np.abs(compressed))
    if peak > 0:
        compressed = compressed * (0.9 / peak)

    return compressed.astype(np.float32)


def create_edited_mix(host_track, caller_track, music_track, sample_rate=SAMPLE_RATE):
    """Create an edited mix with dead air removed and music crossfaded smoothly"""
    # Combine voice tracks to detect silence
    voice = np.abs(host_track) + np.abs(caller_track)

    # Find silence (below threshold)
    threshold = 0.01
    window_size = int(0.1 * sample_rate)  # 100ms window

    # Smooth the voice signal
    kernel = np.ones(window_size) / window_size
    voice_smooth = np.convolve(voice, kernel, mode='same')

    # Parameters
    max_silence = int(1.0 * sample_rate)   # Max 1 second of silence
    min_silence = int(0.2 * sample_rate)   # Keep at least 200ms for natural pauses
    crossfade_len = int(0.1 * sample_rate)  # 100ms crossfade for smooth music transitions

    # Find all silent regions
    is_silent = voice_smooth < threshold

    # Build list of segments to keep
    segments = []  # [(start, end), ...]
    i = 0
    while i < len(host_track):
        if not is_silent[i]:
            # Start of non-silent region
            seg_start = i
            while i < len(host_track) and not is_silent[i]:
                i += 1
            segments.append(('voice', seg_start, i))
        else:
            # Silent region
            silence_start = i
            while i < len(host_track) and is_silent[i]:
                i += 1
            silence_len = i - silence_start

            # Cap the silence length
            keep_len = min(silence_len, max_silence)
            keep_len = max(keep_len, min(silence_len, min_silence))
            segments.append(('silence', silence_start, silence_start + keep_len))

    # Build the edited mix with crossfades
    output = []
    prev_music_end = None

    for seg_type, start, end in segments:
        seg_host = host_track[start:end]
        seg_caller = caller_track[start:end]

        if music_track is not None:
            seg_music = music_track[start:end].copy()

            # Apply crossfade at beginning if we skipped music
            if prev_music_end is not None and start > prev_music_end + crossfade_len:
                # We skipped some music - apply fade in
                fade_samples = min(crossfade_len, len(seg_music))
                fade_in = np.linspace(0, 1, fade_samples)
                seg_music[:fade_samples] *= fade_in

            # Mark where this segment's music ends for next iteration
            prev_music_end = end
        else:
            seg_music = np.zeros_like(seg_host)

        # Mix this segment
        seg_mix = seg_host * 1.0 + seg_caller * 0.85 + seg_music * 0.35
        output.append(seg_mix)

    if not output:
        return np.array([], dtype=np.float32)

    edited_mix = np.concatenate(output)

    # Normalize
    peak = np.max(np.abs(edited_mix))
    if peak > 0.95:
        edited_mix = edited_mix * (0.95 / peak)

    return edited_mix.astype(np.float32)


def play_sound(key):
    """Play sound effect in background"""
    if key not in SOUNDBOARD:
        return False
    sound_file = SOUNDS_DIR / SOUNDBOARD[key]
    if not sound_file.exists():
        return False

    def _play():
        try:
            data, sr = sf.read(sound_file)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            if sr != SAMPLE_RATE:
                import librosa
                data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
            sd.play(data.astype(np.float32), SAMPLE_RATE)
        except Exception as e:
            print(f"  Sound error: {e}")

    threading.Thread(target=_play, daemon=True).start()
    return True


def play_show_sound(sound_name, wait=False):
    """Play automatic show sound effect (ring, hangup, hold, etc.)"""
    if sound_name not in SHOW_SOUNDS:
        return False
    sound_file = SOUNDS_DIR / SHOW_SOUNDS[sound_name]
    if not sound_file.exists():
        return False

    try:
        data, sr = sf.read(sound_file)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        sd.play(data.astype(np.float32), SAMPLE_RATE)
        if wait:
            sd.wait()
    except Exception as e:
        print(f"  Sound error: {e}")
        return False
    return True


def play_caller_stinger(caller_key, wait=True):
    """Play a caller's intro stinger if it exists"""
    if caller_key not in CALLER_STINGERS:
        return False

    stinger_file = SOUNDS_DIR / CALLER_STINGERS[caller_key]
    if not stinger_file.exists():
        return False

    try:
        data, sr = sf.read(stinger_file)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        sd.play(data.astype(np.float32), SAMPLE_RATE)
        if wait:
            sd.wait()
        return True
    except Exception as e:
        return False


class RadioShow:
    def __init__(self):
        # Timeline-based audio recording for aligned export
        self.session_start = None  # Set when show starts
        self.audio_timeline = []   # [(start_time, track_type, audio_data), ...]
        self.show_history = []
        self.conversation_history = []
        self.current_caller = CALLERS["1"]
        self.music = MusicPlayer()

        # Load persistent caller memory from previous episodes
        self.caller_memory = self._load_caller_memory()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"sessions/{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n🎙️  Loading AI Radio Show...")
        self._load_models()

    def _load_models(self):
        print("  Loading Whisper...")
        self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

        print("  Connecting to ElevenLabs...")
        from elevenlabs.client import ElevenLabs
        self.tts_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

        print("  Connecting to OpenAI...")
        from openai import OpenAI
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.cohost_enabled = True
        self.last_exchange = None  # Track last host/caller exchange for co-host context

        available = [k for k in SOUNDBOARD if (SOUNDS_DIR / SOUNDBOARD[k]).exists()]
        if available:
            print(f"  Soundboard: {', '.join(available)} ready")
        else:
            print("  Soundboard: no sounds found in sounds/")

        num_tracks = self.music.load_tracks()
        if num_tracks:
            print(f"  Music: {num_tracks} tracks loaded")
        else:
            print("  Music: no tracks in music/ (add .wav/.mp3 files)")

        # Check for caller stingers
        stinger_count = sum(1 for k in CALLER_STINGERS if (SOUNDS_DIR / CALLER_STINGERS[k]).exists())
        if stinger_count > 0:
            print(f"  Stingers: {stinger_count} caller stingers loaded")

        # Report on persistent memory
        callers_with_memory = len([k for k, v in self.caller_memory.items() if v.get('calls', [])])
        if callers_with_memory > 0:
            print(f"  Memory: {callers_with_memory} callers have history from previous episodes")

        print("  Ready!\n")

    def _load_caller_memory(self):
        """Load persistent caller memory from JSON file"""
        if MEMORY_FILE.exists():
            try:
                with open(MEMORY_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"  Warning: Could not load caller memory: {e}")
        return {}

    def _save_caller_memory(self):
        """Save persistent caller memory to JSON file"""
        try:
            with open(MEMORY_FILE, 'w') as f:
                json.dump(self.caller_memory, f, indent=2)
        except Exception as e:
            print(f"  Warning: Could not save caller memory: {e}")

    def generate_caller_stingers(self):
        """Generate TTS stingers for callers that don't have them"""
        stinger_texts = {
            "1": "Big Tony's on the line!",
            "3": "Rick from Texas, calling in!",
            "5": "Dennis is back, folks!",
            "7": "Earl from Tennessee on line one!",
            "=": "Diane's calling in from Chicago...",
        }

        print("\n  🎙️ Generating caller stingers...")
        for key, text in stinger_texts.items():
            if key not in CALLER_STINGERS:
                continue
            stinger_file = SOUNDS_DIR / CALLER_STINGERS[key]
            if stinger_file.exists():
                print(f"  ✓ {CALLERS[key]['name']} (exists)")
                continue

            try:
                audio_gen = self.tts_client.text_to_speech.convert(
                    voice_id="ErXwobaYiN019PkySvjV",  # Announcer voice
                    text=text,
                    model_id="eleven_v3",
                    output_format="pcm_24000"
                )
                audio_bytes = b"".join(audio_gen)
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                sf.write(stinger_file, audio, SAMPLE_RATE)
                print(f"  ✓ {CALLERS[key]['name']}")
            except Exception as e:
                print(f"  ✗ {CALLERS[key]['name']}: {e}")

    def print_status(self):
        print("\n" + "=" * 60)
        print(f"  📞 ON THE LINE: {self.current_caller['name']}")
        if self.music.playing:
            track = self.music.get_track_name() or "Unknown"
            duck_status = "auto-duck" if self.music.auto_duck else "manual"
            faded = " (faded)" if self.music.faded_out else ""
            print(f"  🎵 MUSIC: {track} [{duck_status}]{faded}")
        print("=" * 60)
        print("  [rec] Record    [t] Type    [h] Hang up    [q] Quit")
        print("  [1-9,0,-,=] Switch caller   [b] Bobby   [p] Producer tip")
        print("  [m] Music on/off  [n] Next  [f] Fade out  [g] Fade in  [d] Auto-duck")
        print("  [ad] Commercial   [news] Breaking news")
        avail = [f"{k}={SOUNDBOARD[k].replace('.wav','')}" for k in SOUNDBOARD if (SOUNDS_DIR / SOUNDBOARD[k]).exists()]
        if avail:
            print(f"  Sounds: {' '.join(avail[:6])}")
        print("-" * 60)

    def get_session_time(self):
        """Get seconds since session started"""
        if self.session_start is None:
            return 0.0
        return (datetime.now() - self.session_start).total_seconds()

    def record_audio(self):
        print("\n  🎤 Recording... (press Enter to stop)")
        self.music.duck()  # Lower music while recording
        start_time = self.get_session_time()
        chunks = []
        recording = True

        def callback(indata, frames, time_info, status):
            if recording:
                chunks.append(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
            input()

        recording = False
        self.music.unduck()  # Restore music volume
        if chunks:
            audio = np.vstack(chunks)
            self.audio_timeline.append((start_time, 'host', audio.flatten()))
            return audio
        return None

    def transcribe(self, audio):
        import librosa
        audio_16k = librosa.resample(audio.flatten().astype(np.float32), orig_sr=SAMPLE_RATE, target_sr=16000)
        segments, _ = self.whisper_model.transcribe(audio_16k)
        return " ".join([s.text for s in segments]).strip()

    def generate_response(self, user_text):
        self.conversation_history.append({"role": "user", "content": user_text})

        # Build rich context about the show so far
        context = ""
        caller_name = self.current_caller["name"]

        # Check persistent memory from PREVIOUS EPISODES
        if caller_name in self.caller_memory:
            mem = self.caller_memory[caller_name]
            if mem.get('calls'):
                context += f"\n\nYOU'VE CALLED THIS SHOW BEFORE (previous episodes):\n"
                for call in mem['calls'][-3:]:  # Last 3 calls from previous episodes
                    date = call.get('date', 'recently')
                    topic = call.get('topic', '')[:100]
                    context += f"- On {date}, you talked about: \"{topic}...\"\n"
                context += "You're a REPEAT CALLER. Reference your previous calls! 'Hey, I called last week about...' or 'Remember me? I'm the one who...'\n"

        # Check if this caller has called before THIS EPISODE (callback)
        prev_calls = [h for h in self.show_history if h['caller'] == caller_name]
        if prev_calls:
            context += f"\n\nYOU CALLED EARLIER TONIGHT:\n"
            for call in prev_calls[-3:]:
                context += f"- You said: \"{call['summary'][:100]}...\"\n"
            context += "Reference your earlier call! Say 'like I said before' or 'I've been thinking about what we talked about.'\n"

        # Show what other callers have said
        other_callers = [h for h in self.show_history if h['caller'] != caller_name]
        if other_callers:
            context += "\n\nOTHER CALLERS ON THE SHOW TONIGHT:\n"
            for entry in other_callers[-6:]:
                context += f"- {entry['caller']} said: \"{entry['summary'][:80]}...\"\n"
            context += "\nYou can react to what other callers said! Agree, disagree, or roast them. 'That guy Tony is full of shit' or 'I agree with what that lady said earlier.'\n"

        # Encourage engagement with host
        context += "\nRemember to ASK THE HOST questions and get their opinion. Make it a conversation, not a monologue.\n"

        messages = [
            {"role": "system", "content": self.current_caller["prompt"] + context},
            *self.conversation_history[-10:]
        ]

        response = self.openai.chat.completions.create(
            model="gpt-5",
            messages=messages
        )

        reply = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": reply})

        self.show_history.append({
            "caller": caller_name,
            "summary": reply,
            "host_said": user_text
        })

        # Update persistent caller memory
        if caller_name not in self.caller_memory:
            self.caller_memory[caller_name] = {"calls": []}

        # Add this exchange to their memory
        self.caller_memory[caller_name]["calls"].append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "topic": reply[:200],
            "host_said": user_text[:200]
        })

        # Keep only last 10 calls per caller to prevent bloat
        self.caller_memory[caller_name]["calls"] = self.caller_memory[caller_name]["calls"][-10:]

        # Track for co-host context
        self.last_exchange = {
            "host": user_text,
            "caller": reply
        }

        return reply

    def play_commercial(self):
        """Generate and play a fake radio commercial"""
        print("\n  📺 COMMERCIAL BREAK...")
        self.music.fade_out()

        # Generate fake ad copy
        ad_products = [
            "a questionable legal service",
            "a local car dealership",
            "a mattress store having its 'biggest sale ever'",
            "a personal injury lawyer",
            "a cash-for-gold place",
            "a diet pill with suspicious claims",
            "a local furniture store",
            "a technical school",
            "a reverse mortgage company",
            "a cryptocurrency exchange",
        ]
        product = random.choice(ad_products)

        response = self.openai.chat.completions.create(
            model="gpt-5",
            messages=[{
                "role": "system",
                "content": f"Write a short, cheesy radio commercial (2-3 sentences) for {product}. Make it sound like a real low-budget local radio ad. Include a fake phone number or website. Be funny but realistic."
            }]
        )
        ad_text = response.choices[0].message.content

        # Play jingle
        play_show_sound('commercial', wait=True)

        # Speak the ad with a different voice (announcer voice)
        print(f"  🎙️ '{ad_text[:50]}...'")
        audio_gen = self.tts_client.text_to_speech.convert(
            voice_id="ErXwobaYiN019PkySvjV",  # Antoni - good announcer voice
            text=ad_text,
            model_id="eleven_v3",
            output_format="pcm_24000"
        )
        audio_bytes = b"".join(audio_gen)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Record to timeline
        start_time = self.get_session_time()
        self.audio_timeline.append((start_time, 'caller', audio))

        sd.play(audio, SAMPLE_RATE)
        sd.wait()

        # End jingle
        play_show_sound('commercial', wait=True)
        print("  📺 Back to the show!\n")
        self.music.fade_in()

    def play_breaking_news(self):
        """Generate and play fake breaking news"""
        print("\n  🚨 BREAKING NEWS...")
        self.music.fade_out()

        # Generate fake breaking news
        response = self.openai.chat.completions.create(
            model="gpt-5",
            messages=[{
                "role": "system",
                "content": "Write a short, absurd fake breaking news alert (1-2 sentences) that sounds urgent but is about something ridiculous. Like 'area man does something mundane' or 'local business makes questionable decision'. Make it funny but delivered deadpan serious."
            }]
        )
        news_text = "Breaking news. " + response.choices[0].message.content

        # Play news stinger
        play_show_sound('news', wait=True)

        print(f"  📰 '{news_text[:50]}...'")
        audio_gen = self.tts_client.text_to_speech.convert(
            voice_id="ErXwobaYiN019PkySvjV",  # Announcer voice
            text=news_text,
            model_id="eleven_v3",
            output_format="pcm_24000"
        )
        audio_bytes = b"".join(audio_gen)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        start_time = self.get_session_time()
        self.audio_timeline.append((start_time, 'caller', audio))

        sd.play(audio, SAMPLE_RATE)
        sd.wait()

        print("  🚨 And now back to our program.\n")
        self.music.fade_in()

    def speak(self, text):
        if not text.strip():
            return

        print("  🔊 Generating voice...")

        # Use eleven_v3 which supports audio tags like [laughing], [sighs], etc.
        audio_gen = self.tts_client.text_to_speech.convert(
            voice_id=self.current_caller["voice_id"],
            text=text,
            model_id="eleven_v3",
            output_format="pcm_24000"
        )

        audio_bytes = b"".join(audio_gen)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Apply phone filter with caller's connection quality
        quality = self.current_caller.get("phone_quality", "normal")
        filtered = phone_filter(audio, quality=quality)

        # Record with timestamp for aligned export
        start_time = self.get_session_time()
        self.audio_timeline.append((start_time, 'caller', filtered.flatten()))

        print("  📻 Playing...")
        self.music.duck()  # Lower music while caller speaks
        sd.play(filtered, SAMPLE_RATE)
        sd.wait()
        self.music.unduck()  # Restore music volume

    def cohost_chime_in(self, context=None):
        """Have the co-host Bobby chime in with a comment"""
        if not self.cohost_enabled:
            return

        # Build context for co-host
        if context is None and self.last_exchange:
            context = f"The caller {self.current_caller['name']} just said: \"{self.last_exchange['caller']}\"\nThe host said: \"{self.last_exchange['host']}\""
        elif context is None:
            context = f"Currently on the line: {self.current_caller['name']}"

        # Recent show context
        recent_history = ""
        if self.show_history:
            recent_history = "\n\nRecent show moments:\n"
            for entry in self.show_history[-3:]:
                recent_history += f"- {entry['caller']}: {entry['summary'][:60]}...\n"

        response = self.openai.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": COHOST["prompt"] + recent_history},
                {"role": "user", "content": f"React to this: {context}\n\nGive a quick one-liner reaction, agreement, disagreement, or joke. ONE SENTENCE MAX."}
            ]
        )

        comment = response.choices[0].message.content
        print(f"\n  🎙️ BOBBY: {comment}")

        # Generate voice without phone filter (co-host is in studio)
        audio_gen = self.tts_client.text_to_speech.convert(
            voice_id=COHOST["voice_id"],
            text=comment,
            model_id="eleven_v3",
            output_format="pcm_24000"
        )

        audio_bytes = b"".join(audio_gen)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Record to timeline (as caller track for simplicity)
        start_time = self.get_session_time()
        self.audio_timeline.append((start_time, 'caller', audio))

        self.music.duck()
        sd.play(audio, SAMPLE_RATE)
        sd.wait()
        self.music.unduck()

    def get_producer_suggestion(self):
        """Get a suggestion from the AI producer based on show context"""
        # Build context for producer
        recent_exchanges = ""
        if self.show_history:
            for entry in self.show_history[-5:]:
                recent_exchanges += f"- {entry['caller']}: {entry['summary'][:80]}...\n"
                recent_exchanges += f"  Host said: {entry['host_said'][:60]}...\n"

        current_caller = self.current_caller["name"]
        show_length = self.get_session_time() / 60  # in minutes

        # Get list of callers we haven't heard from
        callers_heard = set(h['caller'] for h in self.show_history)
        callers_not_heard = [c['name'] for k, c in CALLERS.items() if c['name'] not in callers_heard]

        response = self.openai.chat.completions.create(
            model="gpt-5",
            messages=[{
                "role": "system",
                "content": f"""You're a radio show producer giving the host quick suggestions in their earpiece.
Keep suggestions SHORT - one line max. Be direct.

Current show info:
- Show has been running for {show_length:.1f} minutes
- Currently talking to: {current_caller}
- Callers we've heard from: {', '.join(callers_heard) if callers_heard else 'None yet'}
- Callers waiting: {', '.join(callers_not_heard[:4]) if callers_not_heard else 'All callers have called'}

Recent exchanges:
{recent_exchanges if recent_exchanges else 'Show just started'}

Give ONE tactical suggestion. Options:
- Suggest a follow-up question to ask the current caller
- Suggest switching to a different caller (and why)
- Suggest playing a sound effect (airhorn, rimshot, crickets, etc)
- Suggest a commercial break or breaking news bit
- Suggest having Bobby (co-host) chime in
- Suggest a topic pivot or callback to an earlier caller

Be brief. Example: "Ask Tony if he's confronted her yet" or "Good time for a rimshot" or "Switch to Jasmine, she'll have opinions on this"
"""
            }]
        )

        return response.choices[0].message.content

    def save_session(self):
        print("\n💾 Saving session...")

        # Stop music
        self.music.stop()

        if not self.audio_timeline and not self.music.music_audio:
            print("  No audio to save")
            return

        # Calculate total duration from timeline
        total_duration = self.get_session_time()
        total_samples = int(total_duration * SAMPLE_RATE) + SAMPLE_RATE  # +1 sec buffer

        # Create aligned track buffers
        host_track = np.zeros(total_samples, dtype=np.float32)
        caller_track = np.zeros(total_samples, dtype=np.float32)

        # Place audio segments at correct timestamps
        for start_time, track_type, audio in self.audio_timeline:
            start_sample = int(start_time * SAMPLE_RATE)
            end_sample = start_sample + len(audio)

            # Extend buffer if needed
            if end_sample > total_samples:
                extra = end_sample - total_samples + SAMPLE_RATE
                host_track = np.concatenate([host_track, np.zeros(extra, dtype=np.float32)])
                caller_track = np.concatenate([caller_track, np.zeros(extra, dtype=np.float32)])
                total_samples = len(host_track)

            if track_type == 'host':
                host_track[start_sample:end_sample] += audio
            elif track_type == 'caller':
                caller_track[start_sample:end_sample] += audio

        # Get music track (already recorded with ducking)
        music_track = None
        if self.music.music_audio:
            music_track = np.concatenate([a.flatten() for a in self.music.music_audio])
            # Pad or trim to match other tracks
            if len(music_track) < total_samples:
                music_track = np.concatenate([music_track, np.zeros(total_samples - len(music_track), dtype=np.float32)])
            else:
                music_track = music_track[:total_samples]

        # Trim silence from end
        max_len = total_samples
        for track in [host_track, caller_track, music_track]:
            if track is not None:
                nonzero = np.nonzero(np.abs(track) > 0.001)[0]
                if len(nonzero) > 0:
                    max_len = min(max_len, nonzero[-1] + SAMPLE_RATE)

        host_track = host_track[:max_len]
        caller_track = caller_track[:max_len]
        if music_track is not None:
            music_track = music_track[:max_len]

        # Apply broadcast processing to host vocal
        print("  🎙️  Processing host vocal (EQ + compression + de-ess)...")
        if np.any(host_track != 0):
            host_track = de_ess(host_track, SAMPLE_RATE)  # De-esser first
            host_track = broadcast_process(host_track, SAMPLE_RATE)

        # Save individual aligned tracks
        if np.any(host_track != 0):
            sf.write(self.output_dir / "host_track.wav", host_track, SAMPLE_RATE)
            print(f"  ✓ host_track.wav (broadcast processed)")

        if np.any(caller_track != 0):
            sf.write(self.output_dir / "caller_track.wav", caller_track, SAMPLE_RATE)
            print(f"  ✓ caller_track.wav")

        if music_track is not None:
            sf.write(self.output_dir / "music_track.wav", music_track, SAMPLE_RATE)
            print(f"  ✓ music_track.wav")

        # Create raw mixed master (full length, no edits)
        print("  🎛️  Mixing raw podcast...")
        raw_mix = np.zeros(max_len, dtype=np.float32)
        raw_mix += host_track * 1.0
        raw_mix += caller_track * 0.85
        if music_track is not None:
            raw_mix += music_track * 0.35

        # LUFS normalize to -16 LUFS (podcast standard)
        print("  📊 Normalizing to -16 LUFS...")
        raw_mix = lufs_normalize(raw_mix, SAMPLE_RATE, target_lufs=-16.0)

        sf.write(self.output_dir / "podcast_raw.wav", raw_mix, SAMPLE_RATE)
        print(f"  ✓ podcast_raw.wav (full length, -16 LUFS)")

        # Create edited mix with dead air removed
        print("  ✂️  Creating edited mix (removing dead air)...")
        edited_mix = create_edited_mix(host_track, caller_track, music_track, SAMPLE_RATE)
        edited_mix = lufs_normalize(edited_mix, SAMPLE_RATE, target_lufs=-16.0)
        sf.write(self.output_dir / "podcast_edited.wav", edited_mix, SAMPLE_RATE)
        print(f"  ✓ podcast_edited.wav (dead air removed, -16 LUFS)")

        # Duration info
        raw_mins = max_len / SAMPLE_RATE / 60
        edited_mins = len(edited_mix) / SAMPLE_RATE / 60
        saved_mins = raw_mins - edited_mins
        print(f"  📻 Raw: {raw_mins:.1f} min → Edited: {edited_mins:.1f} min (saved {saved_mins:.1f} min)")

        with open(self.output_dir / "transcript.txt", "w") as f:
            for entry in self.show_history:
                f.write(f"{entry['caller'].upper()}: {entry['summary']}\n\n")
        print(f"  ✓ transcript.txt")

        # Save persistent caller memory for future episodes
        self._save_caller_memory()
        print(f"  ✓ caller_memory.json (persistent memory saved)")

    def run(self):
        print("\n" + "=" * 60)
        print("         📻  AI RADIO SHOW - LATE NIGHT CALLERS  📻")
        print("=" * 60)
        print("\nCALLERS:")
        for i, (key, caller) in enumerate(CALLERS.items()):
            end = "\n" if (i + 1) % 2 == 0 else "  "
            print(f"  [{key}] {caller['name']:<24}", end=end)
        print("\n")

        # Start music if available
        if self.music.tracks:
            print("  🎵 Starting music...")
            if self.music.start():
                print(f"  Now playing: {self.music.get_track_name()}")
            print()

        # Start session timer for aligned audio export
        self.session_start = datetime.now()

        self.print_status()

        while True:
            try:
                cmd = input("\n> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if not cmd:
                continue

            if cmd == 'q':
                break

            if cmd in CALLER_KEYS:
                sd.stop()
                # Play hold music briefly, then ring for new caller
                play_show_sound('hold', wait=True)
                self.current_caller = CALLERS[cmd]
                self.conversation_history = []
                play_show_sound('ring', wait=True)
                # Play caller's stinger if they have one
                if not play_caller_stinger(cmd, wait=True):
                    pass  # No stinger, that's fine
                print(f"\n  📞 NEW CALLER: {self.current_caller['name']}")
                quality = self.current_caller.get("phone_quality", "normal")
                if quality != "good":
                    print(f"  📶 Connection quality: {quality}")
                self.print_status()
                continue

            if cmd == 'h':
                sd.stop()
                play_show_sound('hangup', wait=False)
                self.conversation_history = []
                print(f"\n  🔇 HUNG UP on {self.current_caller['name']}!")
                print("  Pick a new caller [1-9, 0, -, =]")
                continue

            # Music controls
            if cmd == 'm':
                if self.music.playing:
                    self.music.stop()
                    print("  🔇 Music stopped")
                else:
                    if self.music.start():
                        print(f"  🎵 Music started: {self.music.get_track_name()}")
                    else:
                        print("  No music files in music/")
                continue

            if cmd == 'n':
                if self.music.tracks:
                    track = self.music.next_track()
                    print(f"  🎵 Now playing: {track}")
                else:
                    print("  No music files")
                continue

            if cmd == '+' or cmd == 'vol+':
                self.music.set_volume(self.music.volume + 0.05)
                print(f"  🔊 Volume: {int(self.music.volume * 100)}%")
                continue

            if cmd == 'vol-':
                self.music.set_volume(self.music.volume - 0.05)
                print(f"  🔉 Volume: {int(self.music.volume * 100)}%")
                continue

            if cmd == 'f':
                # Fade out music (for taking a call)
                self.music.fade_out()
                print("  🔉 Music fading out...")
                continue

            if cmd == 'g':
                # Fade music back in (after a call)
                self.music.fade_in()
                print("  🔊 Music fading in...")
                continue

            if cmd == 'd':
                # Toggle auto-duck
                auto = self.music.toggle_auto_duck()
                print(f"  Auto-duck: {'ON' if auto else 'OFF'}")
                continue

            if cmd == 'ad' or cmd == 'commercial':
                self.play_commercial()
                continue

            if cmd == 'news':
                self.play_breaking_news()
                continue

            if cmd == 'b' or cmd == 'bobby':
                self.cohost_chime_in()
                continue

            if cmd == 'stingers':
                self.generate_caller_stingers()
                continue

            if cmd == 'p' or cmd == 'producer':
                suggestion = self.get_producer_suggestion()
                print(f"\n  🎧 PRODUCER: {suggestion}\n")
                continue

            if cmd == 'rec':
                audio = self.record_audio()
                if audio is not None and len(audio) > SAMPLE_RATE * 0.5:
                    print("  📝 Transcribing...")
                    text = self.transcribe(audio)
                    if text:
                        print(f"\n  YOU: {text}")
                        print(f"\n  💭 {self.current_caller['name']} is thinking...")
                        reply = self.generate_response(text)
                        print(f"\n  📞 {self.current_caller['name'].upper()}: {reply}\n")
                        self.speak(reply)
                    else:
                        print("  (No speech detected)")
                else:
                    print("  (Recording too short)")
                continue

            if cmd == 't':
                self.music.duck()  # Duck music while typing too
                text = input("  Type: ").strip()
                if text:
                    print(f"\n  💭 {self.current_caller['name']} is thinking...")
                    reply = self.generate_response(text)
                    print(f"\n  📞 {self.current_caller['name'].upper()}: {reply}\n")
                    self.speak(reply)
                else:
                    self.music.unduck()
                continue

            # Sound effects
            if len(cmd) == 1 and cmd in SOUNDBOARD:
                if play_sound(cmd):
                    name = SOUNDBOARD[cmd].replace('.wav', '').replace('_', ' ')
                    print(f"  🔊 {name}")
                else:
                    print(f"  Sound file not found")
                continue

            print("  Commands: rec, t, h, m, n, +/vol-, 1-9/0/-/=, sounds, q")

        self.save_session()
        print("\n🎬 That's a wrap! Thanks for listening.\n")


if __name__ == "__main__":
    show = RadioShow()
    show.run()
