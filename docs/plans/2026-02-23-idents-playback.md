# Idents Playback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an idents section that loads MP3s from `idents/` and plays them through the ads channel (ch 11), with a separate "idents" stem for post-production.

**Architecture:** Mirrors the existing ads system — dropdown + play/stop buttons, same audio channel, mutually exclusive with ads. Idents get their own stem in stem_recorder so they can be mixed independently in post-production.

**Tech Stack:** Python (FastAPI), sounddevice, librosa, vanilla JS

---

### Task 1: Add idents_dir to config

**Files:**
- Modify: `backend/config.py:46-47`

**Step 1: Add idents_dir path**

After `ads_dir` (line 46), add:

```python
    idents_dir: Path = base_dir / "idents"
```

**Step 2: Create the idents directory**

```bash
mkdir -p idents
```

**Step 3: Commit**

```bash
git add backend/config.py
git commit -m "Add idents_dir to config"
```

---

### Task 2: Add "idents" stem to stem_recorder

**Files:**
- Modify: `backend/services/stem_recorder.py:10`

**Step 1: Add "idents" to STEM_NAMES**

Change line 10 from:

```python
STEM_NAMES = ["host", "caller", "music", "sfx", "ads"]
```

to:

```python
STEM_NAMES = ["host", "caller", "music", "sfx", "ads", "idents"]
```

**Step 2: Add "idents" to postprod.py STEM_NAMES**

In `postprod.py:20`, change:

```python
STEM_NAMES = ["host", "caller", "music", "sfx", "ads"]
```

to:

```python
STEM_NAMES = ["host", "caller", "music", "sfx", "ads", "idents"]
```

Also update `postprod.py:72` — the `remove_gaps` content detection line — add idents:

```python
content = stems["host"] + stems["caller"] + stems["sfx"] + stems["ads"] + stems["idents"]
```

And in `mix_stems` (line 411), add idents level:

```python
levels = {"host": 0, "caller": 0, "music": -6, "sfx": -10, "ads": 0, "idents": 0}
```

And in stereo pans (line 420):

```python
pans = {"host": 0.0, "caller": 0.15, "music": 0.0, "sfx": 0.0, "ads": 0.0, "idents": 0.0}
```

And in `match_voice_levels` (line 389), add "idents":

```python
for name in ["host", "caller", "ads", "idents"]:
```

And in gap removal limiter section (line 777-778):

```python
for name in ["ads", "sfx", "idents"]:
```

**Step 3: Commit**

```bash
git add backend/services/stem_recorder.py postprod.py
git commit -m "Add idents stem to recorder and postprod"
```

---

### Task 3: Add play_ident / stop_ident to audio service

**Files:**
- Modify: `backend/services/audio.py`

**Step 1: Add ident state vars to __init__ (after line 40)**

After the ad playback state block (lines 35-40), add:

```python
        # Ident playback state
        self._ident_stream: Optional[sd.OutputStream] = None
        self._ident_data: Optional[np.ndarray] = None
        self._ident_resampled: Optional[np.ndarray] = None
        self._ident_position: int = 0
        self._ident_playing: bool = False
```

**Step 2: Add play_ident method (after stop_ad, ~line 1006)**

Insert after `stop_ad` method. This is a copy of `play_ad` with:
- `_ad_*` → `_ident_*`
- Calls `self.stop_ad()` at the start (mutual exclusion)
- Stem recording writes to `"idents"` instead of `"ads"`

```python
    def play_ident(self, file_path: str):
        """Load and play an ident file once (no loop) on the ad channel"""
        import librosa

        path = Path(file_path)
        if not path.exists():
            print(f"Ident file not found: {file_path}")
            return

        self.stop_ident()
        self.stop_ad()

        try:
            audio, sr = librosa.load(str(path), sr=self.output_sample_rate, mono=True)
            self._ident_data = audio.astype(np.float32)
        except Exception as e:
            print(f"Failed to load ident: {e}")
            return

        self._ident_playing = True
        self._ident_position = 0

        if self.output_device is None:
            num_channels = 2
            device = None
            device_sr = self.output_sample_rate
            channel_idx = 0
        else:
            device_info = sd.query_devices(self.output_device)
            num_channels = device_info['max_output_channels']
            device_sr = int(device_info['default_samplerate'])
            device = self.output_device
            channel_idx = min(self.ad_channel, num_channels) - 1

        if self.output_sample_rate != device_sr:
            self._ident_resampled = librosa.resample(
                self._ident_data, orig_sr=self.output_sample_rate, target_sr=device_sr
            ).astype(np.float32)
        else:
            self._ident_resampled = self._ident_data

        def callback(outdata, frames, time_info, status):
            outdata[:] = 0
            if not self._ident_playing or self._ident_resampled is None:
                return

            remaining = len(self._ident_resampled) - self._ident_position
            if remaining >= frames:
                chunk = self._ident_resampled[self._ident_position:self._ident_position + frames]
                outdata[:, channel_idx] = chunk
                if self.stem_recorder:
                    self.stem_recorder.write_sporadic("idents", chunk.copy(), device_sr)
                self._ident_position += frames
            else:
                if remaining > 0:
                    outdata[:remaining, channel_idx] = self._ident_resampled[self._ident_position:]
                self._ident_playing = False

        try:
            self._ident_stream = sd.OutputStream(
                device=device,
                channels=num_channels,
                samplerate=device_sr,
                dtype=np.float32,
                callback=callback,
                blocksize=2048
            )
            self._ident_stream.start()
            print(f"Ident playback started on ch {self.ad_channel} @ {device_sr}Hz")
        except Exception as e:
            print(f"Ident playback error: {e}")
            self._ident_playing = False

    def stop_ident(self):
        """Stop ident playback"""
        self._ident_playing = False
        if self._ident_stream:
            self._ident_stream.stop()
            self._ident_stream.close()
            self._ident_stream = None
        self._ident_position = 0
```

**Step 3: Add `self.stop_ident()` to top of play_ad (line 935)**

In `play_ad`, after `self.stop_ad()` (line 935), add:

```python
        self.stop_ident()
```

**Step 4: Commit**

```bash
git add backend/services/audio.py
git commit -m "Add play_ident/stop_ident to audio service"
```

---

### Task 4: Add idents API endpoints

**Files:**
- Modify: `backend/main.py` (after ads endpoints, ~line 4362)

**Step 1: Add IDENT_DISPLAY_NAMES and endpoints**

Insert after the ads stop endpoint (line 4362):

```python

# --- Idents Endpoints ---

IDENT_DISPLAY_NAMES = {}


@app.get("/api/idents")
async def get_idents():
    """Get available ident tracks, shuffled"""
    ident_list = []
    if settings.idents_dir.exists():
        for ext in ['*.wav', '*.mp3', '*.flac']:
            for f in settings.idents_dir.glob(ext):
                ident_list.append({
                    "name": IDENT_DISPLAY_NAMES.get(f.stem, f.stem),
                    "file": f.name,
                    "path": str(f)
                })
    random.shuffle(ident_list)
    return {"idents": ident_list}


@app.post("/api/idents/play")
async def play_ident(request: MusicRequest):
    """Play an ident once on the ad channel (ch 11)"""
    ident_path = settings.idents_dir / request.track
    if not ident_path.exists():
        raise HTTPException(404, "Ident not found")

    if audio_service._music_playing:
        audio_service.stop_music(fade_duration=1.0)
        await asyncio.sleep(1.1)
    audio_service.play_ident(str(ident_path))
    return {"status": "playing", "track": request.track}


@app.post("/api/idents/stop")
async def stop_ident():
    """Stop ident playback"""
    audio_service.stop_ident()
    return {"status": "stopped"}
```

**Step 2: Commit**

```bash
git add backend/main.py
git commit -m "Add idents API endpoints"
```

---

### Task 5: Add idents UI section and JS functions

**Files:**
- Modify: `frontend/index.html:113` (after ads section)
- Modify: `frontend/js/app.js`

**Step 1: Add Idents HTML section**

After the Ads section closing `</section>` (line 113), add:

```html
            <!-- Idents -->
            <section class="music-section">
                <h2>Idents</h2>
                <select id="ident-select"></select>
                <div class="music-controls">
                    <button id="ident-play-btn">Play Ident</button>
                    <button id="ident-stop-btn">Stop</button>
                </div>
            </section>
```

**Step 2: Add loadIdents, playIdent, stopIdent to app.js**

After `stopAd()` function (~line 773), add:

```javascript
async function loadIdents() {
    try {
        const res = await fetch('/api/idents');
        const data = await res.json();
        const idents = data.idents || [];

        const select = document.getElementById('ident-select');
        if (!select) return;

        const previousValue = select.value;
        select.innerHTML = '';

        idents.forEach(ident => {
            const option = document.createElement('option');
            option.value = ident.file;
            option.textContent = ident.name;
            select.appendChild(option);
        });

        if (previousValue && [...select.options].some(o => o.value === previousValue)) {
            select.value = previousValue;
        }

        console.log('Loaded', idents.length, 'idents');
    } catch (err) {
        console.error('loadIdents error:', err);
    }
}

async function playIdent() {
    await loadIdents();
    const select = document.getElementById('ident-select');
    const track = select?.value;
    if (!track) return;

    await fetch('/api/idents/play', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ track, action: 'play' })
    });
}

async function stopIdent() {
    await fetch('/api/idents/stop', { method: 'POST' });
}
```

**Step 3: Add event listeners in initEventListeners**

After the ads event listeners (line 190), add:

```javascript
    // Idents
    document.getElementById('ident-play-btn')?.addEventListener('click', playIdent);
    document.getElementById('ident-stop-btn')?.addEventListener('click', stopIdent);
```

**Step 4: Add loadIdents() to DOMContentLoaded init**

After `await loadAds();` (line 59), add:

```javascript
        await loadIdents();
```

**Step 5: Bump cache buster on app.js script tag**

In `index.html:243`, change `?v=17` to `?v=18`.

**Step 6: Commit**

```bash
git add frontend/index.html frontend/js/app.js
git commit -m "Add idents UI section and JS functions"
```
