/**
 * Call-In Page — Browser WebSocket audio streaming
 * Captures mic via AudioWorklet, sends Int16 PCM 16kHz mono over WebSocket.
 * Receives Int16 PCM 16kHz mono back for playback.
 */

let ws = null;
let audioCtx = null;
let micStream = null;
let workletNode = null;
let playbackNode = null;
let callerId = null;

const callBtn = document.getElementById('call-btn');
const hangupBtn = document.getElementById('hangup-btn');
const statusEl = document.getElementById('status');
const statusText = document.getElementById('status-text');
const nameInput = document.getElementById('caller-name');
const micMeter = document.getElementById('mic-meter');
const micMeterFill = document.getElementById('mic-meter-fill');

callBtn.addEventListener('click', startCall);
hangupBtn.addEventListener('click', hangUp);
nameInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') startCall();
});

async function startCall() {
    const name = nameInput.value.trim() || 'Anonymous';
    callBtn.disabled = true;
    setStatus('Connecting...', false);

    try {
        // Get mic access
        micStream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 }
        });

        // Set up AudioContext
        audioCtx = new AudioContext({ sampleRate: 48000 });

        // Register worklet processors inline via blob
        const processorCode = `
// --- Capture processor: downsample to 16kHz, emit small chunks ---
class CallerProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = [];
        this.targetSamples = 640; // 40ms at 16kHz — low latency
    }
    process(inputs) {
        const input = inputs[0][0];
        if (!input) return true;

        const ratio = sampleRate / 16000;
        for (let i = 0; i < input.length; i += ratio) {
            const idx = Math.floor(i);
            if (idx < input.length) {
                this.buffer.push(input[idx]);
            }
        }

        while (this.buffer.length >= this.targetSamples) {
            const chunk = this.buffer.splice(0, this.targetSamples);
            const int16 = new Int16Array(chunk.length);
            for (let i = 0; i < chunk.length; i++) {
                const s = Math.max(-1, Math.min(1, chunk[i]));
                int16[i] = s < 0 ? s * 32768 : s * 32767;
            }
            this.port.postMessage(int16.buffer, [int16.buffer]);
        }
        return true;
    }
}
registerProcessor('caller-processor', CallerProcessor);

// --- Playback processor: ring buffer with 16kHz->sampleRate upsampling ---
class PlaybackProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.ringSize = 16000 * 3; // 3s ring buffer at 16kHz
        this.ring = new Float32Array(this.ringSize);
        this.writePos = 0;
        this.readPos = 0;
        this.available = 0;
        this.started = false;
        this.jitterMs = 80; // buffer 80ms before starting playback
        this.jitterSamples = Math.floor(16000 * this.jitterMs / 1000);

        this.port.onmessage = (e) => {
            const data = e.data;
            for (let i = 0; i < data.length; i++) {
                this.ring[this.writePos] = data[i];
                this.writePos = (this.writePos + 1) % this.ringSize;
            }
            this.available += data.length;
            if (this.available > this.ringSize) {
                // Overflow — skip ahead
                this.available = this.ringSize;
                this.readPos = (this.writePos - this.ringSize + this.ringSize) % this.ringSize;
            }
        };
    }
    process(inputs, outputs) {
        const output = outputs[0][0];
        if (!output) return true;

        // Wait for jitter buffer to fill before starting
        if (!this.started) {
            if (this.available < this.jitterSamples) {
                output.fill(0);
                return true;
            }
            this.started = true;
        }

        const ratio = 16000 / sampleRate;
        const srcNeeded = Math.ceil(output.length * ratio);

        if (this.available >= srcNeeded) {
            for (let i = 0; i < output.length; i++) {
                const srcPos = i * ratio;
                const idx = Math.floor(srcPos);
                const frac = srcPos - idx;
                const p0 = (this.readPos + idx) % this.ringSize;
                const p1 = (p0 + 1) % this.ringSize;
                output[i] = this.ring[p0] * (1 - frac) + this.ring[p1] * frac;
            }
            this.readPos = (this.readPos + srcNeeded) % this.ringSize;
            this.available -= srcNeeded;
        } else {
            // Underrun — silence, reset jitter buffer
            output.fill(0);
            this.started = false;
        }
        return true;
    }
}
registerProcessor('playback-processor', PlaybackProcessor);
`;
        const blob = new Blob([processorCode], { type: 'application/javascript' });
        const blobUrl = URL.createObjectURL(blob);
        await audioCtx.audioWorklet.addModule(blobUrl);
        URL.revokeObjectURL(blobUrl);

        // Connect mic to worklet
        const source = audioCtx.createMediaStreamSource(micStream);
        workletNode = new AudioWorkletNode(audioCtx, 'caller-processor');

        // Connect WebSocket
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${proto}//${location.host}/api/caller/stream`);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            ws.send(JSON.stringify({ type: 'join', name }));
        };

        ws.onmessage = (event) => {
            if (typeof event.data === 'string') {
                handleControlMessage(JSON.parse(event.data));
            } else {
                handleAudioData(event.data);
            }
        };

        ws.onclose = () => {
            setStatus('Disconnected', false);
            cleanup();
        };

        ws.onerror = () => {
            setStatus('Connection error', false);
            cleanup();
        };

        // Forward mic audio to WebSocket
        workletNode.port.onmessage = (e) => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(e.data);
            }
        };

        source.connect(workletNode);
        // Don't connect worklet to destination — we don't want to hear our own mic

        // Set up playback worklet for received audio
        playbackNode = new AudioWorkletNode(audioCtx, 'playback-processor');
        playbackNode.connect(audioCtx.destination);

        // Show mic meter
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);
        startMicMeter(analyser);

        // UI
        nameInput.disabled = true;
        hangupBtn.style.display = 'block';

    } catch (err) {
        console.error('Call error:', err);
        setStatus('Failed: ' + err.message, false);
        callBtn.disabled = false;
        cleanup();
    }
}

function handleControlMessage(msg) {
    if (msg.status === 'queued') {
        callerId = msg.caller_id;
        setStatus(`Waiting in queue (position ${msg.position})...`, false);
    } else if (msg.status === 'on_air') {
        setStatus('ON AIR', true);
    } else if (msg.status === 'disconnected') {
        setStatus('Disconnected', false);
        cleanup();
    }
}

function handleAudioData(buffer) {
    if (!playbackNode) return;

    // Convert Int16 PCM to Float32 and send to playback worklet
    const int16 = new Int16Array(buffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
        float32[i] = int16[i] / 32768;
    }
    playbackNode.port.postMessage(float32, [float32.buffer]);
}

function hangUp() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
    }
    setStatus('Disconnected', false);
    cleanup();
}

function cleanup() {
    if (workletNode) {
        workletNode.disconnect();
        workletNode = null;
    }
    if (playbackNode) {
        playbackNode.disconnect();
        playbackNode = null;
    }
    if (micStream) {
        micStream.getTracks().forEach(t => t.stop());
        micStream = null;
    }
    if (audioCtx) {
        audioCtx.close().catch(() => {});
        audioCtx = null;
    }
    ws = null;
    callerId = null;
    callBtn.disabled = false;
    nameInput.disabled = false;
    hangupBtn.style.display = 'none';
    micMeter.classList.remove('visible');
}

function setStatus(text, isOnAir) {
    statusEl.classList.add('visible');
    statusText.textContent = text;
    if (isOnAir) {
        statusEl.classList.add('on-air');
    } else {
        statusEl.classList.remove('on-air');
    }
}

function startMicMeter(analyser) {
    micMeter.classList.add('visible');
    const data = new Uint8Array(analyser.frequencyBinCount);

    function update() {
        if (!analyser || !audioCtx) return;
        analyser.getByteFrequencyData(data);
        let sum = 0;
        for (let i = 0; i < data.length; i++) sum += data[i];
        const avg = sum / data.length;
        const pct = Math.min(100, (avg / 128) * 100);
        micMeterFill.style.width = pct + '%';
        requestAnimationFrame(update);
    }
    update();
}
