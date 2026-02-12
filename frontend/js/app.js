/**
 * AI Radio Show - Control Panel (Server-Side Audio)
 */

// --- State ---
let currentCaller = null;
let isProcessing = false;
let isRecording = false;
let phoneFilter = false;
let autoScroll = true;
let logPollInterval = null;
let lastLogCount = 0;

// Track lists
let tracks = [];
let sounds = [];


// --- Safe JSON parsing ---
async function safeFetch(url, options = {}, timeoutMs = 30000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const res = await fetch(url, { ...options, signal: controller.signal });
        clearTimeout(timer);
        if (!res.ok) {
            const text = await res.text();
            let detail = text;
            try { detail = JSON.parse(text).detail || text; } catch {}
            throw new Error(detail);
        }
        const text = await res.text();
        if (!text) return {};
        return JSON.parse(text);
    } catch (err) {
        clearTimeout(timer);
        if (err.name === 'AbortError') throw new Error('Request timed out');
        throw err;
    }
}


// --- Init ---
document.addEventListener('DOMContentLoaded', async () => {
    console.log('AI Radio Show initializing...');
    try {
        await loadAudioDevices();
        await loadCallers();
        await loadMusic();
        await loadAds();
        await loadSounds();
        await loadSettings();
        initEventListeners();
        log('Ready. Configure audio devices in Settings, then click a caller to start.');
        console.log('AI Radio Show ready');
    } catch (err) {
        console.error('Init error:', err);
        log('Error loading: ' + err.message);
    }
});


function initEventListeners() {
    // Hangup
    document.getElementById('hangup-btn')?.addEventListener('click', hangup);

    // New Session
    document.getElementById('new-session-btn')?.addEventListener('click', newSession);

    // On-Air + Recording (linked — toggling one toggles both)
    const onAirBtn = document.getElementById('on-air-btn');
    const recBtn = document.getElementById('rec-btn');
    let stemRecording = false;

    function updateRecBtn(recording) {
        stemRecording = recording;
        if (recBtn) {
            recBtn.classList.toggle('recording', recording);
            recBtn.textContent = recording ? '⏺ REC' : 'REC';
        }
    }

    if (onAirBtn) {
        fetch('/api/on-air').then(r => r.json()).then(data => {
            updateOnAirBtn(onAirBtn, data.on_air);
            updateRecBtn(data.recording);
        });
        onAirBtn.addEventListener('click', async () => {
            const isOn = onAirBtn.classList.contains('on');
            const res = await safeFetch('/api/on-air', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ on_air: !isOn }),
            });
            updateOnAirBtn(onAirBtn, res.on_air);
            updateRecBtn(res.recording);
            log(res.on_air ? 'Show is ON AIR + Recording' : 'Show is OFF AIR + Recording stopped');
        });
    }

    if (recBtn) {
        recBtn.addEventListener('click', async () => {
            try {
                const res = await safeFetch('/api/recording/toggle', { method: 'POST' });
                updateRecBtn(res.recording);
                if (onAirBtn) updateOnAirBtn(onAirBtn, res.on_air);
                log(res.recording ? 'Recording started + ON AIR' : 'Recording stopped + OFF AIR');
            } catch (err) {
                log('Recording error: ' + err.message);
            }
        });
    }

    // Export session
    document.getElementById('export-session-btn')?.addEventListener('click', exportSession);

    // Server controls
    document.getElementById('restart-server-btn')?.addEventListener('click', restartServer);
    document.getElementById('stop-server-btn')?.addEventListener('click', stopServer);
    document.getElementById('auto-scroll')?.addEventListener('change', e => {
        autoScroll = e.target.checked;
    });

    // Start log polling
    startLogPolling();

    // Start queue polling
    startQueuePolling();

    // Talk button - now triggers server-side recording
    const talkBtn = document.getElementById('talk-btn');
    if (talkBtn) {
        talkBtn.addEventListener('mousedown', startRecording);
        talkBtn.addEventListener('mouseup', stopRecording);
        talkBtn.addEventListener('mouseleave', () => { if (isRecording) stopRecording(); });
        talkBtn.addEventListener('touchstart', e => { e.preventDefault(); startRecording(); });
        talkBtn.addEventListener('touchend', e => { e.preventDefault(); stopRecording(); });
    }

    // Type button
    document.getElementById('type-btn')?.addEventListener('click', () => {
        document.getElementById('type-modal')?.classList.remove('hidden');
        document.getElementById('type-input')?.focus();
    });
    document.getElementById('send-type')?.addEventListener('click', sendTypedMessage);
    document.getElementById('close-type')?.addEventListener('click', () => {
        document.getElementById('type-modal')?.classList.add('hidden');
    });
    document.getElementById('type-input')?.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendTypedMessage();
        }
    });

    // Music - now server-side
    document.getElementById('play-btn')?.addEventListener('click', playMusic);
    document.getElementById('stop-btn')?.addEventListener('click', stopMusic);
    document.getElementById('volume')?.addEventListener('input', setMusicVolume);

    // Ads
    document.getElementById('ad-play-btn')?.addEventListener('click', playAd);
    document.getElementById('ad-stop-btn')?.addEventListener('click', stopAd);

    // Settings
    document.getElementById('settings-btn')?.addEventListener('click', async () => {
        document.getElementById('settings-modal')?.classList.remove('hidden');
        await loadSettings();  // Reload settings when modal opens
    });
    document.getElementById('close-settings')?.addEventListener('click', () => {
        document.getElementById('settings-modal')?.classList.add('hidden');
    });
    document.getElementById('save-settings')?.addEventListener('click', saveSettings);
    document.getElementById('provider')?.addEventListener('change', updateProviderUI);
    document.getElementById('phone-filter')?.addEventListener('change', e => {
        phoneFilter = e.target.checked;
    });
    document.getElementById('refresh-ollama')?.addEventListener('click', refreshOllamaModels);

    // Real caller hangup
    document.getElementById('hangup-real-btn')?.addEventListener('click', async () => {
        await fetch('/api/hangup/real', { method: 'POST' });
        hideRealCaller();
        log('Real caller disconnected');
    });

    // AI caller hangup (small button in AI caller panel)
    document.getElementById('hangup-ai-btn')?.addEventListener('click', hangup);

    // AI respond button (manual trigger)
    document.getElementById('ai-respond-btn')?.addEventListener('click', triggerAiRespond);

    // Start conversation update polling
    startConversationPolling();

    // AI respond mode toggle
    document.getElementById('mode-manual')?.addEventListener('click', () => {
        document.getElementById('mode-manual')?.classList.add('active');
        document.getElementById('mode-auto')?.classList.remove('active');
        document.getElementById('ai-respond-btn')?.classList.remove('hidden');
        fetch('/api/session/ai-mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'manual' }),
        });
    });

    document.getElementById('mode-auto')?.addEventListener('click', () => {
        document.getElementById('mode-auto')?.classList.add('active');
        document.getElementById('mode-manual')?.classList.remove('active');
        fetch('/api/session/ai-mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'auto' }),
        });
    });

    // Auto follow-up toggle
    document.getElementById('auto-followup')?.addEventListener('change', (e) => {
        fetch('/api/session/auto-followup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: e.target.checked }),
        });
    });
}


async function refreshOllamaModels() {
    const btn = document.getElementById('refresh-ollama');
    const select = document.getElementById('ollama-model');
    if (!select) return;

    btn.textContent = 'Loading...';
    btn.disabled = true;

    try {
        const res = await fetch('/api/settings');
        const data = await res.json();

        select.innerHTML = '';
        const models = data.available_ollama_models || [];

        if (models.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = '(No models found)';
            select.appendChild(option);
        } else {
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                select.appendChild(option);
            });
        }
    } catch (err) {
        console.error('Failed to refresh Ollama models:', err);
    }

    btn.textContent = 'Refresh Models';
    btn.disabled = false;
}


// --- Audio Devices ---
async function loadAudioDevices() {
    try {
        const res = await fetch('/api/audio/devices');
        const data = await res.json();

        const inputSelect = document.getElementById('input-device');
        const outputSelect = document.getElementById('output-device');

        if (!inputSelect || !outputSelect) return;

        // Clear selects
        inputSelect.innerHTML = '<option value="">-- Select --</option>';
        outputSelect.innerHTML = '<option value="">-- Select --</option>';

        data.devices.forEach(device => {
            // Input devices
            if (device.inputs > 0) {
                const opt = document.createElement('option');
                opt.value = device.id;
                opt.textContent = `${device.name} (${device.inputs} ch)`;
                inputSelect.appendChild(opt);
            }
            // Output devices
            if (device.outputs > 0) {
                const opt = document.createElement('option');
                opt.value = device.id;
                opt.textContent = `${device.name} (${device.outputs} ch)`;
                outputSelect.appendChild(opt);
            }
        });

        // Load current settings
        const settingsRes = await fetch('/api/audio/settings');
        const settings = await settingsRes.json();

        if (settings.input_device !== null)
            inputSelect.value = settings.input_device;
        if (settings.output_device !== null)
            outputSelect.value = settings.output_device;

        // Channel settings
        const inputCh = document.getElementById('input-channel');
        const callerCh = document.getElementById('caller-channel');
        const liveCallerCh = document.getElementById('live-caller-channel');
        const musicCh = document.getElementById('music-channel');
        const sfxCh = document.getElementById('sfx-channel');
        const adCh = document.getElementById('ad-channel');

        if (inputCh) inputCh.value = settings.input_channel || 1;
        if (callerCh) callerCh.value = settings.caller_channel || 1;
        if (liveCallerCh) liveCallerCh.value = settings.live_caller_channel || 9;
        if (musicCh) musicCh.value = settings.music_channel || 2;
        if (sfxCh) sfxCh.value = settings.sfx_channel || 3;
        if (adCh) adCh.value = settings.ad_channel || 11;

        // Phone filter setting
        const phoneFilterEl = document.getElementById('phone-filter');
        if (phoneFilterEl) {
            phoneFilterEl.checked = settings.phone_filter ?? false;
            phoneFilter = phoneFilterEl.checked;
        }

        console.log('Audio devices loaded');
    } catch (err) {
        console.error('loadAudioDevices error:', err);
    }
}


async function saveAudioDevices() {
    const inputDevice = document.getElementById('input-device')?.value;
    const outputDevice = document.getElementById('output-device')?.value;
    const inputChannel = document.getElementById('input-channel')?.value;
    const callerChannel = document.getElementById('caller-channel')?.value;
    const liveCallerChannel = document.getElementById('live-caller-channel')?.value;
    const musicChannel = document.getElementById('music-channel')?.value;
    const sfxChannel = document.getElementById('sfx-channel')?.value;
    const adChannel = document.getElementById('ad-channel')?.value;
    const phoneFilterChecked = document.getElementById('phone-filter')?.checked ?? false;

    await fetch('/api/audio/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            input_device: inputDevice ? parseInt(inputDevice) : null,
            input_channel: inputChannel ? parseInt(inputChannel) : 1,
            output_device: outputDevice ? parseInt(outputDevice) : null,
            caller_channel: callerChannel ? parseInt(callerChannel) : 1,
            live_caller_channel: liveCallerChannel ? parseInt(liveCallerChannel) : 9,
            music_channel: musicChannel ? parseInt(musicChannel) : 2,
            sfx_channel: sfxChannel ? parseInt(sfxChannel) : 3,
            ad_channel: adChannel ? parseInt(adChannel) : 11,
            phone_filter: phoneFilterChecked
        })
    });

    // Update local state
    phoneFilter = phoneFilterChecked;

    log('Audio routing saved');
}


// --- Callers ---
async function loadCallers() {
    try {
        const res = await fetch('/api/callers');
        const data = await res.json();

        const grid = document.getElementById('callers');
        if (!grid) return;
        grid.innerHTML = '';

        data.callers.forEach(caller => {
            const btn = document.createElement('button');
            btn.className = 'caller-btn';
            if (caller.returning) btn.classList.add('returning');
            btn.textContent = caller.returning ? `\u2605 ${caller.name}` : caller.name;
            btn.dataset.key = caller.key;
            btn.addEventListener('click', () => startCall(caller.key, caller.name));
            grid.appendChild(btn);
        });

        // Show session ID
        const sessionEl = document.getElementById('session-id');
        if (sessionEl && data.session_id) {
            sessionEl.textContent = `(${data.session_id})`;
        }

        console.log('Loaded', data.callers.length, 'callers, session:', data.session_id);
    } catch (err) {
        console.error('loadCallers error:', err);
    }
}


async function startCall(key, name) {
    if (isProcessing) return;

    const res = await fetch(`/api/call/${key}`, { method: 'POST' });
    const data = await res.json();

    currentCaller = { key, name };

    // Check if real caller is active (three-way scenario)
    const realCallerActive = document.getElementById('real-caller-info') &&
        !document.getElementById('real-caller-info').classList.contains('hidden');

    if (realCallerActive) {
        document.getElementById('call-status').textContent = `Three-way: ${name} (AI) + Real Caller`;
    } else {
        document.getElementById('call-status').textContent = `On call: ${name}`;
    }

    document.getElementById('hangup-btn').disabled = false;

    // Show AI caller in active call indicator
    const aiInfo = document.getElementById('ai-caller-info');
    const aiName = document.getElementById('ai-caller-name');
    if (aiInfo) aiInfo.classList.remove('hidden');
    if (aiName) aiName.textContent = name;

    // Show caller background in disclosure triangle
    const bgDetails = document.getElementById('caller-background-details');
    const bgEl = document.getElementById('caller-background');
    if (bgDetails && bgEl && data.background) {
        bgEl.textContent = data.background;
        bgDetails.classList.remove('hidden');
    }

    document.querySelectorAll('.caller-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.key === key);
    });

    log(`Connected to ${name}` + (realCallerActive ? ' (three-way)' : ''));
    if (!realCallerActive) clearChat();

    updateActiveCallIndicator();
}


async function newSession() {
    // Hangup if on a call
    if (currentCaller) {
        await hangup();
    }

    await fetch('/api/session/reset', { method: 'POST' });
    conversationSince = 0;

    // Hide caller background
    const bgDetails = document.getElementById('caller-background-details');
    if (bgDetails) bgDetails.classList.add('hidden');

    // Reload callers to get new session ID
    await loadCallers();

    log('New session started - all callers have fresh backgrounds');
}


async function hangup() {
    if (!currentCaller) return;

    await fetch('/api/tts/stop', { method: 'POST' });
    await fetch('/api/hangup', { method: 'POST' });

    log(`Hung up on ${currentCaller.name}`);

    currentCaller = null;
    isProcessing = false;
    hideStatus();

    document.getElementById('call-status').textContent = 'No active call';
    document.getElementById('hangup-btn').disabled = true;
    document.querySelectorAll('.caller-btn').forEach(btn => btn.classList.remove('active'));

    // Hide caller background
    const bgDetails2 = document.getElementById('caller-background-details');
    if (bgDetails2) bgDetails2.classList.add('hidden');

    // Hide AI caller indicator
    document.getElementById('ai-caller-info')?.classList.add('hidden');
    updateActiveCallIndicator();
}


// --- Server-Side Recording ---
async function startRecording() {
    if (!currentCaller || isProcessing) return;

    try {
        const res = await fetch('/api/record/start', { method: 'POST' });
        if (!res.ok) {
            const err = await res.json();
            log('Record error: ' + (err.detail || 'Failed to start'));
            return;
        }

        isRecording = true;
        document.getElementById('talk-btn').classList.add('recording');
        document.getElementById('talk-btn').textContent = 'Recording...';

    } catch (err) {
        log('Record error: ' + err.message);
    }
}


async function stopRecording() {
    if (!isRecording) return;

    document.getElementById('talk-btn').classList.remove('recording');
    document.getElementById('talk-btn').textContent = 'Hold to Talk';

    isRecording = false;
    isProcessing = true;
    showStatus('Processing...');

    try {
        // Stop recording and get transcription
        const data = await safeFetch('/api/record/stop', { method: 'POST' });

        if (!data.text) {
            log('(No speech detected)');
            isProcessing = false;
            hideStatus();
            return;
        }

        addMessage('You', data.text);

        // Chat
        showStatus(`${currentCaller.name} is thinking...`);

        const chatData = await safeFetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: data.text })
        });

        addMessage(chatData.caller, chatData.text);

        // TTS (plays on server) - only if we have text
        if (chatData.text && chatData.text.trim()) {
            showStatus(`${currentCaller.name} is speaking...`);

            await safeFetch('/api/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: chatData.text,
                    voice_id: chatData.voice_id,
                    phone_filter: phoneFilter
                })
            });
        }

    } catch (err) {
        log('Error: ' + err.message);
    }

    isProcessing = false;
    hideStatus();
}


async function sendTypedMessage() {
    const input = document.getElementById('type-input');
    const text = input.value.trim();
    if (!text || !currentCaller || isProcessing) return;

    input.value = '';
    document.getElementById('type-modal').classList.add('hidden');

    isProcessing = true;
    addMessage('You', text);

    try {
        showStatus(`${currentCaller.name} is thinking...`);

        const chatData = await safeFetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        addMessage(chatData.caller, chatData.text);

        // TTS (plays on server) - only if we have text
        if (chatData.text && chatData.text.trim()) {
            showStatus(`${currentCaller.name} is speaking...`);

            await safeFetch('/api/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: chatData.text,
                    voice_id: chatData.voice_id,
                    phone_filter: phoneFilter
                })
            });
        }

    } catch (err) {
        log('Error: ' + err.message);
    }

    isProcessing = false;
    hideStatus();
}


// --- Music (Server-Side) ---
async function loadMusic() {
    try {
        const res = await fetch('/api/music');
        const data = await res.json();
        tracks = data.tracks || [];

        const select = document.getElementById('track-select');
        if (!select) return;

        const previousValue = select.value;
        select.innerHTML = '';

        tracks.forEach((track, i) => {
            const option = document.createElement('option');
            option.value = track.file;
            option.textContent = track.name;
            select.appendChild(option);
        });

        // Restore previous selection if it still exists
        if (previousValue && [...select.options].some(o => o.value === previousValue)) {
            select.value = previousValue;
        }

        console.log('Loaded', tracks.length, 'tracks');
    } catch (err) {
        console.error('loadMusic error:', err);
    }
}


async function playMusic() {
    await loadMusic();
    const select = document.getElementById('track-select');
    const track = select?.value;
    if (!track) return;

    await fetch('/api/music/play', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ track, action: 'play' })
    });
}


async function stopMusic() {
    await fetch('/api/music/stop', { method: 'POST' });
}


async function setMusicVolume(e) {
    const volume = e.target.value / 100;
    await fetch('/api/music/volume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ track: '', action: 'volume', volume })
    });
}


async function loadAds() {
    try {
        const res = await fetch('/api/ads');
        const data = await res.json();
        const ads = data.ads || [];

        const select = document.getElementById('ad-select');
        if (!select) return;

        const previousValue = select.value;
        select.innerHTML = '';

        ads.forEach(ad => {
            const option = document.createElement('option');
            option.value = ad.file;
            option.textContent = ad.name;
            select.appendChild(option);
        });

        if (previousValue && [...select.options].some(o => o.value === previousValue)) {
            select.value = previousValue;
        }

        console.log('Loaded', ads.length, 'ads');
    } catch (err) {
        console.error('loadAds error:', err);
    }
}


async function playAd() {
    await loadAds();
    const select = document.getElementById('ad-select');
    const track = select?.value;
    if (!track) return;

    await fetch('/api/ads/play', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ track, action: 'play' })
    });
}

async function stopAd() {
    await fetch('/api/ads/stop', { method: 'POST' });
}


// --- Sound Effects (Server-Side) ---
async function loadSounds() {
    try {
        const res = await fetch('/api/sounds');
        const data = await res.json();
        sounds = data.sounds || [];

        const board = document.getElementById('soundboard');
        if (!board) return;
        board.innerHTML = '';

        sounds.forEach(sound => {
            const btn = document.createElement('button');
            btn.className = 'sound-btn';
            btn.textContent = sound.name;
            btn.addEventListener('click', () => playSFX(sound.file));
            board.appendChild(btn);
        });
        console.log('Loaded', sounds.length, 'sounds');
    } catch (err) {
        console.error('loadSounds error:', err);
    }
}


async function playSFX(soundFile) {
    await fetch('/api/sfx/play', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sound: soundFile })
    });
}


// --- Settings ---
async function loadSettings() {
    try {
        const res = await fetch('/api/settings');
        const data = await res.json();

        const providerEl = document.getElementById('provider');
        if (providerEl) providerEl.value = data.provider || 'openrouter';

        const modelSelect = document.getElementById('openrouter-model');
        if (modelSelect) {
            modelSelect.innerHTML = '';
            (data.available_openrouter_models || []).forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                if (model === data.openrouter_model) option.selected = true;
                modelSelect.appendChild(option);
            });
        }

        const ollamaModel = document.getElementById('ollama-model');
        const ollamaHost = document.getElementById('ollama-host');
        if (ollamaHost) ollamaHost.value = data.ollama_host || 'http://localhost:11434';

        // Populate Ollama models dropdown
        if (ollamaModel) {
            ollamaModel.innerHTML = '';
            const ollamaModels = data.available_ollama_models || [];
            console.log('Ollama models from API:', ollamaModels.length, ollamaModels);
            if (ollamaModels.length === 0) {
                const option = document.createElement('option');
                option.value = data.ollama_model || 'llama3.2';
                option.textContent = data.ollama_model || 'llama3.2';
                ollamaModel.appendChild(option);
            } else {
                ollamaModels.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    if (model === data.ollama_model) option.selected = true;
                    ollamaModel.appendChild(option);
                });
            }
            console.log('Ollama dropdown options:', ollamaModel.options.length);
        } else {
            console.log('Ollama model element not found!');
        }

        // TTS provider
        const ttsProvider = document.getElementById('tts-provider');
        if (ttsProvider) ttsProvider.value = data.tts_provider || 'elevenlabs';

        updateProviderUI();
        console.log('Settings loaded:', data.provider, 'TTS:', data.tts_provider);
    } catch (err) {
        console.error('loadSettings error:', err);
    }
}


function updateProviderUI() {
    const isOpenRouter = document.getElementById('provider')?.value === 'openrouter';
    document.getElementById('openrouter-settings')?.classList.toggle('hidden', !isOpenRouter);
    document.getElementById('ollama-settings')?.classList.toggle('hidden', isOpenRouter);
}


async function saveSettings() {
    // Save audio devices
    await saveAudioDevices();

    // Save LLM and TTS settings
    await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            provider: document.getElementById('provider')?.value,
            openrouter_model: document.getElementById('openrouter-model')?.value,
            ollama_model: document.getElementById('ollama-model')?.value,
            ollama_host: document.getElementById('ollama-host')?.value,
            tts_provider: document.getElementById('tts-provider')?.value
        })
    });

    document.getElementById('settings-modal')?.classList.add('hidden');
    log('Settings saved');
}


// --- UI Helpers ---
function addMessage(sender, text) {
    const chat = document.getElementById('chat');
    if (!chat) {
        console.log(`[${sender}]: ${text}`);
        return;
    }
    const div = document.createElement('div');

    let className = 'message';
    if (sender === 'You') {
        className += ' host';
    } else if (sender === 'System') {
        className += ' system';
    } else if (sender.includes('(caller)') || sender.includes('Caller #')) {
        className += ' real-caller';
    } else {
        className += ' ai-caller';
    }

    div.className = className;
    div.innerHTML = `<strong>${sender}:</strong> ${text}`;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}


function clearChat() {
    const chat = document.getElementById('chat');
    if (chat) chat.innerHTML = '';
}


function log(text) {
    addMessage('System', text);
}

function updateOnAirBtn(btn, isOn) {
    btn.classList.toggle('on', isOn);
    btn.classList.toggle('off', !isOn);
    btn.textContent = isOn ? 'ON AIR' : 'OFF AIR';
}


function showStatus(text) {
    const status = document.getElementById('status');
    if (status) {
        status.textContent = text;
        status.classList.remove('hidden');
    }
}


function hideStatus() {
    const status = document.getElementById('status');
    if (status) status.classList.add('hidden');
}


// --- Server Control & Logging ---

function startLogPolling() {
    // Poll for logs every second
    logPollInterval = setInterval(fetchLogs, 1000);
    // Initial fetch
    fetchLogs();
}


async function fetchLogs() {
    try {
        const res = await fetch('/api/logs?lines=200');
        const data = await res.json();

        const logEl = document.getElementById('server-log');
        if (!logEl) return;

        // Only update if we have new logs
        if (data.logs.length !== lastLogCount) {
            lastLogCount = data.logs.length;

            logEl.innerHTML = data.logs.map(line => {
                let className = 'log-line';
                if (line.includes('Error') || line.includes('error') || line.includes('ERROR')) {
                    className += ' error';
                } else if (line.includes('Warning') || line.includes('WARNING')) {
                    className += ' warning';
                } else if (line.includes('[TTS]')) {
                    className += ' tts';
                } else if (line.includes('[Chat]')) {
                    className += ' chat';
                }
                return `<div class="${className}">${escapeHtml(line)}</div>`;
            }).join('');

            if (autoScroll) {
                logEl.scrollTop = logEl.scrollHeight;
            }
        }
    } catch (err) {
        // Server might be down, that's ok
        console.log('Log fetch failed (server may be restarting)');
    }
}


function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


async function restartServer() {
    if (!confirm('Restart the server? This will briefly disconnect you.')) return;

    try {
        await fetch('/api/server/restart', { method: 'POST' });
        log('Server restart requested...');

        // Clear the log and wait for server to come back
        document.getElementById('server-log').innerHTML = '<div class="log-line">Restarting server...</div>';

        // Poll until server is back
        let attempts = 0;
        const checkServer = setInterval(async () => {
            attempts++;
            try {
                const res = await fetch('/api/server/status');
                if (res.ok) {
                    clearInterval(checkServer);
                    log('Server restarted successfully');
                    await loadSettings();
                }
            } catch (e) {
                if (attempts > 30) {
                    clearInterval(checkServer);
                    log('Server did not restart - check terminal');
                }
            }
        }, 1000);

    } catch (err) {
        log('Failed to restart server: ' + err.message);
    }
}


// --- Call Queue ---
let queuePollInterval = null;

function startQueuePolling() {
    queuePollInterval = setInterval(fetchQueue, 3000);
    fetchQueue();
}

async function fetchQueue() {
    try {
        const res = await fetch('/api/queue');
        const data = await res.json();
        renderQueue(data.queue);
    } catch (err) {}
}

function renderQueue(queue) {
    const el = document.getElementById('call-queue');
    if (!el) return;

    if (queue.length === 0) {
        el.innerHTML = '<div class="queue-empty">No callers waiting</div>';
        return;
    }

    el.innerHTML = queue.map(caller => {
        const mins = Math.floor(caller.wait_time / 60);
        const secs = caller.wait_time % 60;
        const waitStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
        const displayName = caller.caller_name || caller.phone;
        const screenBadge = caller.screening_status === 'complete'
            ? '<span class="screening-badge screened">Screened</span>'
            : caller.screening_status === 'screening'
            ? '<span class="screening-badge screening">Screening...</span>'
            : '';
        const summary = caller.screening_summary
            ? `<div class="screening-summary">${caller.screening_summary}</div>`
            : '';
        return `
            <div class="queue-item">
                <span class="queue-name">${displayName}</span>
                ${screenBadge}
                <span class="queue-wait">waiting ${waitStr}</span>
                ${summary}
                <button class="queue-take-btn" onclick="takeCall('${caller.caller_id}')">Take Call</button>
                <button class="queue-drop-btn" onclick="dropCall('${caller.caller_id}')">Drop</button>
            </div>
        `;
    }).join('');
}

async function takeCall(callerId) {
    try {
        const res = await fetch(`/api/queue/take/${callerId}`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'on_air') {
            showRealCaller(data.caller);
            log(`${data.caller.phone} is on air — Channel ${data.caller.channel}`);

            // Auto-select an AI caller if none is active
            if (!currentCaller) {
                const callerBtns = document.querySelectorAll('.caller-btn');
                if (callerBtns.length > 0) {
                    const randomIdx = Math.floor(Math.random() * callerBtns.length);
                    const btn = callerBtns[randomIdx];
                    const key = btn.dataset.key;
                    const name = btn.textContent;
                    log(`Auto-selecting ${name} as AI caller`);
                    await startCall(key, name);
                }
            }
        }
    } catch (err) {
        log('Failed to take call: ' + err.message);
    }
}

async function dropCall(callerId) {
    try {
        await fetch(`/api/queue/drop/${callerId}`, { method: 'POST' });
        fetchQueue();
    } catch (err) {
        log('Failed to drop call: ' + err.message);
    }
}


// --- Active Call Indicator ---
let realCallerTimer = null;
let realCallerStartTime = null;

function updateActiveCallIndicator() {
    const container = document.getElementById('active-call');
    const realInfo = document.getElementById('real-caller-info');
    const aiInfo = document.getElementById('ai-caller-info');
    const statusEl = document.getElementById('call-status');

    const hasReal = realInfo && !realInfo.classList.contains('hidden');
    const hasAi = aiInfo && !aiInfo.classList.contains('hidden');

    if (hasReal || hasAi) {
        container?.classList.remove('hidden');
        statusEl?.classList.add('hidden');
    } else {
        container?.classList.add('hidden');
        statusEl?.classList.remove('hidden');
        if (statusEl) statusEl.textContent = 'No active call';
    }
}

function showRealCaller(callerInfo) {
    const nameEl = document.getElementById('real-caller-name');
    const chEl = document.getElementById('real-caller-channel');
    if (nameEl) nameEl.textContent = callerInfo.phone;
    if (chEl) chEl.textContent = `Ch ${callerInfo.channel}`;

    document.getElementById('real-caller-info')?.classList.remove('hidden');
    realCallerStartTime = Date.now();

    if (realCallerTimer) clearInterval(realCallerTimer);
    realCallerTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - realCallerStartTime) / 1000);
        const mins = Math.floor(elapsed / 60);
        const secs = elapsed % 60;
        const durEl = document.getElementById('real-caller-duration');
        if (durEl) durEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
    }, 1000);

    updateActiveCallIndicator();
}

function hideRealCaller() {
    document.getElementById('real-caller-info')?.classList.add('hidden');
    if (realCallerTimer) clearInterval(realCallerTimer);
    realCallerTimer = null;
    updateActiveCallIndicator();
}


// --- AI Respond (manual trigger) ---
async function triggerAiRespond() {
    if (!currentCaller) {
        log('No AI caller active — click a caller first');
        return;
    }

    const btn = document.getElementById('ai-respond-btn');
    if (btn) { btn.disabled = true; btn.textContent = 'Thinking...'; }
    showStatus(`${currentCaller.name} is thinking...`);

    try {
        const data = await safeFetch('/api/ai-respond', { method: 'POST' });
        if (data.text) {
            addMessage(data.caller, data.text);
            showStatus(`${data.caller} is speaking...`);
            const duration = data.text.length * 60;
            setTimeout(hideStatus, Math.min(duration, 15000));
        }
    } catch (err) {
        log('AI respond error: ' + err.message);
    }

    if (btn) { btn.disabled = false; btn.textContent = 'Let them respond'; }
}


// --- Conversation Update Polling ---
let conversationSince = 0;

function startConversationPolling() {
    setInterval(fetchConversationUpdates, 1000);
}

async function fetchConversationUpdates() {
    try {
        const res = await fetch(`/api/conversation/updates?since=${conversationSince}`);
        const data = await res.json();
        if (data.messages && data.messages.length > 0) {
            for (const msg of data.messages) {
                conversationSince = msg.id + 1;
                if (msg.type === 'caller_disconnected') {
                    hideRealCaller();
                    log(`${msg.phone} disconnected (${msg.reason})`);
                } else if (msg.type === 'chat') {
                    addMessage(msg.sender, msg.text);
                } else if (msg.type === 'ai_status') {
                    showStatus(msg.text);
                } else if (msg.type === 'ai_done') {
                    hideStatus();
                } else if (msg.type === 'caller_queued') {
                    // Queue poll will pick this up, just ensure it refreshes
                    fetchQueue();
                }
            }
        }
    } catch (err) {}
}


async function exportSession() {
    try {
        const res = await safeFetch('/api/session/export');
        const blob = new Blob([JSON.stringify(res, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `session-${res.session_id}.json`;
        a.click();
        URL.revokeObjectURL(url);
        log(`Exported session: ${res.call_count} calls`);
    } catch (err) {
        log('Export error: ' + err.message);
    }
}


async function stopServer() {
    if (!confirm('Stop the server? You will need to restart it manually.')) return;

    try {
        await fetch('/api/server/stop', { method: 'POST' });
        log('Server stop requested...');
        document.getElementById('server-log').innerHTML = '<div class="log-line">Server stopped. Run ./run.sh to restart.</div>';
    } catch (err) {
        log('Failed to stop server: ' + err.message);
    }
}
