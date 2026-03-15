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
let isMusicPlaying = false;
let soundboardExpanded = false;

// --- Show Clock ---
let showStartTime = null;      // when ON AIR was pressed
let showContentTime = 0;       // seconds of "active" content (calls, music, etc.)
let showContentTracking = false; // whether we're in active content right now
let showClockInterval = null;

function initClock() {
    // Always show current time
    if (!showClockInterval) {
        showClockInterval = setInterval(updateShowClock, 1000);
        updateShowClock();
    }
}

function startShowClock() {
    showStartTime = Date.now();
    showContentTime = 0;
    showContentTracking = false;
    document.getElementById('show-timers')?.classList.remove('hidden');
}

function stopShowClock() {
    document.getElementById('show-timers')?.classList.add('hidden');
    showStartTime = null;
}

function updateShowClock() {
    // Current time
    const now = new Date();
    const timeEl = document.getElementById('clock-time');
    if (timeEl) timeEl.textContent = now.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', second: '2-digit', hour12: true });

    if (!showStartTime) return;

    // Track content time — count seconds when a call is active or music is playing
    const isContent = !!(currentCaller || isMusicPlaying);
    if (isContent && !showContentTracking) {
        showContentTracking = true;
    } else if (!isContent && showContentTracking) {
        showContentTracking = false;
    }
    if (isContent) showContentTime++;

    // Show runtime (wall clock since ON AIR)
    const runtimeSec = Math.floor((Date.now() - showStartTime) / 1000);
    const runtimeEl = document.getElementById('clock-runtime');
    if (runtimeEl) runtimeEl.textContent = formatDuration(runtimeSec);

    // Estimated final length after post-prod
    // Post-prod removes 2-8 second gaps (TTS latency). Estimate:
    // - Content time stays ~100% (it's all talking/music)
    // - Dead air (runtime - content) gets ~70% removed (not all silence is cut)
    const deadAir = Math.max(0, runtimeSec - showContentTime);
    const estimatedFinal = showContentTime + (deadAir * 0.3);
    const estEl = document.getElementById('clock-estimate');
    if (estEl) estEl.textContent = formatDuration(Math.round(estimatedFinal));
}

function formatDuration(totalSec) {
    const h = Math.floor(totalSec / 3600);
    const m = Math.floor((totalSec % 3600) / 60);
    const s = totalSec % 60;
    if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    return `${m}:${String(s).padStart(2, '0')}`;
}


// --- Helpers ---
function _isTyping() {
    const el = document.activeElement;
    if (!el) return false;
    const tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}


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
        await loadIdents();
        await loadSounds();
        await loadSettings();
        initEventListeners();
        initClock();
        loadVoicemails();
        setInterval(loadVoicemails, 30000);
        loadEmails();
        setInterval(loadEmails, 30000);
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

    // Log toggle (collapsed by default)
    const logToggleBtn = document.getElementById('log-toggle-btn');
    if (logToggleBtn) {
        logToggleBtn.addEventListener('click', () => {
            const logBody = document.querySelector('.log-body');
            if (!logBody) return;
            const collapsed = logBody.classList.toggle('collapsed');
            logToggleBtn.textContent = collapsed ? 'Show \u25BC' : 'Hide \u25B2';
        });
    }

    // Start log polling
    startLogPolling();

    // Start queue polling
    startQueuePolling();

    // Start cost polling
    startCostPolling();

    // Talk button - now triggers server-side recording
    const talkBtn = document.getElementById('talk-btn');
    if (talkBtn) {
        talkBtn.addEventListener('mousedown', startRecording);
        // Listen on document for mouseup so layout shifts don't orphan the release
        document.addEventListener('mouseup', () => { if (isRecording) stopRecording(); });
        talkBtn.addEventListener('touchstart', e => { e.preventDefault(); startRecording(); });
        talkBtn.addEventListener('touchend', e => { e.preventDefault(); stopRecording(); });
    }

    // Spacebar push-to-talk — blur buttons so Space doesn't also trigger button click
    document.addEventListener('keydown', e => {
        if (e.code !== 'Space' || e.repeat || _isTyping()) return;
        e.preventDefault();
        // Blur any focused button so browser doesn't fire its click
        if (document.activeElement?.tagName === 'BUTTON') document.activeElement.blur();
        startRecording();
    });
    document.addEventListener('keyup', e => {
        if (e.code !== 'Space' || _isTyping()) return;
        e.preventDefault();
        stopRecording();
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', e => {
        if (_isTyping()) return;
        // Don't fire shortcuts when a modal is open (except Escape to close it)
        const modalOpen = document.querySelector('.modal:not(.hidden)');
        if (e.key === 'Escape') {
            if (modalOpen) {
                modalOpen.classList.add('hidden');
                e.preventDefault();
            }
            return;
        }
        if (modalOpen) return;

        const key = e.key.toLowerCase();
        // 1-9, 0: Start call with caller in that slot
        if (/^[0-9]$/.test(key)) {
            e.preventDefault();
            const idx = key === '0' ? 9 : parseInt(key) - 1;
            const btns = document.querySelectorAll('.caller-btn');
            if (btns[idx]) btns[idx].click();
            return;
        }
        switch (key) {
            case 'h':
                e.preventDefault();
                hangup();
                break;
            case 'w':
                e.preventDefault();
                wrapUp();
                break;
            case 'm':
                e.preventDefault();
                toggleMusic();
                break;
            case 'd':
                e.preventDefault();
                document.getElementById('devon-input')?.focus();
                break;
        }
    });

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

    // Idents
    document.getElementById('ident-play-btn')?.addEventListener('click', playIdent);
    document.getElementById('ident-stop-btn')?.addEventListener('click', stopIdent);

    // Devon (Intern)
    document.getElementById('devon-ask-btn')?.addEventListener('click', () => {
        const input = document.getElementById('devon-input');
        if (input?.value.trim()) {
            askDevon(input.value.trim());
            input.value = '';
        }
    });
    document.getElementById('devon-input')?.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
            e.preventDefault();
            const input = e.target;
            if (input.value.trim()) {
                askDevon(input.value.trim());
                input.value = '';
            }
        }
    });
    document.getElementById('devon-interject-btn')?.addEventListener('click', interjectDevon);
    document.getElementById('devon-monitor')?.addEventListener('change', e => {
        toggleInternMonitor(e.target.checked);
    });
    document.getElementById('devon-play-btn')?.addEventListener('click', playDevonSuggestion);
    document.getElementById('devon-dismiss-btn')?.addEventListener('click', dismissDevonSuggestion);

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

    // Wrap-up button
    document.getElementById('wrapup-btn')?.addEventListener('click', wrapUp);

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
        const identCh = document.getElementById('ident-channel');

        if (inputCh) inputCh.value = settings.input_channel || 1;
        if (callerCh) callerCh.value = settings.caller_channel || 1;
        if (liveCallerCh) liveCallerCh.value = settings.live_caller_channel || 9;
        if (musicCh) musicCh.value = settings.music_channel || 2;
        if (sfxCh) sfxCh.value = settings.sfx_channel || 3;
        if (adCh) adCh.value = settings.ad_channel || 11;
        if (identCh) identCh.value = settings.ident_channel || 15;

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
    const identChannel = document.getElementById('ident-channel')?.value;
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
            ident_channel: identChannel ? parseInt(identChannel) : 15,
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

        data.callers.forEach((caller, idx) => {
            const btn = document.createElement('button');
            btn.className = 'caller-btn';
            if (caller.returning) btn.classList.add('returning');
            btn.dataset.key = caller.key;

            let html = '';
            if (caller.energy_level) {
                const energyColors = { low: '#4a7ab5', medium: '#5a8a3c', high: '#e8791d', very_high: '#cc2222' };
                const color = energyColors[caller.energy_level] || '#9a8b78';
                html += `<span class="energy-dot" style="background:${color}" title="${caller.energy_level} energy"></span>`;
            }
            html += caller.returning ? `<span class="caller-name">\u2605 ${caller.name}</span>` : `<span class="caller-name">${caller.name}</span>`;
            if (caller.call_shape && caller.call_shape !== 'standard') {
                const shapeLabels = {
                    escalating_reveal: 'ER', am_i_the_asshole: 'AITA', confrontation: 'VS',
                    celebration: '\u{1F389}', quick_hit: 'QH', bait_and_switch: 'B&S',
                    the_hangup: 'HU', reactive: 'RE'
                };
                const label = shapeLabels[caller.call_shape] || caller.call_shape.substring(0, 2).toUpperCase();
                html += `<span class="shape-badge" title="${caller.call_shape.replace(/_/g, ' ')}">${label}</span>`;
            }
            // Shortcut label: 1-9 for first 9, 0 for 10th
            if (idx < 10) {
                const shortcutKey = idx === 9 ? '0' : String(idx + 1);
                html += `<span class="shortcut-label">${shortcutKey}</span>`;
            }
            btn.innerHTML = html;
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
    document.querySelector('.callers-section')?.classList.add('call-active');
    document.querySelector('.chat-section')?.classList.add('call-active');

    // Check if real caller is active (three-way scenario)
    const realCallerActive = document.getElementById('real-caller-info') &&
        !document.getElementById('real-caller-info').classList.contains('hidden');

    if (realCallerActive) {
        document.getElementById('call-status').textContent = `Three-way: ${name} (AI) + Real Caller`;
    } else {
        document.getElementById('call-status').textContent = `On call: ${name}`;
    }

    document.getElementById('hangup-btn').disabled = false;
    const wrapupBtn = document.getElementById('wrapup-btn');
    if (wrapupBtn) { wrapupBtn.disabled = false; wrapupBtn.classList.remove('active'); }

    // Show AI caller in active call indicator
    const aiInfo = document.getElementById('ai-caller-info');
    const aiName = document.getElementById('ai-caller-name');
    if (aiInfo) aiInfo.classList.remove('hidden');
    if (aiName) aiName.textContent = name;

    // Show caller info panel with structured data
    const infoPanel = document.getElementById('caller-info-panel');
    if (infoPanel && data.caller_info) {
        const ci = data.caller_info;
        const energyColors = { low: '#4a7ab5', medium: '#5a8a3c', high: '#e8791d', very_high: '#cc2222' };
        const shapeBadge = document.getElementById('caller-shape-badge');
        const energyBadge = document.getElementById('caller-energy-badge');
        const emotionBadge = document.getElementById('caller-emotion');
        const signature = document.getElementById('caller-signature');
        const situation = document.getElementById('caller-situation');
        if (shapeBadge) shapeBadge.textContent = (ci.call_shape || 'standard').replace(/_/g, ' ');
        if (energyBadge) { energyBadge.textContent = (ci.energy_level || '').replace('_', ' '); energyBadge.style.background = energyColors[ci.energy_level] || '#9a8b78'; }
        if (emotionBadge) emotionBadge.textContent = ci.emotional_state || '';
        if (signature) signature.textContent = ci.signature_detail ? `"${ci.signature_detail}"` : '';
        if (situation) situation.textContent = ci.situation_summary || '';
        infoPanel.classList.remove('hidden');
    }
    const bgEl = document.getElementById('caller-background');
    if (bgEl && data.background) bgEl.textContent = data.background;

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
    document.querySelector('.callers-section')?.classList.remove('call-active');
    document.querySelector('.chat-section')?.classList.remove('call-active');

    document.getElementById('call-status').textContent = 'No active call';
    document.getElementById('hangup-btn').disabled = true;
    const wrapBtn = document.getElementById('wrapup-btn');
    if (wrapBtn) { wrapBtn.disabled = true; wrapBtn.classList.remove('active'); }
    document.querySelectorAll('.caller-btn').forEach(btn => btn.classList.remove('active'));

    // Hide caller info panel and background
    document.getElementById('caller-info-panel')?.classList.add('hidden');
    const bgDetails2 = document.getElementById('caller-background-details');
    if (bgDetails2) bgDetails2.classList.add('hidden');

    // Hide AI caller indicator
    document.getElementById('ai-caller-info')?.classList.add('hidden');
    updateActiveCallIndicator();
}


async function wrapUp() {
    if (!currentCaller) return;
    try {
        const res = await fetch('/api/wrap-up', { method: 'POST' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            log(`Wrap-up failed: ${err.detail || res.status}`);
            return;
        }
        const wrapupBtn = document.getElementById('wrapup-btn');
        if (wrapupBtn) wrapupBtn.classList.add('active');
        log(`Wrapping up ${currentCaller.name}...`);
    } catch (err) {
        log(`Wrap-up error: ${err.message}`);
    }
}

function toggleMusic() {
    if (isMusicPlaying) {
        stopMusic();
    } else {
        playMusic();
    }
}

// --- Server-Side Recording ---
async function startRecording() {
    if (isProcessing) return;

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

        if (!currentCaller) {
            // No active call — route voice to Devon
            showStatus('Devon is thinking...');
            await askDevon(data.text, { skipHostMessage: true });
        } else {
            // Active call — talk to caller
            showStatus(`${currentCaller.name} is thinking...`);

            const chatData = await safeFetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: data.text })
            });

            // If routed to Devon, the SSE broadcast handles the message
            if (chatData.routed_to !== 'devon') {
                addMessage(chatData.caller, chatData.text);
            }

            // TTS (plays on server) - only if we have text and not routed to Devon
            if (chatData.text && chatData.text.trim() && chatData.routed_to !== 'devon') {
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
    if (!text || isProcessing) return;

    input.value = '';
    document.getElementById('type-modal').classList.add('hidden');

    isProcessing = true;
    addMessage('You', text);

    try {
        if (!currentCaller) {
            // No active call — route to Devon
            showStatus('Devon is thinking...');
            await askDevon(text, { skipHostMessage: true });
        } else {
            showStatus(`${currentCaller.name} is thinking...`);

            const chatData = await safeFetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (chatData.routed_to !== 'devon') {
                addMessage(chatData.caller, chatData.text);
            }

            // TTS (plays on server) - only if we have text and not routed to Devon
            if (chatData.text && chatData.text.trim() && chatData.routed_to !== 'devon') {
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

        // Group tracks by genre
        const genres = {};
        tracks.forEach(track => {
            const genre = track.genre || 'Other';
            if (!genres[genre]) genres[genre] = [];
            genres[genre].push(track);
        });

        // Sort genre names, but put "Other" last
        const genreOrder = Object.keys(genres).sort((a, b) => {
            if (a === 'Other') return 1;
            if (b === 'Other') return -1;
            return a.localeCompare(b);
        });

        genreOrder.forEach(genre => {
            const group = document.createElement('optgroup');
            group.label = genre;
            // Shuffle within each genre group
            const genreTracks = genres[genre];
            for (let i = genreTracks.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [genreTracks[i], genreTracks[j]] = [genreTracks[j], genreTracks[i]];
            }
            genreTracks.forEach(track => {
                const option = document.createElement('option');
                option.value = track.file;
                option.textContent = track.name;
                group.appendChild(option);
            });
            select.appendChild(group);
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
    isMusicPlaying = true;
}


async function stopMusic() {
    await fetch('/api/music/stop', { method: 'POST' });
    isMusicPlaying = false;
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


// --- Sound Effects (Server-Side) ---
async function loadSounds() {
    try {
        const res = await fetch('/api/sounds');
        const data = await res.json();
        sounds = data.sounds || [];

        const board = document.getElementById('soundboard');
        if (!board) return;
        board.innerHTML = '';

        const pinnedNames = ['cheer', 'applause', 'boo', 'correct'];
        const pinned = [];
        const rest = [];

        sounds.forEach(sound => {
            const lower = ((sound.name || '') + ' ' + (sound.file || '')).toLowerCase();
            if (pinnedNames.some(p => lower.includes(p))) {
                pinned.push(sound);
            } else {
                rest.push(sound);
            }
        });

        // Pinned buttons — always visible
        const pinnedRow = document.createElement('div');
        pinnedRow.className = 'soundboard-pinned';
        pinned.forEach(sound => {
            const btn = document.createElement('button');
            btn.className = 'sound-btn pinned';
            const lower = (sound.name || sound.file || '').toLowerCase();
            if (lower.includes('cheer')) btn.classList.add('pin-cheer');
            else if (lower.includes('applause')) btn.classList.add('pin-applause');
            else if (lower.includes('boo')) btn.classList.add('pin-boo');
            btn.textContent = sound.name;
            btn.addEventListener('click', () => playSFX(sound.file));
            pinnedRow.appendChild(btn);
        });
        board.appendChild(pinnedRow);

        // Collapsible section for remaining sounds
        if (rest.length > 0) {
            const toggle = document.createElement('button');
            toggle.className = 'soundboard-toggle';
            toggle.innerHTML = 'More Sounds <span class="toggle-arrow">&#9660;</span>';
            toggle.addEventListener('click', () => {
                soundboardExpanded = !soundboardExpanded;
                grid.classList.toggle('hidden', !soundboardExpanded);
                toggle.querySelector('.toggle-arrow').innerHTML = soundboardExpanded ? '&#9650;' : '&#9660;';
            });
            board.appendChild(toggle);

            const grid = document.createElement('div');
            grid.className = 'soundboard-grid hidden';
            rest.forEach(sound => {
                const btn = document.createElement('button');
                btn.className = 'sound-btn';
                btn.textContent = sound.name;
                btn.addEventListener('click', () => playSFX(sound.file));
                grid.appendChild(btn);
            });
            board.appendChild(grid);
        }

        console.log('Loaded', sounds.length, 'sounds', `(${pinned.length} pinned)`);
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
        // System messages are compact — no avatar, small text
        div.className = className;
        div.innerHTML = `<div class="msg-content system-compact">${text}</div>`;
        chat.appendChild(div);
        chat.scrollTop = chat.scrollHeight;
        return;
    } else if (sender === 'DEVON') {
        className += ' devon';
    } else if (sender.includes('(caller)') || sender.includes('Caller #')) {
        className += ' real-caller';
    } else {
        className += ' ai-caller';
    }

    div.className = className;

    // Build avatar — real face images from /api/avatar/{name}
    let avatarHtml = '';
    if (sender === 'You') {
        avatarHtml = '<img class="msg-avatar" src="/images/host-avatar.png" alt="Luke">';
    } else if (sender === 'DEVON') {
        avatarHtml = '<img class="msg-avatar msg-avatar-devon" src="/api/avatar/Devon" alt="Devon">';
    } else if (sender === 'System') {
        avatarHtml = '<span class="msg-avatar msg-avatar-system">!</span>';
    } else {
        const name = sender.replace(/[^a-zA-Z]/g, '') || 'Caller';
        avatarHtml = `<img class="msg-avatar msg-avatar-caller" src="/api/avatar/${encodeURIComponent(name)}" alt="${name}">`;
    }

    div.innerHTML = `${avatarHtml}<div class="msg-content"><strong>${sender}:</strong> ${text}</div>`;
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
    if (isOn && !showStartTime) startShowClock();
    else if (!isOn && showStartTime) stopShowClock();
}


function showStatus(text) {
    const status = document.getElementById('status');
    if (status) {
        status.textContent = text;
        status.classList.remove('hidden');
    }
    document.getElementById('chat')?.classList.add('thinking');
}


function hideStatus() {
    const status = document.getElementById('status');
    if (status) status.classList.add('hidden');
    document.getElementById('chat')?.classList.remove('thinking');
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


// --- Cost Polling ---

function startCostPolling() {
    setInterval(fetchCosts, 5000);
}

async function fetchCosts() {
    try {
        const res = await fetch('/api/costs');
        if (!res.ok) return;
        const data = await res.json();
        const el = document.getElementById('clock-cost');
        if (!el) return;
        el.textContent = '$' + data.total_cost_usd.toFixed(2);
        el.classList.remove('cost-low', 'cost-mid', 'cost-high');
        if (data.total_cost_usd < 0.50) el.classList.add('cost-low');
        else if (data.total_cost_usd < 2.00) el.classList.add('cost-mid');
        else el.classList.add('cost-high');
    } catch (err) {}
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
                } else if (msg.type === 'intern_response') {
                    addMessage('DEVON', msg.text);
                } else if (msg.type === 'intern_suggestion') {
                    showDevonSuggestion(msg.text);
                }
            }
        }
        // Check for intern suggestion in polling response
        if (data.intern_suggestion) {
            showDevonSuggestion(data.intern_suggestion.text);
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


// --- Voicemail ---
let _currentVmAudio = null;

async function loadVoicemails() {
    try {
        const res = await fetch('/api/voicemails');
        const data = await res.json();
        renderVoicemails(data);
    } catch (err) {}
}

function renderVoicemails(voicemails) {
    const list = document.getElementById('voicemail-list');
    const badge = document.getElementById('voicemail-badge');
    if (!list) return;

    const unlistened = voicemails.filter(v => !v.listened).length;
    if (badge) {
        badge.textContent = unlistened;
        badge.classList.toggle('hidden', unlistened === 0);
    }

    if (voicemails.length === 0) {
        list.innerHTML = '<div class="queue-empty">No voicemails</div>';
        return;
    }

    list.innerHTML = voicemails.map(v => {
        const date = new Date(v.timestamp * 1000);
        const timeStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const mins = Math.floor(v.duration / 60);
        const secs = v.duration % 60;
        const durStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
        const unlistenedCls = v.listened ? '' : ' vm-unlistened';
        return `<div class="vm-item${unlistenedCls}" data-id="${v.id}">
            <div class="vm-info">
                <span class="vm-phone">${v.phone}</span>
                <span class="vm-time">${timeStr}</span>
                <span class="vm-dur">${durStr}</span>
            </div>
            <div class="vm-actions">
                <button class="vm-btn listen" onclick="listenVoicemail('${v.id}')">Listen</button>
                <button class="vm-btn on-air" onclick="playVoicemailOnAir('${v.id}')">On Air</button>
                <button class="vm-btn save" onclick="saveVoicemail('${v.id}')">Save</button>
                <button class="vm-btn delete" onclick="deleteVoicemail('${v.id}')">Del</button>
            </div>
        </div>`;
    }).join('');
}

function listenVoicemail(id) {
    if (_currentVmAudio) {
        _currentVmAudio.pause();
        _currentVmAudio = null;
    }
    _currentVmAudio = new Audio(`/api/voicemail/${id}/audio`);
    _currentVmAudio.play();
    fetch(`/api/voicemail/${id}/mark-listened`, { method: 'POST' }).then(() => loadVoicemails());
}

async function playVoicemailOnAir(id) {
    try {
        await safeFetch(`/api/voicemail/${id}/play-on-air`, { method: 'POST' });
        log('Playing voicemail on air');
        loadVoicemails();
    } catch (err) {
        log('Failed to play voicemail: ' + err.message);
    }
}

async function saveVoicemail(id) {
    try {
        await safeFetch(`/api/voicemail/${id}/save`, { method: 'POST' });
        log('Voicemail saved to archive');
    } catch (err) {
        log('Failed to save voicemail: ' + err.message);
    }
}

async function deleteVoicemail(id) {
    if (!confirm('Delete this voicemail?')) return;
    try {
        await safeFetch(`/api/voicemail/${id}`, { method: 'DELETE' });
        loadVoicemails();
    } catch (err) {
        log('Failed to delete voicemail: ' + err.message);
    }
}


// --- Listener Emails ---
async function loadEmails() {
    try {
        const res = await fetch('/api/emails');
        const data = await res.json();
        renderEmails(data);
    } catch (err) {}
}

function renderEmails(emails) {
    const list = document.getElementById('email-list');
    const badge = document.getElementById('email-badge');
    if (!list) return;

    const unread = emails.filter(e => !e.read_on_air).length;
    if (badge) {
        badge.textContent = unread;
        badge.classList.toggle('hidden', unread === 0);
    }

    if (emails.length === 0) {
        list.innerHTML = '<div class="queue-empty">No emails</div>';
        return;
    }

    list.innerHTML = emails.map(e => {
        const date = new Date(e.timestamp * 1000);
        const timeStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const preview = e.body.length > 120 ? e.body.substring(0, 120) + '…' : e.body;
        const unreadCls = e.read_on_air ? '' : ' vm-unlistened';
        const senderName = e.sender.replace(/<.*>/, '').trim() || e.sender;
        return `<div class="email-item${unreadCls}" data-id="${e.id}">
            <div class="email-header">
                <span class="email-sender">${escapeHtml(senderName)}</span>
                <span class="vm-time">${timeStr}</span>
            </div>
            <div class="email-subject">${escapeHtml(e.subject)}</div>
            <div class="email-preview">${escapeHtml(preview)}</div>
            <div class="vm-actions">
                <button class="vm-btn listen" onclick="viewEmail('${e.id}')">View</button>
                <button class="vm-btn on-air" onclick="playEmailOnAir('${e.id}')">On Air (TTS)</button>
                <button class="vm-btn delete" onclick="deleteEmail('${e.id}')">Del</button>
            </div>
        </div>`;
    }).join('');
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function viewEmail(id) {
    const item = document.querySelector(`.email-item[data-id="${id}"]`);
    if (!item) return;
    const existing = item.querySelector('.email-body-expanded');
    if (existing) { existing.remove(); return; }
    fetch('/api/emails').then(r => r.json()).then(emails => {
        const em = emails.find(e => e.id === id);
        if (!em) return;
        const div = document.createElement('div');
        div.className = 'email-body-expanded';
        div.textContent = em.body;
        item.appendChild(div);
    });
}

async function playEmailOnAir(id) {
    try {
        await safeFetch(`/api/email/${id}/play-on-air`, { method: 'POST' });
        log('Reading email on air (TTS)');
        loadEmails();
    } catch (err) {
        log('Failed to play email: ' + err.message);
    }
}

async function deleteEmail(id) {
    if (!confirm('Delete this email?')) return;
    try {
        await safeFetch(`/api/email/${id}`, { method: 'DELETE' });
        loadEmails();
    } catch (err) {
        log('Failed to delete email: ' + err.message);
    }
}


// --- Devon (Intern) ---

async function askDevon(question, { skipHostMessage = false } = {}) {
    if (!skipHostMessage) addMessage('You', `Devon, ${question}`);
    log(`[Devon] Looking up: ${question}`);
    try {
        const res = await safeFetch('/api/intern/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });
        if (res.text) {
            // Don't addMessage here — the SSE broadcast_event("intern_response") handles it
            log(`[Devon] Responded (tools: ${(res.sources || []).join(', ') || 'none'})`);
        } else {
            log('[Devon] No response');
        }
    } catch (err) {
        log('[Devon] Error: ' + err.message);
    }
}

async function interjectDevon() {
    log('[Devon] Checking for interjection...');
    try {
        const res = await safeFetch('/api/intern/interject', { method: 'POST' });
        if (res.text) {
            // Don't addMessage here — SSE broadcast handles it
            log('[Devon] Interjected');
        } else {
            log('[Devon] Nothing to add');
        }
    } catch (err) {
        log('[Devon] Interject error: ' + err.message);
    }
}

async function toggleInternMonitor(enabled) {
    try {
        await safeFetch('/api/intern/monitor', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled }),
        });
        log(`[Devon] Monitor ${enabled ? 'on' : 'off'}`);
    } catch (err) {
        log('[Devon] Monitor toggle error: ' + err.message);
    }
}

function showDevonSuggestion(text) {
    const el = document.getElementById('devon-suggestion');
    const textEl = el?.querySelector('.devon-suggestion-text');
    if (el && textEl) {
        textEl.textContent = text ? `Devon: "${text.substring(0, 60)}${text.length > 60 ? '...' : ''}"` : 'Devon has something';
        el.classList.remove('hidden');
    }
}

async function playDevonSuggestion() {
    try {
        const res = await safeFetch('/api/intern/suggestion/play', { method: 'POST' });
        // Don't addMessage here — SSE broadcast handles it
        document.getElementById('devon-suggestion')?.classList.add('hidden');
        log('[Devon] Played suggestion');
    } catch (err) {
        log('[Devon] Play suggestion error: ' + err.message);
    }
}

async function dismissDevonSuggestion() {
    try {
        await safeFetch('/api/intern/suggestion/dismiss', { method: 'POST' });
        document.getElementById('devon-suggestion')?.classList.add('hidden');
    } catch (err) {}
}
