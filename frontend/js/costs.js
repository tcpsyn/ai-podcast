let currentPeriod = 'all';
let charts = {};

const COLORS = {
    accent: '#e8791d',
    devon: '#c4944a',
    green: '#5a8a3c',
    red: '#cc2222',
    blue: '#4a8ac4',
    purple: '#8a5ac4',
    tan: '#c4845a',
    teal: '#3c8a7a',
    pink: '#c45a8a',
    slate: '#6a7a8a',
};
const COLOR_LIST = Object.values(COLORS);

Chart.defaults.color = '#9a8b78';
Chart.defaults.borderColor = 'rgba(245, 240, 229, 0.08)';

// --- Utilities ---

function formatCost(n) {
    if (n == null) return '--';
    return n < 0.01 ? `$${n.toFixed(4)}` : `$${n.toFixed(2)}`;
}

function formatDate(iso) {
    if (!iso) return '--';
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) +
        ', ' + d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}

function shortenModel(name) {
    if (!name) return '--';
    const i = name.indexOf('/');
    return i >= 0 ? name.slice(i + 1) : name;
}

function destroyChart(key) {
    if (charts[key]) {
        charts[key].destroy();
        charts[key] = null;
    }
}

// --- Data Loading ---

async function loadDashboard() {
    const p = currentPeriod;
    try {
        const [summary, timeline, models, categories, sessions, expensive] = await Promise.all([
            fetch(`/api/costs/summary?period=${p}`).then(r => r.json()),
            fetch(`/api/costs/timeline?period=${p}&group_by=session`).then(r => r.json()),
            fetch(`/api/costs/models?period=${p}`).then(r => r.json()),
            fetch(`/api/costs/categories?period=${p}`).then(r => r.json()),
            fetch(`/api/costs/sessions?period=${p}`).then(r => r.json()),
            fetch(`/api/costs/expensive?period=${p}&limit=10`).then(r => r.json()),
        ]);
        renderSummary(summary);
        renderTimeline(timeline);
        renderModels(models);
        renderCategories(categories);
        renderSessionBars(timeline);
        renderExpensiveTable(expensive);
        renderSessionsTable(sessions);
    } catch (e) {
        console.error('Dashboard load error:', e);
    }
}

// --- Render Functions ---

function renderSummary(data) {
    document.getElementById('total-spend').textContent = formatCost(data.total_cost);
    document.getElementById('llm-tts-split').textContent =
        `${formatCost(data.llm_cost)} / ${formatCost(data.tts_cost)}`;
    document.getElementById('session-count').textContent = data.sessions;
    document.getElementById('avg-cost').textContent = formatCost(data.avg_cost_per_session);

    const changeEl = document.getElementById('total-change');
    if (data.cost_change_pct != null) {
        const up = data.cost_change_pct > 0;
        changeEl.textContent = `${up ? '\u2191' : '\u2193'} ${Math.abs(data.cost_change_pct)}% vs prev period`;
        changeEl.className = 'card-change ' + (up ? 'positive' : 'negative');
    } else {
        changeEl.textContent = '';
        changeEl.className = 'card-change';
    }
}

function renderTimeline(data) {
    destroyChart('timeline');
    const ctx = document.getElementById('timeline-chart');
    const labels = data.map(d => d.date || formatDate(d.started_at));
    charts.timeline = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'LLM',
                    data: data.map(d => d.llm_cost),
                    borderColor: COLORS.accent,
                    backgroundColor: COLORS.accent + '22',
                    fill: true,
                    tension: 0.3,
                },
                {
                    label: 'TTS',
                    data: data.map(d => d.tts_cost),
                    borderColor: COLORS.devon,
                    backgroundColor: COLORS.devon + '22',
                    fill: true,
                    tension: 0.3,
                },
            ],
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
            },
            scales: {
                y: { beginAtZero: true, ticks: { callback: v => '$' + v.toFixed(2) } },
            },
        },
    });
}

function renderModels(data) {
    destroyChart('models');
    const ctx = document.getElementById('model-chart');
    charts.models = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: data.map(d => shortenModel(d.model)),
            datasets: [{
                data: data.map(d => d.cost),
                backgroundColor: data.map((_, i) => COLOR_LIST[i % COLOR_LIST.length]),
                borderWidth: 0,
            }],
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'right', labels: { boxWidth: 12, padding: 8, font: { size: 11 } } },
                tooltip: { callbacks: { label: ctx => `${ctx.label}: ${formatCost(ctx.parsed)}` } },
            },
        },
    });
}

function renderCategories(data) {
    destroyChart('categories');
    const ctx = document.getElementById('category-chart');
    charts.categories = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.category),
            datasets: [{
                data: data.map(d => d.cost),
                backgroundColor: COLORS.accent + 'cc',
                borderRadius: 4,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { beginAtZero: true, ticks: { callback: v => '$' + v.toFixed(2) } },
            },
        },
    });
}

function renderSessionBars(data) {
    destroyChart('sessionBars');
    const ctx = document.getElementById('session-chart');
    const costs = data.map(d => d.total_cost);
    const avg = costs.length ? costs.reduce((a, b) => a + b, 0) / costs.length : 0;
    charts.sessionBars = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.date || formatDate(d.started_at)),
            datasets: [{
                data: costs,
                backgroundColor: costs.map(c => c > avg ? COLORS.red + 'cc' : COLORS.green + 'cc'),
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, ticks: { callback: v => '$' + v.toFixed(2) } },
            },
        },
    });
}

function renderExpensiveTable(data) {
    const tbody = document.querySelector('#expensive-table tbody');
    tbody.innerHTML = data.map(d => `
        <tr>
            <td>${shortenModel(d.model)}</td>
            <td>${d.category}</td>
            <td>${d.caller_name || '\u2014'}</td>
            <td>${(d.prompt_tokens + d.completion_tokens).toLocaleString()}</td>
            <td>${formatCost(d.cost)}</td>
            <td>${d.latency_ms.toFixed(0)}ms</td>
        </tr>
    `).join('');
}

function renderSessionsTable(data) {
    const tbody = document.querySelector('#sessions-table tbody');
    tbody.innerHTML = data.map(d => `
        <tr class="clickable" data-session="${d.session_id}">
            <td>${formatDate(d.started_at)}</td>
            <td>${formatCost(d.llm_cost)}</td>
            <td>${formatCost(d.tts_cost)}</td>
            <td>${formatCost(d.total_cost)}</td>
            <td>${d.total_llm_calls}</td>
            <td><button class="view-btn" data-session="${d.session_id}">View</button></td>
        </tr>
    `).join('');

    tbody.querySelectorAll('.view-btn').forEach(btn => {
        btn.addEventListener('click', e => {
            e.stopPropagation();
            showSessionDetail(btn.dataset.session);
        });
    });
    tbody.querySelectorAll('.clickable').forEach(row => {
        row.addEventListener('click', () => showSessionDetail(row.dataset.session));
    });
}

// --- Session Detail ---

async function showSessionDetail(sessionId) {
    try {
        const detail = await fetch(`/api/costs/session/${sessionId}`).then(r => r.json());

        document.getElementById('detail-session-id').textContent = sessionId;

        populateDetailTable('#detail-caller-table tbody', detail.by_caller,
            d => `<td>${d.caller_name}</td><td>${d.calls}</td><td>${formatCost(d.cost)}</td>`);

        populateDetailTable('#detail-category-table tbody', detail.by_category,
            d => `<td>${d.category}</td><td>${d.calls}</td><td>${formatCost(d.cost)}</td>`);

        populateDetailTable('#detail-model-table tbody', detail.by_model,
            d => `<td>${shortenModel(d.model)}</td><td>${d.calls}</td><td>${formatCost(d.cost)}</td>`);

        populateDetailTable('#detail-expensive-table tbody', detail.expensive_calls,
            d => `<td>${d.category}</td><td>${shortenModel(d.model)}</td><td>${d.caller_name || '\u2014'}</td>` +
                 `<td>${formatCost(d.cost)}</td><td>${(d.prompt_tokens + d.completion_tokens).toLocaleString()}</td>` +
                 `<td>${d.latency_ms.toFixed(0)}ms</td>`);

        document.querySelectorAll('.costs-main > section:not(.session-detail)').forEach(
            s => s.style.display = 'none');
        document.getElementById('session-detail').classList.remove('hidden');
    } catch (e) {
        console.error('Detail load error:', e);
    }
}

function populateDetailTable(selector, data, rowFn) {
    const tbody = document.querySelector(selector);
    tbody.innerHTML = data.map(d => `<tr>${rowFn(d)}</tr>`).join('');
}

function closeDetail() {
    document.getElementById('session-detail').classList.add('hidden');
    document.querySelectorAll('.costs-main > section:not(.session-detail)').forEach(
        s => s.style.display = '');
}

// --- Init ---

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.period-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelector('.period-tab.active').classList.remove('active');
            tab.classList.add('active');
            currentPeriod = tab.dataset.period;
            loadDashboard();
        });
    });
    document.getElementById('close-detail').addEventListener('click', closeDetail);
    loadDashboard();
});
