# Show Theme Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a "show theme" input to the header bar that injects theme context into caller background generation and conversation prompts, nudging callers toward the theme without forcing it.

**Architecture:** Store theme as a string on the Session object. Pass it into `_generate_caller_background_llm()` to bias character creation and into `get_caller_prompt()` so callers are aware of the show's theme during dialog. Frontend adds a text input in the header with a set/clear button, persisted via a new API endpoint.

**Tech Stack:** Python (FastAPI backend), vanilla HTML/CSS/JS frontend

---

### Task 1: Add theme to Session and API endpoint

**Files:**
- Modify: `backend/main.py:6192-6218` (Session class)
- Modify: `backend/main.py:8731-8759` (settings endpoints area)

**Step 1: Add `show_theme` field to Session.__init__**

In `backend/main.py`, inside `Session.__init__` (line ~6217, after `self.intern_monitoring`), add:

```python
self.show_theme: str = ""  # Current show theme (e.g. "St. Patrick's Day")
```

**Step 2: Add GET/POST endpoints for show theme**

Add these endpoints near the existing settings endpoints (~line 8760):

```python
@app.get("/api/show-theme")
async def get_show_theme():
    return {"theme": session.show_theme}


@app.post("/api/show-theme")
async def set_show_theme(data: dict):
    theme = data.get("theme", "").strip()
    old_theme = session.show_theme
    session.show_theme = theme
    if theme:
        print(f"[Theme] Show theme set: {theme}")
    elif old_theme:
        print(f"[Theme] Show theme cleared (was: {old_theme})")
    return {"theme": session.show_theme}
```

**Step 3: Verify the server starts without errors**

Run: `curl -s http://localhost:8000/api/show-theme | python -m json.tool`
Expected: `{"theme": ""}`

**Step 4: Commit**

```bash
git add backend/main.py
git commit -m "Add show theme to Session and API endpoints"
```

---

### Task 2: Inject theme into caller background generation

**Files:**
- Modify: `backend/main.py:5208-5330` (`_generate_caller_background_llm`)
- Modify: `backend/main.py:5381-5396` (`_pregenerate_backgrounds`)

**Step 1: Pass theme into background generation prompt**

In `_generate_caller_background_llm()` (~line 5307), after the line that builds `prompt = f"""Write a brief character description...`, add theme context. Find this section of the prompt string (around line 5316):

```python
{f'CALLER ENERGY: {style_hint}' if style_hint else ''}
```

Immediately after that line (still inside the f-string), add:

```python
{f"SHOW THEME: Tonight's show theme is '{session.show_theme}'. This caller might have a story or angle related to this theme — or they might not. Not every caller has to be about the theme, but if their reason for calling can naturally connect to it, lean into that connection. The theme should feel like a through-line, not a mandate." if session.show_theme else ''}
```

**Step 2: Verify backgrounds generate with theme**

Set a theme via API, then start a new session and check the server logs for background generation.

Run:
```bash
curl -s -X POST http://localhost:8000/api/show-theme -H 'Content-Type: application/json' -d '{"theme": "St. Patricks Day"}'
```

**Step 3: Commit**

```bash
git add backend/main.py
git commit -m "Inject show theme into caller background generation"
```

---

### Task 3: Inject theme into conversation system prompt

**Files:**
- Modify: `backend/main.py:5995-6094` (`get_caller_prompt`)

**Step 1: Add theme block to get_caller_prompt**

In `get_caller_prompt()`, after the `world_context` block is built (around line 6015), add:

```python
theme_context = ""
if session.show_theme:
    theme_context = f"\nSHOW THEME: Tonight's show theme is \"{session.show_theme}\". You're aware of the theme — the host mentioned it at the top of the show. If your story or situation connects to it, you might bring it up naturally. But don't force it. Not every caller has to be about the theme. If the host steers you toward the theme, go with it.\n"
```

Then inject `{theme_context}` into the return f-string. Find this line (~6063):

```python
{relationship_context}{history}{world_context}{emotional_read}
```

Change it to:

```python
{relationship_context}{history}{world_context}{theme_context}{emotional_read}
```

**Step 2: Verify prompt includes theme**

This can be verified by checking server logs during a call (the full prompt is logged at debug level).

**Step 3: Commit**

```bash
git add backend/main.py
git commit -m "Inject show theme into caller conversation prompt"
```

---

### Task 4: Add theme input to frontend header

**Files:**
- Modify: `frontend/index.html` (header section, lines 11-33)
- Modify: `frontend/css/style.css`
- Modify: `frontend/js/app.js`

**Step 1: Add theme input HTML to header**

In `frontend/index.html`, inside the `<header>` section, after the `.header-buttons` div (line ~19) and before the `#show-clock` div (line ~20), add:

```html
<div class="theme-bar">
    <label for="show-theme-input" class="theme-label">Theme:</label>
    <input type="text" id="show-theme-input" class="theme-input" placeholder="e.g. St. Patrick's Day" maxlength="100">
    <button id="set-theme-btn" class="theme-btn set" title="Set show theme">Set</button>
    <button id="clear-theme-btn" class="theme-btn clear hidden" title="Clear theme">✕</button>
</div>
```

**Step 2: Add CSS for theme bar**

In `frontend/css/style.css`, add styles for the theme bar. Place near other header styles:

```css
.theme-bar {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 6px;
}

.theme-label {
    font-size: 0.8rem;
    color: #aaa;
    white-space: nowrap;
}

.theme-input {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 4px;
    color: #fff;
    padding: 4px 8px;
    font-size: 0.85rem;
    width: 200px;
}

.theme-input:focus {
    outline: none;
    border-color: #f5a623;
}

.theme-input.active {
    border-color: #f5a623;
    background: rgba(245, 166, 35, 0.1);
}

.theme-btn {
    padding: 4px 10px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    font-size: 0.8rem;
}

.theme-btn.set {
    background: #f5a623;
    color: #000;
}

.theme-btn.set:hover {
    background: #e6991a;
}

.theme-btn.clear {
    background: rgba(255, 255, 255, 0.1);
    color: #aaa;
    padding: 4px 6px;
}

.theme-btn.clear:hover {
    background: rgba(255, 80, 80, 0.3);
    color: #ff5050;
}
```

**Step 3: Add JS for theme set/clear**

In `frontend/js/app.js`, add theme management functions and wire up event listeners. Add near other initialization code:

```javascript
async function loadShowTheme() {
    try {
        const res = await fetch('/api/show-theme');
        const data = await res.json();
        const input = document.getElementById('show-theme-input');
        const setBtn = document.getElementById('set-theme-btn');
        const clearBtn = document.getElementById('clear-theme-btn');
        if (data.theme) {
            input.value = data.theme;
            input.classList.add('active');
            setBtn.classList.add('hidden');
            clearBtn.classList.remove('hidden');
        }
    } catch (e) {
        console.error('Failed to load show theme:', e);
    }
}

async function setShowTheme() {
    const input = document.getElementById('show-theme-input');
    const theme = input.value.trim();
    if (!theme) return;
    try {
        const res = await fetch('/api/show-theme', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ theme })
        });
        const data = await res.json();
        if (data.theme) {
            input.classList.add('active');
            document.getElementById('set-theme-btn').classList.add('hidden');
            document.getElementById('clear-theme-btn').classList.remove('hidden');
        }
    } catch (e) {
        console.error('Failed to set show theme:', e);
    }
}

async function clearShowTheme() {
    try {
        await fetch('/api/show-theme', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ theme: '' })
        });
        const input = document.getElementById('show-theme-input');
        input.value = '';
        input.classList.remove('active');
        document.getElementById('set-theme-btn').classList.remove('hidden');
        document.getElementById('clear-theme-btn').classList.add('hidden');
    } catch (e) {
        console.error('Failed to clear show theme:', e);
    }
}
```

Wire up event listeners (in the DOMContentLoaded or init block):

```javascript
document.getElementById('set-theme-btn').addEventListener('click', setShowTheme);
document.getElementById('clear-theme-btn').addEventListener('click', clearShowTheme);
document.getElementById('show-theme-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') setShowTheme();
});
loadShowTheme();
```

**Step 4: Test in browser**

1. Open http://localhost:8000
2. Type a theme in the input, click Set — input should highlight amber, Set button hides, X button appears
3. Click X — input clears, reverts to normal state
4. Refresh page — theme should persist (loaded from API)

**Step 5: Commit**

```bash
git add frontend/index.html frontend/css/style.css frontend/js/app.js
git commit -m "Add show theme input to header bar"
```

---

### Task 5: Clear theme on new session

**Files:**
- Modify: `backend/main.py` (session reset logic)

**Step 1: Find session reset and ensure theme clears**

Search for where `session = Session()` is called (the new session endpoint). The theme field is already in `__init__` with default `""`, so creating a new Session automatically clears it. No code change needed here — but verify the frontend reloads the theme on new session.

In the frontend, find the new session button handler and add `loadShowTheme()` after the session reset call completes, so the UI reflects the cleared theme.

**Step 2: Commit (if any changes needed)**

```bash
git add frontend/js/app.js
git commit -m "Reload theme state on new session"
```
