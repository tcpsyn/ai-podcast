# Website JS Infrastructure, SEO, Shared Components & Content Fixes

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix JS duplication, add worker-level social meta injection, standardize analytics proxying, improve security and UX, and clean up content/SEO issues across lukeattheroost.com.

**Architecture:** Extract shared footer into `js/footer.js`, extract shared audio player into `js/player.js`, enhance `_worker.js` to intercept social crawler requests and inject episode-specific meta tags, switch all subpages to proxied Plausible analytics, add episode pagination, fix XSS surfaces, and clean up sitemap/clips data.

**Tech Stack:** Vanilla JS, Cloudflare Pages Worker (ES module), static HTML, XML sitemap

---

### Task 1: Create shared footer component (`js/footer.js`)

**Files:**
- Create: `website/js/footer.js`

**Step 1: Write footer.js**

The footer HTML is duplicated across 7 pages (index.html:265-306, episode.html:95-136, clips.html:68-109, stats.html, privacy.html, terms.html, how-it-works.html). Extract the footer from `index.html` as the canonical version.

```js
function initFooter() {
  const footer = document.querySelector('.footer');
  if (!footer) return;

  footer.innerHTML = `
    <div class="footer-nav">
      <a href="/">Home</a>
      <a href="/how-it-works">How It Works</a>
      <a href="/clips">Clips</a>
      <a href="/stats">Stats</a>
    </div>
    <div class="footer-icons">
      <span class="footer-icons-label">Listen On</span>
      <div class="footer-icons-row">
        <a href="https://open.spotify.com/show/0ZrpMigG1fo0CCN7F4YmuF?si=f990713adce84ba4" target="_blank" rel="noopener" class="footer-icon-link" aria-label="Spotify"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/></svg></a>
        <a href="https://www.youtube.com/watch?v=xryGLifMBTY&list=PLGq4uZyNV1yYH_rcitTTPVysPbC6-7pe-" target="_blank" rel="noopener" class="footer-icon-link" aria-label="YouTube"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814z"/><path d="M9.545 15.568V8.432L15.818 12z" fill="#1a1209"/></svg></a>
        <a href="https://podcasts.apple.com/us/podcast/luke-at-the-roost/id1875205848" target="_blank" rel="noopener" class="footer-icon-link" aria-label="Apple Podcasts"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.477 2 2 6.477 2 12c0 3.293 1.592 6.214 4.05 8.04.13-.455.283-.942.457-1.393A9 9 0 0 1 3 12a9 9 0 0 1 18 0 9 9 0 0 1-3.507 7.127c.174.42.327.893.456 1.333A10 10 0 0 0 22 12c0-5.523-4.477-10-10-10zm0 4a6 6 0 0 0-6 6c0 1.87.856 3.54 2.2 4.64.196-.46.43-.91.692-1.31A4.5 4.5 0 0 1 7.5 12a4.5 4.5 0 0 1 9 0c0 1.21-.478 2.31-1.256 3.12.24.37.462.8.655 1.24A6 6 0 0 0 18 12a6 6 0 0 0-6-6zm0 4.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zM12 15c-.75 0-1.158.54-1.28 1.2-.17.94-.28 1.91-.33 2.88-.03.48.34.82.73.82h1.76c.39 0 .76-.34.73-.82-.05-.97-.16-1.94-.33-2.88-.122-.66-.53-1.2-1.28-1.2z"/></svg></a>
        <a href="https://podcast.macneilmediagroup.com/@LukeAtTheRoost/feed.xml" target="_blank" rel="noopener" class="footer-icon-link" aria-label="RSS"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M6.503 20.752c0 1.794-1.456 3.248-3.251 3.248S0 22.546 0 20.752s1.456-3.248 3.252-3.248 3.251 1.454 3.251 3.248zM.002 9.473v4.594c5.508.163 9.929 4.584 10.092 10.091h4.594C14.524 16.21 7.849 9.636.002 9.473zM.006 0v4.604C10.81 4.77 19.23 13.19 19.396 24h4.604C23.834 10.952 13.054.166.006 0z"/></svg></a>
      </div>
    </div>
    <div class="footer-icons">
      <span class="footer-icons-label">Follow</span>
      <div class="footer-icons-row">
        <a href="https://discord.gg/5CnQZxDM" target="_blank" rel="noopener" class="footer-icon-link" aria-label="Discord"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.095 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.095 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/></svg></a>
        <a href="https://www.facebook.com/profile.php?id=61588191627949" target="_blank" rel="noopener" class="footer-icon-link" aria-label="Facebook"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg></a>
        <a href="https://www.instagram.com/lukeattheroost/" target="_blank" rel="noopener" class="footer-icon-link" aria-label="Instagram"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zM12 0C8.741 0 8.333.014 7.053.072 2.695.272.273 2.69.073 7.052.014 8.333 0 8.741 0 12c0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98C8.333 23.986 8.741 24 12 24c3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98C15.668.014 15.259 0 12 0zm0 5.838a6.162 6.162 0 1 0 0 12.324 6.162 6.162 0 0 0 0-12.324zM12 16a4 4 0 1 1 0-8 4 4 0 0 1 0 8zm6.406-11.845a1.44 1.44 0 1 0 0 2.881 1.44 1.44 0 0 0 0-2.881z"/></svg></a>
        <a href="https://x.com/lukeattheroost" target="_blank" rel="noopener" class="footer-icon-link" aria-label="X"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg></a>
        <a href="https://bsky.app/profile/lukeattheroost.bsky.social" target="_blank" rel="noopener" class="footer-icon-link" aria-label="Bluesky"><svg viewBox="0 0 568 501" fill="currentColor"><path d="M123.121 33.664C188.241 82.553 258.281 181.68 284 234.873c25.719-53.192 95.759-152.32 160.879-201.21C491.866-1.611 568-28.906 568 57.947c0 17.346-9.945 145.713-15.778 166.555-20.275 72.453-94.155 90.933-159.875 79.748C507.222 323.8 536.444 388.56 473.333 453.32c-119.86 122.992-172.272-30.859-185.702-70.281-2.462-7.227-3.614-10.608-3.631-7.733-.017-2.875-1.169.506-3.631 7.733-13.43 39.422-65.842 193.273-185.702 70.281-63.111-64.76-33.89-129.52 80.986-149.071-65.72 11.185-139.6-7.295-159.875-79.748C10.945 203.659 1 75.291 1 57.946 1-28.906 76.134-1.612 123.121 33.664z"/></svg></a>
        <a href="https://mastodon.macneilmediagroup.com/@lukeattheroost" target="_blank" rel="me noopener" class="footer-icon-link" aria-label="Mastodon"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M23.268 5.313c-.35-2.578-2.617-4.61-5.304-5.004C17.51.242 15.792 0 11.813 0h-.03c-3.98 0-4.835.242-5.288.309C3.882.692 1.496 2.518.917 5.127.64 6.412.61 7.837.661 9.143c.074 1.874.088 3.745.26 5.611.118 1.24.325 2.47.62 3.68.55 2.237 2.777 4.098 4.96 4.857 2.336.792 4.849.923 7.256.38.265-.061.527-.132.786-.213.585-.184 1.27-.39 1.774-.753a.057.057 0 0 0 .023-.043v-1.809a.052.052 0 0 0-.02-.041.053.053 0 0 0-.046-.01 20.282 20.282 0 0 1-4.709.545c-2.73 0-3.463-1.284-3.674-1.818a5.593 5.593 0 0 1-.319-1.433.053.053 0 0 1 .066-.054 19.648 19.648 0 0 0 4.636.528c.164 0 .329 0 .494-.002 1.694-.042 3.48-.152 5.12-.554 2.21-.543 4.137-2.186 4.348-4.55.162-1.808.21-3.627.142-5.43-.02-.6-.168-1.874-.168-1.874z"/><path d="M19.903 7.515v5.834c0 1.226-.996 2.222-2.222 2.222h-.796c-1.226 0-2.222-.996-2.222-2.222V7.628c0-1.226.996-2.222 2.222-2.222h.796c.122 0 .242.01.36.03 1.076.164 1.862 1.098 1.862 2.192zM9.337 7.515v5.834c0 1.226-.996 2.222-2.222 2.222h-.796c-1.226 0-2.222-.996-2.222-2.222V7.628c0-1.226.996-2.222 2.222-2.222h.796c.122 0 .242.01.36.03 1.076.164 1.862 1.098 1.862 2.192z" fill="#1a1209"/></svg></a>
        <a href="https://primal.net/p/nprofile1qqswsam9cx06j7sxzpl498uquk3kgrwedxtq48j57zxkuj8fs82xtugge0wtg" target="_blank" rel="noopener" class="footer-icon-link" aria-label="Nostr"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M12.186.31a.27.27 0 0 0-.372 0C8.46 3.487 2.666 9.93 2.666 15.042c0 5.176 4.183 8.958 9.334 8.958s9.334-3.782 9.334-8.958c0-5.112-5.794-11.555-9.148-14.732z"/></svg></a>
        <a href="https://www.threads.com/@lukeattheroost" target="_blank" rel="noopener" class="footer-icon-link" aria-label="Threads"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M12.186 24h-.007c-3.581-.024-6.334-1.205-8.184-3.509C2.35 18.44 1.5 15.586 1.472 12.01v-.017c.03-3.579.879-6.43 2.525-8.482C5.845 1.205 8.6.024 12.18 0h.014c2.746.02 5.043.725 6.826 2.098 1.677 1.29 2.858 3.13 3.509 5.467l-2.04.569c-1.104-3.96-3.898-5.984-8.304-6.015-2.91.022-5.11.936-6.54 2.717C4.307 6.504 3.616 8.914 3.59 12c.025 3.086.718 5.496 2.057 7.164 1.432 1.781 3.632 2.695 6.54 2.717 2.227-.017 4.048-.59 5.413-1.703 1.428-1.163 2.076-2.645 1.925-4.403-.098-1.13-.578-2.065-1.39-2.7-.811-.636-1.905-.993-3.164-1.033a11.253 11.253 0 0 0-.04 0c-1.078.007-2.044.289-2.79.816-.68.481-1.069 1.108-1.125 1.813-.057.72.264 1.32.877 1.64.554.29 1.317.437 2.271.437l.013-.001c.652-.004 1.383-.078 2.172-.218l.386 2.022c-.947.18-1.837.273-2.643.278a10.35 10.35 0 0 1-.143 0c-1.425-.013-2.657-.284-3.66-.804-1.237-.643-1.928-1.745-1.836-2.93.099-1.258.738-2.316 1.849-3.064 1.088-.732 2.466-1.12 3.988-1.124h.05c1.644.044 3.088.528 4.178 1.398 1.133.905 1.8 2.185 1.935 3.703.2 2.258-.697 4.2-2.598 5.75-1.668 1.36-3.863 2.087-6.348 2.105z"/></svg></a>
        <a href="https://www.linkedin.com/company/luke-at-the-roost" target="_blank" rel="noopener" class="footer-icon-link" aria-label="LinkedIn"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg></a>
        <a href="https://www.tiktok.com/@luke.at.the.roost" target="_blank" rel="noopener" class="footer-icon-link" aria-label="TikTok"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M12.525.02c1.31-.02 2.61-.01 3.91-.02.08 1.53.63 3.09 1.75 4.17 1.12 1.11 2.7 1.62 4.24 1.79v4.03c-1.44-.05-2.89-.35-4.2-.97-.57-.26-1.1-.59-1.62-.93-.01 2.92.01 5.84-.02 8.75-.08 1.4-.54 2.79-1.35 3.94-1.31 1.92-3.58 3.17-5.91 3.21-1.43.08-2.86-.31-4.08-1.03-2.02-1.19-3.44-3.37-3.65-5.71-.02-.5-.03-1-.01-1.49.18-1.9 1.12-3.72 2.58-4.96 1.66-1.44 3.98-2.13 6.15-1.72.02 1.48-.04 2.96-.04 4.44-.99-.32-2.15-.23-3.02.37-.63.41-1.11 1.04-1.36 1.75-.21.51-.15 1.07-.14 1.61.24 1.64 1.82 3.02 3.5 2.87 1.12-.01 2.19-.66 2.77-1.61.19-.33.4-.67.41-1.06.1-1.79.06-3.57.07-5.36.01-4.03-.01-8.05.02-12.07z"/></svg></a>
      </div>
    </div>
    <div class="footer-projects">
      <span class="footer-projects-label">More from Luke</span>
      <div class="footer-projects-links">
        <a href="https://macneilmediagroup.com" target="_blank" rel="noopener">MacNeil Media Group</a>
        <a href="https://prints.macneilmediagroup.com" target="_blank" rel="noopener">Photography Prints</a>
        <a href="https://youtube.com/lukemacneil" target="_blank" rel="noopener">YouTube</a>
      </div>
    </div>
    <p class="footer-contact"><a href="https://ko-fi.com/lukemacneil" target="_blank" rel="noopener">Support the Show</a></p>
    <p class="footer-contact">Sales &amp; Collaboration: <a href="mailto:luke@lukeattheroost.com">luke@lukeattheroost.com</a></p>
    <p>&copy; 2026 Luke at the Roost &middot; <a href="/privacy">Privacy Policy</a> &middot; <a href="/terms">Terms of Service</a> &middot; <a href="https://monitoring.macneilmediagroup.com/status/lukeattheroost" target="_blank" rel="noopener">System Status</a></p>
  `;
}

initFooter();
```

**Step 2: Commit**

```bash
git add website/js/footer.js
git commit -m "Add shared footer component (js/footer.js)"
```

---

### Task 2: Replace inline footers with shared component

**Files:**
- Modify: `website/index.html` — replace lines 265-306 (inline footer content) with empty `<footer class="footer"></footer>`, add `<script src="js/footer.js"></script>` before closing `</body>`
- Modify: `website/episode.html` — replace lines 95-136 with empty footer, add script tag
- Modify: `website/clips.html` — replace lines 68-109 with empty footer, add script tag
- Modify: `website/stats.html` — replace inline footer with empty footer, add script tag
- Modify: `website/privacy.html` — replace inline footer with empty footer, add script tag
- Modify: `website/terms.html` — replace inline footer with empty footer, add script tag
- Modify: `website/how-it-works.html` — replace inline footer with empty footer, add script tag

**Step 1: Update each page**

For each of the 7 HTML files:
1. Replace the entire `<footer class="footer">...</footer>` block with just `<footer class="footer"></footer>`
2. Add `<script src="js/footer.js"></script>` near the end of `<body>`, before any page-specific scripts

Note: index.html's footer has slightly different nav links (no "Home" link since it IS home). The shared footer includes "Home" which is fine — clicking Home on the homepage just reloads it.

**Step 2: Verify no footer content remains inline**

Search for `footer-icons-label` in all HTML files — should only appear in `js/footer.js`.

**Step 3: Commit**

```bash
git add website/index.html website/episode.html website/clips.html website/stats.html website/privacy.html website/terms.html website/how-it-works.html
git commit -m "Replace inline footers with shared footer.js component"
```

---

### Task 3: Extract shared audio player module (`js/player.js`)

**Files:**
- Create: `website/js/player.js`

The audio player code is duplicated: `app.js:1-11,14-23,143-226` and `episode.html:159-346` (inline `<script>`). Extract the shared player logic.

**Step 1: Write player.js**

```js
const audio = document.getElementById('audio-element');
const stickyPlayer = document.getElementById('sticky-player');
const playerPlayBtn = document.getElementById('player-play-btn');
const playerTitle = document.getElementById('player-title');
const playerProgress = document.getElementById('player-progress');
const playerProgressFill = document.getElementById('player-progress-fill');
const playerTime = document.getElementById('player-time');

function formatTime(seconds) {
  if (!seconds || isNaN(seconds)) return '0:00';
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  return `${m}:${String(sec).padStart(2, '0')}`;
}

function updatePlayIcons(playing) {
  const iconPlay = playerPlayBtn.querySelector('.icon-play');
  const iconPause = playerPlayBtn.querySelector('.icon-pause');
  if (iconPlay) iconPlay.style.display = playing ? 'none' : 'block';
  if (iconPause) iconPause.style.display = playing ? 'block' : 'none';
}

audio.addEventListener('play', () => updatePlayIcons(true));
audio.addEventListener('pause', () => updatePlayIcons(false));
audio.addEventListener('ended', () => updatePlayIcons(false));
audio.addEventListener('timeupdate', () => {
  if (audio.duration) {
    playerProgressFill.style.width = (audio.currentTime / audio.duration * 100) + '%';
    playerTime.textContent = `${formatTime(audio.currentTime)} / ${formatTime(audio.duration)}`;
  }
});

playerPlayBtn.addEventListener('click', () => {
  if (audio.src) { audio.paused ? audio.play() : audio.pause(); }
});

playerProgress.addEventListener('click', (e) => {
  if (audio.duration) {
    const rect = playerProgress.getBoundingClientRect();
    audio.currentTime = ((e.clientX - rect.left) / rect.width) * audio.duration;
  }
});
```

**Step 2: Commit**

```bash
git add website/js/player.js
git commit -m "Extract shared audio player module (js/player.js)"
```

---

### Task 4: Refactor app.js and episode.html to use player.js

**Files:**
- Modify: `website/js/app.js` — remove duplicated player code (element lookups, formatTime, audio event listeners, updatePlayIcons, playerPlayBtn click, playerProgress click). Keep: FEED_URL, episode-specific logic (fetchEpisodes, renderEpisodes, playEpisode with card-specific icon toggling), formatDate, parseDuration, truncate, testimonials, on-air.
- Modify: `website/episode.html` — remove inline `<script>` block (lines 159-347), replace with `<script src="js/player.js"></script>` then `<script src="js/episode.js"></script>`
- Create: `website/js/episode.js` — episode-specific logic extracted from episode.html inline script (load episode from RSS, populate header, transcript loading, play button)
- Modify: `website/index.html` — add `<script src="js/player.js"></script>` before `app.js`

**Step 1: Refactor app.js**

Remove from app.js:
- Lines 1-11 (element lookups — now in player.js)
- Lines 14-23 (formatTime — now in player.js)
- Lines 173-226 (audio event listeners, updatePlayIcons, playerPlayBtn click, playerProgress click — now in player.js)

Keep the `currentEpisodeCard` variable and the card-specific icon toggling in `updatePlayIcons`. Since player.js handles the sticky player icons, app.js only needs to handle the episode card icons. Add a listener:

```js
audio.addEventListener('play', () => {
  if (currentEpisodeCard) {
    const btn = currentEpisodeCard.querySelector('.episode-play-btn');
    if (btn) { btn.innerHTML = pauseSVG; btn.classList.add('playing'); }
  }
});
audio.addEventListener('pause', () => {
  if (currentEpisodeCard) {
    const btn = currentEpisodeCard.querySelector('.episode-play-btn');
    if (btn) { btn.innerHTML = playSVG; btn.classList.remove('playing'); }
  }
});
audio.addEventListener('ended', () => {
  if (currentEpisodeCard) {
    const btn = currentEpisodeCard.querySelector('.episode-play-btn');
    if (btn) { btn.innerHTML = playSVG; btn.classList.remove('playing'); }
  }
});
```

**Step 2: Create episode.js**

Extract episode-specific logic from episode.html inline script. Use the global `audio`, `playerTitle`, `stickyPlayer` from player.js. Include: `formatDate`, `parseDuration`, `stripHtml`, slug parsing, `loadEpisode()`.

**Step 3: Update HTML script tags**

In `index.html`, change script loading order:
```html
<script src="js/footer.js"></script>
<script src="js/clips.js"></script>
<script>renderFeaturedClipsInline('home-clips');</script>
<script src="js/player.js"></script>
<script src="js/app.js?v=3"></script>
```

In `episode.html`, replace inline `<script>` (lines 159-347) with:
```html
<script src="js/footer.js"></script>
<script src="js/player.js"></script>
<script src="js/episode.js"></script>
```

**Step 4: Commit**

```bash
git add website/js/app.js website/js/player.js website/js/episode.js website/index.html website/episode.html
git commit -m "Deduplicate audio player code into shared player.js module"
```

---

### Task 5: Fix Plausible analytics — switch all subpages to proxied version

**Files:**
- Modify: `website/episode.html` line 51-52
- Modify: `website/clips.html` line 43-44
- Modify: `website/stats.html` line 43-44
- Modify: `website/privacy.html` line 37-38
- Modify: `website/terms.html` line 37-38
- Modify: `website/how-it-works.html` line 66-67

**Step 1: In each file, replace the direct Plausible script tag**

Replace:
```html
    <script defer data-domain="lukeattheroost.com" src="https://plausible.macneilmediagroup.com/js/script.file-downloads.hash.outbound-links.pageview-props.revenue.tagged-events.js"></script>
    <script>window.plausible = window.plausible || function() { (window.plausible.q = window.plausible.q || []).push(arguments) }</script>
```

With:
```html
    <script defer data-domain="lukeattheroost.com" data-api="/p/event" src="/p/script"></script>
    <script>window.plausible = window.plausible || function() { (window.plausible.q = window.plausible.q || []).push(arguments) }</script>
```

**Step 2: Verify**

Grep for `plausible.macneilmediagroup.com` in HTML files — should return 0 matches (only `_worker.js` should have it).

**Step 3: Commit**

```bash
git add website/episode.html website/clips.html website/stats.html website/privacy.html website/terms.html website/how-it-works.html
git commit -m "Switch all subpages to proxied Plausible analytics"
```

---

### Task 6: Worker — social crawler meta tag injection for episode pages

**Files:**
- Modify: `website/_worker.js`

**Step 1: Add social crawler detection and meta injection**

Before the `return env.ASSETS.fetch(request)` line (line 90), add a handler for `/episode.html` requests from social crawlers:

```js
// Social crawler meta injection for episode pages
if (url.pathname === "/episode.html" && url.searchParams.get("slug")) {
  const ua = (request.headers.get("User-Agent") || "").toLowerCase();
  const isCrawler = /facebookexternalhit|twitterbot|linkedinbot|slackbot|discordbot|telegrambot|whatsapp|pinterest|redditbot/i.test(ua);

  if (isCrawler) {
    const slug = url.searchParams.get("slug");

    // Fetch RSS to find episode info
    try {
      const feedResp = await fetch("https://podcast.macneilmediagroup.com/@LukeAtTheRoost/feed.xml", {
        signal: AbortSignal.timeout(5000),
      });
      if (feedResp.ok) {
        const feedXml = await feedResp.text();

        // Simple string-based extraction (no DOM parser in Workers)
        const items = feedXml.split("<item>");
        let title = "";
        let description = "";

        for (let i = 1; i < items.length; i++) {
          const item = items[i];
          const linkMatch = item.match(/<link>(.*?)<\/link>/);
          if (linkMatch) {
            const itemSlug = linkMatch[1].split("/episodes/").pop()?.replace(/\/$/, "");
            if (itemSlug === slug) {
              const titleMatch = item.match(/<title>(.*?)<\/title>/);
              title = titleMatch ? titleMatch[1].replace(/<!\[CDATA\[|\]\]>/g, "").trim() : "";
              const descMatch = item.match(/<description>(.*?)<\/description>/s);
              description = descMatch
                ? descMatch[1].replace(/<!\[CDATA\[|\]\]>/g, "").replace(/<[^>]+>/g, "").trim().slice(0, 200)
                : "";
              break;
            }
          }
        }

        if (title) {
          // Fetch the actual HTML page
          const pageResp = await env.ASSETS.fetch(request);
          let html = await pageResp.text();

          const escTitle = title.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
          const escDesc = description.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
          const canonicalUrl = `https://lukeattheroost.com/episode.html?slug=${slug}`;

          // Replace placeholder meta tags
          html = html.replace(
            /<meta property="og:title"[^>]*>/,
            `<meta property="og:title" content="${escTitle}">`
          );
          html = html.replace(
            /<meta property="og:description"[^>]*>/,
            `<meta property="og:description" content="${escDesc}">`
          );
          html = html.replace(
            /<meta property="og:url"[^>]*>/,
            `<meta property="og:url" content="${canonicalUrl}">`
          );
          html = html.replace(
            /<meta name="twitter:title"[^>]*>/,
            `<meta name="twitter:title" content="${escTitle}">`
          );
          html = html.replace(
            /<meta name="twitter:description"[^>]*>/,
            `<meta name="twitter:description" content="${escDesc}">`
          );
          html = html.replace(
            /<title[^>]*>.*?<\/title>/,
            `<title>${escTitle} — Luke at the Roost</title>`
          );

          return new Response(html, {
            status: 200,
            headers: { "Content-Type": "text/html;charset=UTF-8" },
          });
        }
      }
    } catch (e) {
      // Fall through to static page
    }
  }
}
```

**Step 2: Commit**

```bash
git add website/_worker.js
git commit -m "Add social crawler meta tag injection for episode pages"
```

---

### Task 7: Security — sanitize innerHTML XSS surfaces

**Files:**
- Modify: `website/js/episode.js` (created in Task 4)
- Modify: `website/js/app.js`

**Step 1: Fix episode description XSS in episode.js**

In the `loadEpisode` function, change line that sets description:
```js
// BEFORE (XSS):
document.getElementById('ep-desc').innerHTML = episode.description || '';

// AFTER (safe):
document.getElementById('ep-desc').textContent = stripHtml(episode.description || '');
```

**Step 2: Fix title escaping in app.js episode card rendering**

In `renderEpisodes()`, the title goes into a `data-title` attribute with basic `.replace(/"/g, '&quot;')`. Use the `escapeHTML` pattern from clips.js. Add a helper at top of app.js:

```js
function escapeAttr(str) {
  return str.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
```

Then change line 125:
```js
// BEFORE:
data-title="${ep.title.replace(/"/g, '&quot;')}"

// AFTER:
data-title="${escapeAttr(ep.title)}"
```

Also escape the title in the aria-label and visible title:
```js
<button class="episode-play-btn" aria-label="Play ${escapeAttr(ep.title)}" data-url="${escapeAttr(ep.audioUrl)}" data-title="${escapeAttr(ep.title)}">
```

And escape the visible title output:
```js
<div class="episode-title">${escapeAttr(ep.title)}</div>
```

**Step 3: Commit**

```bash
git add website/js/episode.js website/js/app.js
git commit -m "Fix XSS: sanitize innerHTML and improve attribute escaping"
```

---

### Task 8: Episode pagination — show 10, load more

**Files:**
- Modify: `website/js/app.js`

**Step 1: Modify renderEpisodes to support pagination**

```js
const EPISODES_PER_PAGE = 10;
let allEpisodes = [];
let displayedCount = 0;

function renderEpisodes(episodes) {
  allEpisodes = episodes;
  displayedCount = 0;
  episodesList.innerHTML = '';
  showMoreEpisodes();
}

function showMoreEpisodes() {
  const batch = allEpisodes.slice(displayedCount, displayedCount + EPISODES_PER_PAGE);
  batch.forEach((ep) => {
    // ... existing card creation code ...
    episodesList.appendChild(card);
  });
  displayedCount += batch.length;

  // Remove existing load-more button if present
  const existing = document.getElementById('load-more-btn');
  if (existing) existing.remove();

  // Add load-more button if there are remaining episodes
  if (displayedCount < allEpisodes.length) {
    const btn = document.createElement('button');
    btn.id = 'load-more-btn';
    btn.className = 'load-more-btn';
    btn.textContent = `Load More (${allEpisodes.length - displayedCount} remaining)`;
    btn.addEventListener('click', showMoreEpisodes);
    episodesList.after(btn);
  }
}
```

Note: The `.load-more-btn` CSS class likely needs to be created by the ui-ux task. For now, add minimal inline styling if the class doesn't exist yet. Actually, since we're told not to touch CSS, just use the class name and it will be styled later.

**Step 2: Commit**

```bash
git add website/js/app.js
git commit -m "Add episode pagination with Load More button"
```

---

### Task 9: Truncate at word boundaries

**Files:**
- Modify: `website/js/app.js`

**Step 1: Fix the truncate function**

```js
// BEFORE (line 47-53):
function truncate(html, maxLen) {
  const div = document.createElement('div');
  div.innerHTML = html || '';
  const text = div.textContent || '';
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trimEnd() + '...';
}

// AFTER:
function truncate(html, maxLen) {
  const div = document.createElement('div');
  div.innerHTML = html || '';
  const text = div.textContent || '';
  if (text.length <= maxLen) return text;
  const truncated = text.slice(0, maxLen);
  const lastSpace = truncated.lastIndexOf(' ');
  return (lastSpace > maxLen * 0.5 ? truncated.slice(0, lastSpace) : truncated).trimEnd() + '...';
}
```

The `lastSpace > maxLen * 0.5` guard ensures we don't cut too aggressively if the word boundary is very early.

**Step 2: Commit**

```bash
git add website/js/app.js
git commit -m "Fix truncate to break at word boundaries"
```

---

### Task 10: Deduplicate featured clips on clips page

**Files:**
- Modify: `website/js/clips.js`

**Step 1: Fix initClipsPage to exclude featured from "All Clips" grid**

```js
// BEFORE (line 73-77):
  if (gridContainer) {
    clips.forEach(clip => {
      gridContainer.appendChild(renderClipCard(clip, false));
    });
  }

// AFTER:
  if (gridContainer) {
    clips.filter(c => !c.featured).forEach(clip => {
      gridContainer.appendChild(renderClipCard(clip, false));
    });
  }
```

**Step 2: Commit**

```bash
git add website/js/clips.js
git commit -m "Deduplicate featured clips from All Clips grid"
```

---

### Task 11: Fix content issues — empty clip description and duplicate sitemap entry

**Files:**
- Modify: `website/data/clips.json` — episode 31 clip (line 58): add description
- Modify: `website/sitemap.xml` — remove duplicate episode 32 entry (lines 237-242)

**Step 1: Add description for episode 31 clip**

```json
{
  "title": "Started a Fight and Can't Stop Reading About Wars",
  "description": "A caller starts a fight with their partner and now can't stop obsessively reading about historical wars. Luke tries to unpack the connection.",
  "episode_number": 31,
  ...
}
```

**Step 2: Remove duplicate sitemap entry**

Remove lines 237-242 (the `episode-32-tacos-taxes-and-tense-conversations` entry). Keep `episode-32-tacos-taxes-and-tall-tales` (lines 231-236) as the canonical one, OR check RSS feed to determine which slug is correct. If both exist in the feed, keep both — but episode numbering suggests one is a duplicate/rename. Remove the second one (`tense-conversations` variant).

**Step 3: Commit**

```bash
git add website/data/clips.json website/sitemap.xml
git commit -m "Fix empty clip description and remove duplicate sitemap entry"
```

---

### Task 12: Create custom 404 page

**Files:**
- Create: `website/404.html`

**Step 1: Write 404.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Page Not Found — Luke at the Roost</title>
  <meta name="description" content="The page you're looking for doesn't exist.">
  <meta name="theme-color" content="#1a1209">
  <link rel="icon" href="favicon.ico" sizes="48x48">
  <link rel="icon" type="image/svg+xml" href="favicon.svg">
  <link rel="stylesheet" href="css/style.css?v=3">
  <script defer data-domain="lukeattheroost.com" data-api="/p/event" src="/p/script"></script>
  <script>window.plausible = window.plausible || function() { (window.plausible.q = window.plausible.q || []).push(arguments) }</script>
</head>
<body>

  <nav class="page-nav">
    <a href="/" class="nav-home">Luke at the Roost</a>
  </nav>

  <main>
    <section class="page-header">
      <h1>404 — Page Not Found</h1>
      <p class="page-subtitle">Looks like this page wandered off into the desert.</p>
    </section>

    <section class="about-section">
      <p>The page you're looking for doesn't exist or may have been moved.</p>
      <p><a href="/">Back to the show</a> &middot; <a href="/clips">Watch clips</a> &middot; <a href="/how-it-works">How it works</a></p>
    </section>
  </main>

  <footer class="footer"></footer>
  <script src="js/footer.js"></script>
</body>
</html>
```

**Step 2: Commit**

```bash
git add website/404.html
git commit -m "Add custom 404 page"
```

---

### Task 13: Enhance llms.txt

**Files:**
- Modify: `website/llms.txt`

**Step 1: Add episode listing section and structured links**

Add after the FAQ section:

```markdown
## Recent Episodes

Episodes are published daily. Each has a full transcript available at:
https://lukeattheroost.com/episode.html?slug=EPISODE-SLUG

Episode transcript URLs follow the pattern: episode-N-title-slug
Example: https://lukeattheroost.com/episode.html?slug=episode-37-secrets-lies-and-coffee-runs

## Clip Highlights

Popular clips with video:
- "I Faked Cancer to Skip a Wedding" (Episode 32) — https://youtube.com/watch?v=NUkhsPfMx9o
- "Neighbor's Roomba Breaks Into Kitchen at 2:30 AM" (Episode 26) — https://youtube.com/watch?v=J7bfT6jsykA
- "Shopping Cart Theory: Moral Test or Crazy?" (Episode 21) — https://youtube.com/watch?v=KijyJsMZfkA

## Sitemap

Full sitemap: https://lukeattheroost.com/sitemap.xml
```

**Step 2: Commit**

```bash
git add website/llms.txt
git commit -m "Enhance llms.txt with episode patterns and clip highlights"
```

---

## Execution Order & Dependencies

```
Task 1  (footer.js)         — no deps
Task 3  (player.js)         — no deps
Task 5  (analytics fix)     — no deps
Task 9  (truncate fix)      — no deps
Task 10 (clips dedup)       — no deps
Task 11 (content fixes)     — no deps
Task 12 (404 page)          — depends on Task 1 (uses footer.js)
Task 13 (llms.txt)          — no deps

Task 2  (replace footers)   — depends on Task 1
Task 4  (refactor to use player.js) — depends on Task 3
Task 6  (worker meta injection) — no deps
Task 7  (security fixes)    — depends on Task 4 (episode.js must exist)
Task 8  (pagination)        — can run anytime, modifies app.js
```

**Parallel batch 1** (independent): Tasks 1, 3, 5, 6, 9, 10, 11, 13
**Parallel batch 2** (deps resolved): Tasks 2, 4, 12
**Parallel batch 3** (deps resolved): Tasks 7, 8
