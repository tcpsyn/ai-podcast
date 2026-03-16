const FEED_URL = '/feed';
const EPISODES_PER_PAGE = 10;

const episodesList = document.getElementById('episodes-list');

let currentEpisodeCard = null;
let allEpisodes = [];
let displayedCount = 0;

function escapeAttr(str) {
  return str.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Format duration from itunes:duration (could be seconds or HH:MM:SS)
function parseDuration(raw) {
  if (!raw) return '';
  if (raw.includes(':')) {
    const parts = raw.split(':').map(Number);
    let totalSec = 0;
    if (parts.length === 3) totalSec = parts[0] * 3600 + parts[1] * 60 + parts[2];
    else if (parts.length === 2) totalSec = parts[0] * 60 + parts[1];
    return `${Math.round(totalSec / 60)} min`;
  }
  const sec = parseInt(raw, 10);
  if (isNaN(sec)) return '';
  return `${Math.round(sec / 60)} min`;
}

// Format date nicely
function formatDate(dateStr) {
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

// Strip HTML tags and truncate at word boundary (returns escaped text safe for innerHTML)
function truncate(html, maxLen) {
  const div = document.createElement('div');
  div.innerHTML = html || '';
  const text = div.textContent || '';
  let result;
  if (text.length <= maxLen) {
    result = text;
  } else {
    const truncated = text.slice(0, maxLen);
    const lastSpace = truncated.lastIndexOf(' ');
    result = (lastSpace > maxLen * 0.5 ? truncated.slice(0, lastSpace) : truncated).trimEnd() + '...';
  }
  return escapeAttr(result);
}

// SVG icons
const playSVG = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
const pauseSVG = '<svg viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>';

// Fetch with timeout
function fetchWithTimeout(url, ms = 8000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), ms);
  return fetch(url, { signal: controller.signal }).finally(() => clearTimeout(timeout));
}

// Fetch and parse RSS feed
async function fetchEpisodes() {
  let xml;
  const maxRetries = 2;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const res = await fetchWithTimeout(FEED_URL);
      if (!res.ok) throw new Error('Fetch failed');
      xml = await res.text();
      if (xml.includes('<item>')) break;
      throw new Error('Invalid response');
    } catch (err) {
      if (attempt === maxRetries) {
        episodesList.innerHTML = '<div class="episodes-error">Unable to load episodes. <a href="https://podcast.macneilmediagroup.com/@LukeAtTheRoost/feed.xml" target="_blank">View RSS feed</a></div>';
        return;
      }
    }
  }

  const parser = new DOMParser();
  const doc = parser.parseFromString(xml, 'text/xml');
  const items = doc.querySelectorAll('item');

  if (items.length === 0) {
    episodesList.innerHTML = '<div class="episodes-error">No episodes found.</div>';
    return;
  }

  const episodes = Array.from(items).map((item) => {
    const title = item.querySelector('title')?.textContent || 'Untitled';
    const description = item.querySelector('description')?.textContent || '';
    const enclosure = item.querySelector('enclosure');
    const audioUrl = enclosure?.getAttribute('url') || '';
    const pubDate = item.querySelector('pubDate')?.textContent || '';
    const duration = item.getElementsByTagNameNS('http://www.itunes.com/dtds/podcast-1.0.dtd', 'duration')[0]?.textContent || '';
    const episodeNum = item.getElementsByTagNameNS('http://www.itunes.com/dtds/podcast-1.0.dtd', 'episode')[0]?.textContent || '';
    const link = item.querySelector('link')?.textContent || '';

    return { title, description, audioUrl, pubDate, duration, episodeNum, link };
  });

  renderEpisodes(episodes);
}

function renderEpisodes(episodes) {
  allEpisodes = episodes;
  displayedCount = 0;
  episodesList.innerHTML = '';
  showMoreEpisodes();
}

function createEpisodeCard(ep) {
  const card = document.createElement('div');
  card.className = 'episode-card';

  const epLabel = ep.episodeNum ? `Ep ${ep.episodeNum}` : '';
  const dateStr = ep.pubDate ? formatDate(ep.pubDate) : '';
  const durStr = parseDuration(ep.duration);

  const metaParts = [epLabel, dateStr, durStr].filter(Boolean).join(' &middot; ');
  const epSlug = ep.link ? ep.link.split('/episodes/').pop()?.replace(/\/$/, '') : '';

  card.innerHTML = `
    <button class="episode-play-btn" aria-label="Play ${escapeAttr(ep.title)}" data-url="${escapeAttr(ep.audioUrl)}" data-title="${escapeAttr(ep.title)}">
      ${playSVG}
    </button>
    <div class="episode-info">
      <div class="episode-meta">${metaParts}</div>
      <div class="episode-title">${escapeAttr(ep.title)}</div>
      <div class="episode-desc">${truncate(ep.description, 150)}</div>
      ${epSlug ? `<a href="/episode.html?slug=${encodeURIComponent(epSlug)}" class="episode-transcript-link">Read Transcript</a>` : ''}
    </div>
  `;

  const btn = card.querySelector('.episode-play-btn');
  btn.addEventListener('click', () => playEpisode(ep.audioUrl, ep.title, card, btn));

  return card;
}

function showMoreEpisodes() {
  const batch = allEpisodes.slice(displayedCount, displayedCount + EPISODES_PER_PAGE);
  batch.forEach((ep) => {
    episodesList.appendChild(createEpisodeCard(ep));
  });
  displayedCount += batch.length;

  const existing = document.getElementById('load-more-btn');
  if (existing) existing.remove();

  if (displayedCount < allEpisodes.length) {
    const btn = document.createElement('button');
    btn.id = 'load-more-btn';
    btn.className = 'load-more-btn';
    btn.textContent = `Load More (${allEpisodes.length - displayedCount} remaining)`;
    btn.addEventListener('click', showMoreEpisodes);
    episodesList.after(btn);
  }
}

function playEpisode(url, title, card, btn) {
  if (!url) return;

  if (audio.src === url || audio.src === encodeURI(url)) {
    if (audio.paused) {
      audio.play();
    } else {
      audio.pause();
    }
    return;
  }

  if (currentEpisodeCard) {
    const prevBtn = currentEpisodeCard.querySelector('.episode-play-btn');
    if (prevBtn) {
      prevBtn.innerHTML = playSVG;
      prevBtn.classList.remove('playing');
    }
  }

  currentEpisodeCard = card;
  audio.src = url;
  audio.play();

  playerTitle.textContent = title;
  stickyPlayer.classList.add('active');
}

// Episode card icon sync (sticky player icons handled by player.js)
function updateCardIcon(playing) {
  if (currentEpisodeCard) {
    const btn = currentEpisodeCard.querySelector('.episode-play-btn');
    if (btn) {
      btn.innerHTML = playing ? pauseSVG : playSVG;
      btn.classList.toggle('playing', playing);
    }
  }
}

audio.addEventListener('play', () => updateCardIcon(true));
audio.addEventListener('pause', () => updateCardIcon(false));
audio.addEventListener('ended', () => updateCardIcon(false));

// Testimonials Slider
function initTestimonials() {
  const track = document.getElementById('testimonials-track');
  const dotsContainer = document.getElementById('testimonials-dots');
  const cards = track.querySelectorAll('.testimonial-card');
  if (!cards.length) return;

  let currentIndex = 0;
  let autoplayTimer = null;
  const maxIndex = () => Math.max(0, cards.length - 1);

  function buildDots() {
    dotsContainer.innerHTML = '';
    for (let i = 0; i < cards.length; i++) {
      const dot = document.createElement('button');
      dot.className = 'testimonial-dot' + (i === currentIndex ? ' active' : '');
      dot.setAttribute('aria-label', `Testimonial ${i + 1}`);
      dot.addEventListener('click', () => goTo(i));
      dotsContainer.appendChild(dot);
    }
  }

  function updatePosition() {
    const cardWidth = cards[0].offsetWidth;
    track.style.transform = `translateX(-${currentIndex * cardWidth}px)`;
    dotsContainer.querySelectorAll('.testimonial-dot').forEach((d, i) => {
      d.classList.toggle('active', i === currentIndex);
    });
  }

  function goTo(index) {
    currentIndex = Math.max(0, Math.min(index, maxIndex()));
    updatePosition();
    resetAutoplay();
  }

  function next() {
    goTo(currentIndex >= maxIndex() ? 0 : currentIndex + 1);
  }

  function resetAutoplay() {
    clearInterval(autoplayTimer);
    autoplayTimer = setInterval(next, 10000);
  }

  // Touch/swipe support
  let touchStartX = 0;
  let touchDelta = 0;
  track.addEventListener('touchstart', (e) => {
    touchStartX = e.touches[0].clientX;
    touchDelta = 0;
    clearInterval(autoplayTimer);
  }, { passive: true });

  track.addEventListener('touchmove', (e) => {
    touchDelta = e.touches[0].clientX - touchStartX;
  }, { passive: true });

  track.addEventListener('touchend', () => {
    if (Math.abs(touchDelta) > 50) {
      touchDelta < 0 ? goTo(currentIndex + 1) : goTo(currentIndex - 1);
    }
    resetAutoplay();
  });

  // Recalculate on resize
  window.addEventListener('resize', () => {
    if (currentIndex > maxIndex()) currentIndex = maxIndex();
    buildDots();
    updatePosition();
  });

  buildDots();
  updatePosition();
  resetAutoplay();
}

// On-Air Status
function checkOnAir() {
  fetch(`https://cdn.lukeattheroost.com/status.json?_=${Date.now()}`)
    .then(r => r.json())
    .then(data => {
      const badge = document.getElementById('on-air-badge');
      const offBadge = document.getElementById('off-air-badge');
      const phone = document.getElementById('phone-section');
      const live = !!data.on_air;
      if (badge) badge.classList.toggle('visible', live);
      if (offBadge) offBadge.classList.toggle('hidden', live);
      if (phone) phone.classList.toggle('live', live);
    })
    .catch(() => {});
}

// Init
fetchEpisodes();
initTestimonials();
checkOnAir();
setInterval(checkOnAir, 15000);
