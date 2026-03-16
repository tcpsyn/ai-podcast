const CLIPS_JSON_URL = '/data/clips.json';

const clipPlaySVG = '<svg viewBox="0 0 24 24" fill="#fff"><path d="M8 5v14l11-7z"/></svg>';
const clipShareSVG = '<svg viewBox="0 0 24 24" fill="#fff"><path d="M16 5l-1.42 1.42-1.59-1.59V16h-2V4.83L9.42 6.42 8 5l4-4 4 4zm4 5v11a2 2 0 01-2 2H6a2 2 0 01-2-2V10a2 2 0 012-2h3v2H6v11h12V10h-3V8h3a2 2 0 012 2z"/></svg>';

function escapeHTML(str) {
  const el = document.createElement('span');
  el.textContent = str;
  return el.innerHTML;
}

async function shareClipContent(title, url, btn) {
  if (navigator.share) {
    try {
      await navigator.share({ title, url });
      return;
    } catch (e) {
      if (e.name === 'AbortError') return;
    }
  }
  try {
    await navigator.clipboard.writeText(url);
    const orig = btn.innerHTML;
    btn.innerHTML = 'Copied!';
    btn.classList.add('share-copied');
    setTimeout(() => { btn.innerHTML = orig; btn.classList.remove('share-copied'); }, 2000);
  } catch (e) {
    prompt('Copy this link:', url);
  }
}

function renderClipCard(clip, featured) {
  const card = document.createElement('div');
  card.className = 'clip-card' + (featured ? ' clip-card-featured' : '');

  const youtubeId = (clip.youtube_id || '').replace(/[^a-zA-Z0-9_-]/g, '');
  if (youtubeId) card.dataset.youtubeId = youtubeId;
  const hasVideo = !!youtubeId;
  const epLabel = clip.episode_number ? `Episode ${Number(clip.episode_number)}` : '';
  const title = escapeHTML(clip.title || '');
  const desc = escapeHTML(clip.description || '');

  const thumbImg = clip.thumbnail && /^[\w\/.-]+$/.test(clip.thumbnail)
    ? `<img class="clip-card-thumb" src="/${clip.thumbnail}" alt="${title}" loading="lazy">`
    : '';

  card.innerHTML = `
    <div class="clip-card-inner">
      ${thumbImg}
      <div class="clip-card-overlay">
        <span class="clip-episode-label">${epLabel}</span>
        <h3 class="clip-card-title">${title}</h3>
        <p class="clip-card-desc">${desc}</p>
        ${hasVideo ? `<button class="clip-play-btn" aria-label="Play clip">${clipPlaySVG}</button>` : ''}
        ${hasVideo ? `<button class="clip-share-btn" aria-label="Share clip">${clipShareSVG}</button>` : ''}
      </div>
    </div>
  `;

  if (hasVideo) {
    card.querySelector('.clip-play-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      const inner = card.querySelector('.clip-card-inner');
      inner.innerHTML = `<iframe src="https://www.youtube-nocookie.com/embed/${youtubeId}?autoplay=1&rel=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>`;
    });
    const shareBtn = card.querySelector('.clip-share-btn');
    if (shareBtn) {
      shareBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        shareClipContent(clip.title || '', `https://youtube.com/watch?v=${youtubeId}`, shareBtn);
      });
    }
  }

  return card;
}

async function fetchClips() {
  try {
    const res = await fetch(CLIPS_JSON_URL);
    if (!res.ok) throw new Error('Failed to fetch clips');
    return await res.json();
  } catch (err) {
    console.error('Error loading clips:', err);
    return [];
  }
}

async function initClipsPage() {
  const clips = await fetchClips();
  if (!clips.length) return;

  const featuredContainer = document.querySelector('.clips-featured');
  const gridContainer = document.querySelector('.clips-grid');

  if (featuredContainer) {
    clips.filter(c => c.featured).forEach(clip => {
      featuredContainer.appendChild(renderClipCard(clip, true));
    });
  }

  if (gridContainer) {
    clips.filter(c => !c.featured).forEach(clip => {
      gridContainer.appendChild(renderClipCard(clip, false));
    });
  }
}

async function renderFeaturedClipsInline(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const clips = await fetchClips();
  const featured = clips.filter(c => c.featured);

  featured.forEach(clip => {
    container.appendChild(renderClipCard(clip, true));
  });
}

// Auto-init if clips page containers exist
if (document.querySelector('.clips-featured') || document.querySelector('.clips-grid')) {
  initClipsPage();
}
