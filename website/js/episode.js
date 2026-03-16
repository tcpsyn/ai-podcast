const FEED_URL = '/feed';

function formatDate(dateStr) {
  return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function parseDuration(raw) {
  if (!raw) return '';
  if (raw.includes(':')) {
    const parts = raw.split(':').map(Number);
    let t = 0;
    if (parts.length === 3) t = parts[0]*3600 + parts[1]*60 + parts[2];
    else if (parts.length === 2) t = parts[0]*60 + parts[1];
    return `${Math.round(t/60)} min`;
  }
  const sec = parseInt(raw, 10);
  return isNaN(sec) ? '' : `${Math.round(sec/60)} min`;
}

function stripHtml(html) {
  const div = document.createElement('div');
  div.innerHTML = html || '';
  return div.textContent || '';
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Get slug from URL
const params = new URLSearchParams(window.location.search);
const slug = params.get('slug');

if (!slug) {
  document.getElementById('ep-title').textContent = 'Episode not found';
  document.getElementById('transcript-body').innerHTML = '<p>No episode specified. <a href="/">Go back to episodes.</a></p>';
} else {
  loadEpisode(slug);
}

async function loadEpisode(slug) {
  try {
    const res = await fetch(FEED_URL);
    const xml = await res.text();
    const parser = new DOMParser();
    const doc = parser.parseFromString(xml, 'text/xml');
    const items = doc.querySelectorAll('item');

    let episode = null;
    for (const item of items) {
      const link = item.querySelector('link')?.textContent || '';
      const itemSlug = link.split('/episodes/').pop()?.replace(/\/$/, '');
      if (itemSlug === slug) {
        episode = {
          title: item.querySelector('title')?.textContent || 'Untitled',
          description: item.querySelector('description')?.textContent || '',
          audioUrl: item.querySelector('enclosure')?.getAttribute('url') || '',
          pubDate: item.querySelector('pubDate')?.textContent || '',
          duration: item.getElementsByTagNameNS('http://www.itunes.com/dtds/podcast-1.0.dtd', 'duration')[0]?.textContent || '',
          episodeNum: item.getElementsByTagNameNS('http://www.itunes.com/dtds/podcast-1.0.dtd', 'episode')[0]?.textContent || '',
        };
        break;
      }
    }

    if (!episode) {
      document.getElementById('ep-title').textContent = 'Episode not found';
      document.getElementById('transcript-body').innerHTML = '<p>Could not find this episode. <a href="/">Go back to episodes.</a></p>';
      return;
    }

    // Populate header
    const metaParts = [
      episode.episodeNum ? `Episode ${episode.episodeNum}` : '',
      episode.pubDate ? formatDate(episode.pubDate) : '',
      parseDuration(episode.duration),
    ].filter(Boolean).join(' \u00b7 ');

    document.getElementById('ep-meta').textContent = metaParts;
    document.getElementById('ep-title').textContent = episode.title;
    document.getElementById('ep-desc').textContent = stripHtml(episode.description || '');

    // Update page meta
    document.title = `${episode.title} — Luke at the Roost`;
    document.getElementById('page-description')?.setAttribute('content', `Full transcript of ${episode.title} from Luke at the Roost.`);
    document.getElementById('og-title')?.setAttribute('content', episode.title);
    document.getElementById('og-description')?.setAttribute('content', stripHtml(episode.description).slice(0, 200));
    const canonicalUrl = `https://lukeattheroost.com/episode.html?slug=${slug}`;
    document.getElementById('page-canonical')?.setAttribute('href', canonicalUrl);
    document.getElementById('og-url')?.setAttribute('content', canonicalUrl);
    document.getElementById('tw-title')?.setAttribute('content', episode.title);
    document.getElementById('tw-description')?.setAttribute('content', stripHtml(episode.description).slice(0, 200));

    // Update JSON-LD structured data
    const jsonLd = document.getElementById('episode-jsonld');
    if (jsonLd) {
      const ld = JSON.parse(jsonLd.textContent);
      ld.name = episode.title;
      ld.url = canonicalUrl;
      ld.description = stripHtml(episode.description).slice(0, 300);
      if (episode.pubDate) ld.datePublished = new Date(episode.pubDate).toISOString().split('T')[0];
      if (episode.episodeNum) ld.episodeNumber = parseInt(episode.episodeNum, 10);
      if (episode.audioUrl) {
        ld.associatedMedia = {
          "@type": "MediaObject",
          "contentUrl": episode.audioUrl
        };
      }
      jsonLd.textContent = JSON.stringify(ld);
    }

    // Play button
    if (episode.audioUrl) {
      const playBtn = document.getElementById('ep-play-btn');
      playBtn.style.display = 'inline-flex';
      playBtn.addEventListener('click', () => {
        audio.src = episode.audioUrl;
        audio.play();
        playerTitle.textContent = episode.title;
        stickyPlayer.classList.add('active');
      });
    }
  } catch (e) {
    document.getElementById('ep-title').textContent = 'Error loading episode';
  }

  // Fetch transcript
  try {
    const txRes = await fetch(`/transcripts/${slug}.txt`);
    if (!txRes.ok) throw new Error('Not found');
    const text = await txRes.text();
    const paragraphs = text.split(/\n\n+/).filter(Boolean);
    const html = paragraphs.map(p => {
      const escaped = escapeHtml(p);
      const labeled = escaped.replace(/^([A-Z][A-Z\s'\-]+?):\s*/, '<span class="speaker-label">$1:</span> ');
      return `<p>${labeled.replace(/\n/g, '<br>')}</p>`;
    }).join('');
    document.getElementById('transcript-body').innerHTML = html;
  } catch (e) {
    document.getElementById('transcript-body').innerHTML = '<p class="transcript-unavailable">Transcript not yet available for this episode.</p>';
  }
}
