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
