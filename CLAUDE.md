# AI Podcast - Project Instructions

## Git Remote (Gitea)
- **Repo**: `git@gitea-nas:luke/ai-podcast.git`
- **Web**: http://mmgnas:3000/luke/ai-podcast
- **SSH Host**: `gitea-nas` (configured in ~/.ssh/config)
  - HostName: `mmgnas` (use `mmgnas-10g` if wired connection issues)
  - Port: `2222`
  - User: `git`
  - IdentityFile: `~/.ssh/gitea_mmgnas`

## NAS Access
- **Hostname**: `mmgnas` (wireless) or `mmgnas-10g` (wired/10G)
- **SSH Port**: 8001
- **User**: luke
- **Docker path**: `/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker`

## Castopod (Podcast Publishing)
- **URL**: https://podcast.macneilmediagroup.com
- **Podcast handle**: `@LukeAtTheRoost`
- **API Auth**: Basic auth (admin/REDACTED_CASTOPOD_PASSWORD)
- **Container**: `castopod-castopod-1`
- **Database**: `castopod-mariadb-1` (user: castopod, db: castopod)

## Running the App
```bash
# Start backend
cd /Users/lukemacneil/ai-podcast
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Or use run.sh
./run.sh
```

## Publishing Episodes
```bash
python publish_episode.py ~/Desktop/episode.mp3
```

## Environment Variables
Required in `.env`:
- OPENROUTER_API_KEY
- ELEVENLABS_API_KEY (optional)
- INWORLD_API_KEY (for Inworld TTS)
