#!/bin/bash
# Deploy podcast_stats.py to NAS as a long-running Docker container that updates hourly.
#
# Usage: ./deploy_stats_cron.sh

set -e

NAS_HOST="mmgnas-10g"
NAS_USER="luke"
NAS_PORT="8001"
DOCKER_BIN="/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker"
DEPLOY_DIR="/share/CACHEDEV1_DATA/podcast-stats"
CONTAINER_NAME="podcast-stats"

echo "Deploying podcast stats to NAS..."

# Create deploy dir and copy files
ssh -p "$NAS_PORT" "$NAS_USER@$NAS_HOST" "mkdir -p $DEPLOY_DIR"
scp -P "$NAS_PORT" podcast_stats.py "$NAS_USER@$NAS_HOST:$DEPLOY_DIR/podcast_stats.py"

# Create Dockerfile locally, then copy it over (NAS /tmp is tiny)
TMPFILE=$(mktemp)
cat > "$TMPFILE" << 'DOCKERFILE'
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz | tar xz --strip-components=1 -C /usr/local/bin docker/docker
RUN pip install --no-cache-dir requests yt-dlp
COPY podcast_stats.py /app/podcast_stats.py
COPY run_loop.sh /app/run_loop.sh
RUN chmod +x /app/run_loop.sh
WORKDIR /app
CMD ["/app/run_loop.sh"]
DOCKERFILE
scp -P "$NAS_PORT" "$TMPFILE" "$NAS_USER@$NAS_HOST:$DEPLOY_DIR/Dockerfile"
rm "$TMPFILE"

# Create the loop script
TMPFILE=$(mktemp)
cat > "$TMPFILE" << 'LOOPSCRIPT'
#!/bin/sh
echo "podcast-stats: starting hourly loop"
while true; do
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') Running stats update..."
    if python podcast_stats.py --json --upload 2>&1; then
        [ -n "$HEARTBEAT_URL" ] && curl -s "${HEARTBEAT_URL}?status=up&msg=OK" > /dev/null
        echo "  ...done, heartbeat sent"
    else
        echo "  ...failed, will retry next hour"
    fi
    echo "Sleeping 1 hour..."
    sleep 3600
done
LOOPSCRIPT
scp -P "$NAS_PORT" "$TMPFILE" "$NAS_USER@$NAS_HOST:$DEPLOY_DIR/run_loop.sh"
rm "$TMPFILE"

echo "Building Docker image on NAS..."
ssh -p "$NAS_PORT" "$NAS_USER@$NAS_HOST" \
    "TMPDIR=$DEPLOY_DIR $DOCKER_BIN build -t $CONTAINER_NAME $DEPLOY_DIR"

# Stop old container if running
ssh -p "$NAS_PORT" "$NAS_USER@$NAS_HOST" \
    "$DOCKER_BIN rm -f $CONTAINER_NAME 2>/dev/null || true"

# Run as a daemon with auto-restart (survives reboots)
echo "Starting container..."
ssh -p "$NAS_PORT" "$NAS_USER@$NAS_HOST" \
    "$DOCKER_BIN run -d --name $CONTAINER_NAME --restart unless-stopped --network host -v /var/run/docker.sock:/var/run/docker.sock $CONTAINER_NAME"

echo "Verifying..."
sleep 3
ssh -p "$NAS_PORT" "$NAS_USER@$NAS_HOST" \
    "$DOCKER_BIN logs $CONTAINER_NAME 2>&1 | tail -5"

echo ""
echo "Done! Container runs hourly in a loop with --restart unless-stopped."
echo "  Logs: ssh -p $NAS_PORT $NAS_USER@$NAS_HOST '$DOCKER_BIN logs -f $CONTAINER_NAME'"
