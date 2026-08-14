#!/bin/bash
# Deploy SearXNG to the QNAP NAS (mmgnas) as Devon's always-on web-search backend.
# Public image — no build. Ships a settings.yml that enables the JSON API and
# disables the bot limiter so the backend can query format=json programmatically.
set -e

NAS_PORT="8001"
NAS_USER="luke"
NAS_HOST="${NAS_HOST:-mmgnas}"          # override with NAS_HOST=mmgnas-10g for wired
DOCKER="/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker"
DEPLOY_DIR="/share/CACHEDEV1_DATA/docker/searxng"
CONTAINER="searxng"
HOST_PORT="8888"
IMAGE="searxng/searxng:latest"

SSH="ssh -p $NAS_PORT $NAS_USER@$NAS_HOST"

echo "==> Ensuring deploy dir $DEPLOY_DIR"
$SSH "mkdir -p $DEPLOY_DIR"

# Generate settings.yml only if absent (preserve secret_key across redeploys)
if ! $SSH "test -f $DEPLOY_DIR/settings.yml"; then
  echo "==> Writing settings.yml (first deploy)"
  SECRET=$(openssl rand -hex 32)
  TMP=$(mktemp)
  cat > "$TMP" <<EOF
use_default_settings: true
server:
  secret_key: "$SECRET"
  bind_address: "0.0.0.0"
  limiter: false
  image_proxy: true
search:
  safe_search: 0
  formats:
    - html
    - json
EOF
  scp -P "$NAS_PORT" "$TMP" "$NAS_USER@$NAS_HOST:$DEPLOY_DIR/settings.yml"
  rm "$TMP"
else
  echo "==> settings.yml already present — leaving it (and its secret_key) intact"
fi

echo "==> Pulling $IMAGE"
$SSH "$DOCKER pull $IMAGE"

echo "==> (Re)starting container"
$SSH "$DOCKER rm -f $CONTAINER 2>/dev/null || true"
$SSH "$DOCKER run -d --name $CONTAINER --restart unless-stopped \
  -p $HOST_PORT:8080 \
  -v $DEPLOY_DIR:/etc/searxng \
  -e SEARXNG_BASE_URL=http://$NAS_HOST:$HOST_PORT/ \
  $IMAGE"

echo "==> Verifying"
sleep 6
$SSH "$DOCKER ps --filter name=$CONTAINER --format '{{.Status}}'"
$SSH "$DOCKER logs $CONTAINER 2>&1 | tail -15"
