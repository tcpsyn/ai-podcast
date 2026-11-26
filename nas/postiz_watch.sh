#!/bin/sh
# NAS watchdog for the social publishing pipeline + QTS root disk.
# Runs from QTS cron every 5 min. Reports to an Uptime Kuma push monitor.
#   - Postiz worker hung (no Temporal pollers / stale backlog) -> restart orchestrator (max 2/hour)
#   - Posts overdue or failed in the last 3h -> alert
#   - QTS root ramdisk filling -> alert; runaway QVPN.log -> truncate
# Config: $DIR/watch.env with KUMA_PUSH_URL=https://monitoring.../api/push/<token>

DIR=$(cd "$(dirname "$0")" && pwd)
D=/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker
STATE="$DIR/restarts.log"
LOG="$DIR/watch.log"
ROOT_ALERT_PCT=92
QVPN_LOG=/var/log/network/QVPN.log
QVPN_MAX_KB=5120
MAX_RESTARTS_PER_HOUR=2
[ -f "$DIR/watch.env" ] && . "$DIR/watch.env"

problems=""
note() { problems="${problems:+$problems; }$1"; }
log() { echo "$(date '+%F %T') $1" >> "$LOG"; }

# --- Postiz worker health ---
tq=$($D exec postiz-temporal sh -c 'temporal task-queue describe --address $(hostname -i):7233 --task-queue main' 2>&1)
pollers=$(echo "$tq" | sed -n '/^Pollers:/,$p' | grep -c UNVERSIONED)
backlog_age=$(echo "$tq" | awk '$1=="UNVERSIONED" && $2=="workflow" && NR<6 {for(i=4;i<=NF-3;i++) printf "%s", $i; exit}')
# Temporal prints durations like "9h41m43.07s", "12m3s", "850ms", "0s".
age_secs=$(echo "$backlog_age" | awk '{ s=0; x=$0
  if (match(x,/[0-9.]+h/)) s+=substr(x,RSTART,RLENGTH-1)*3600
  sub(/[0-9.]+ms/,"",x)
  if (match(x,/[0-9.]+m/)) s+=substr(x,RSTART,RLENGTH-1)*60
  printf "%d", s }')
stale=0
[ "${age_secs:-0}" -ge 600 ] && stale=1

if [ "$pollers" -eq 0 ] || [ "$stale" -eq 1 ]; then
  now=$(date +%s)
  recent=$(awk -v t=$((now - 3600)) '$1 > t' "$STATE" 2>/dev/null | wc -l)
  if [ "$recent" -lt "$MAX_RESTARTS_PER_HOUR" ]; then
    $D exec postiz pm2 restart orchestrator >/dev/null 2>&1
    echo "$now" >> "$STATE"
    log "worker hung (pollers=$pollers backlog_age=$backlog_age) -> restarted orchestrator"
    note "Postiz worker hung (pollers=$pollers, backlog $backlog_age) - restarted"
  else
    log "worker hung, restart limit reached"
    note "Postiz worker hung - restarted $recent times this hour, needs a look"
  fi
fi
[ -f "$STATE" ] && tail -n 50 "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"

# --- Overdue / failed posts ---
posts=$($D exec -i postiz-postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At -F " "' <<'SQL' 2>&1
SELECT i."providerIdentifier",
       CASE WHEN p.state = 'ERROR' THEN 'failed' ELSE 'overdue' END
FROM "Post" p JOIN "Integration" i ON i.id = p."integrationId"
WHERE p."deletedAt" IS NULL
  AND ((p.state = 'QUEUE' AND p."publishDate" < now() - interval '15 minutes'
        AND p."publishDate" > now() - interval '3 days')
    OR (p.state = 'ERROR' AND p."updatedAt" > now() - interval '3 hours'))
ORDER BY 2, 1;
SQL
)
if echo "$posts" | grep -qiE "error|could not"; then
  note "post check failed: $(echo "$posts" | head -1)"
elif [ -n "$posts" ]; then
  note "posts: $(echo "$posts" | tr '\n' ',' | sed 's/,$//; s/,/, /g')"
fi

# --- QTS root ramdisk ---
if [ -f "$QVPN_LOG" ] && [ "$(du -k "$QVPN_LOG" | cut -f1)" -gt "$QVPN_MAX_KB" ]; then
  tail -c 512000 "$QVPN_LOG" > "$QVPN_LOG.tmp" && cat "$QVPN_LOG.tmp" > "$QVPN_LOG" && rm -f "$QVPN_LOG.tmp"
  log "truncated runaway QVPN.log"
  note "QVPN.log was over ${QVPN_MAX_KB}KB again (truncated) - qvpnd may be crash-looping"
fi
root_pct=$(df -P / | awk 'NR==2 {sub("%","",$5); print $5}')
[ "${root_pct:-0}" -ge "$ROOT_ALERT_PCT" ] && note "QTS root disk ${root_pct}% full"

# --- Report ---
[ -z "$KUMA_PUSH_URL" ] && { [ -n "$problems" ] && log "DOWN $problems"; exit 0; }
if [ -z "$problems" ]; then
  status=up; msg="OK root=${root_pct}% pollers=$pollers"
else
  status=down; msg="$problems"
  log "DOWN $msg"
fi
enc=$(printf '%s' "$msg" | sed 's/%/%25/g; s/ /%20/g; s/&/%26/g; s/#/%23/g; s/+/%2B/g; s/;/%3B/g; s/,/%2C/g; s/=/%3D/g')
curl -fsS -m 20 "$KUMA_PUSH_URL?status=$status&msg=$enc&ping=" >/dev/null 2>&1 || log "kuma push failed"
