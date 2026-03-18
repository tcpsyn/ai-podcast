#!/bin/bash
# Daily backup of critical AI podcast data to NAS
# Backs up: Castopod MariaDB dump, local data/ directory, publish state
#
# Usage: ./backup.sh
# Cron:  0 3 * * * /Users/lukemacneil/code/ai-podcast/backup.sh >> /tmp/ai-podcast-backup.log 2>&1

set -euo pipefail

NAS_HOST="mmgnas"
NAS_USER="luke"
NAS_PORT="8001"
DOCKER_BIN="/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker"
BACKUP_BASE="/share/CACHEDEV1_DATA/backups/ai-podcast"
PROJECT_DIR="/Users/lukemacneil/code/ai-podcast"
DATE=$(date +%Y-%m-%d)
KEEP_DAYS=14

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') Starting backup..."

# 1. Dump Castopod MariaDB on NAS
echo "  Dumping MariaDB..."
ssh -p "$NAS_PORT" "$NAS_USER@$NAS_HOST" \
    "$DOCKER_BIN exec castopod-mariadb-1 mysqldump -u castopod --password=\$(cat /run/secrets/db_password 2>/dev/null || echo REDACTED_DB_PASSWORD) castopod" \
    > "/tmp/castopod-db-${DATE}.sql" 2>/dev/null

if [ -s "/tmp/castopod-db-${DATE}.sql" ]; then
    gzip -f "/tmp/castopod-db-${DATE}.sql"
    scp -P "$NAS_PORT" "/tmp/castopod-db-${DATE}.sql.gz" \
        "$NAS_USER@$NAS_HOST:$BACKUP_BASE/castopod-db-${DATE}.sql.gz"
    rm -f "/tmp/castopod-db-${DATE}.sql.gz"
    echo "  MariaDB dump: OK"
else
    echo "  WARNING: MariaDB dump is empty or failed"
fi

# 2. Sync data/ directory to NAS (rsync for efficiency)
echo "  Syncing data/ directory..."
rsync -az --delete \
    -e "ssh -p $NAS_PORT" \
    "$PROJECT_DIR/data/" \
    "$NAS_USER@$NAS_HOST:$BACKUP_BASE/data/"
echo "  data/ sync: OK"

# 3. Backup .env (contains API keys — critical for disaster recovery)
echo "  Backing up .env..."
scp -P "$NAS_PORT" "$PROJECT_DIR/.env" \
    "$NAS_USER@$NAS_HOST:$BACKUP_BASE/env-${DATE}.bak"
echo "  .env backup: OK"

# 4. Prune old backups
echo "  Pruning backups older than ${KEEP_DAYS} days..."
ssh -p "$NAS_PORT" "$NAS_USER@$NAS_HOST" \
    "find $BACKUP_BASE -name 'castopod-db-*.sql.gz' -mtime +${KEEP_DAYS} -delete 2>/dev/null; \
     find $BACKUP_BASE -name 'env-*.bak' -mtime +${KEEP_DAYS} -delete 2>/dev/null"
echo "  Prune: OK"

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') Backup complete."
