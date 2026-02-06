#!/bin/bash
# AI Radio Show - Server Runner with restart support

PORT=8000
LOG_FILE="/tmp/ai-radio-show.log"
RESTART_FLAG="/tmp/ai-radio-show.restart"
STOP_FLAG="/tmp/ai-radio-show.stop"

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Cleanup old flags
rm -f "$RESTART_FLAG" "$STOP_FLAG"

# Check if port is already in use
if lsof -i ":$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    EXISTING_PID=$(lsof -i ":$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1)
    echo "ERROR: Port $PORT is already in use by PID $EXISTING_PID"
    echo "Run: kill $EXISTING_PID"
    exit 1
fi

kill_server() {
    local pid=$1
    if ! kill -0 "$pid" 2>/dev/null; then
        return
    fi
    kill "$pid" 2>/dev/null
    # Wait up to 5 seconds for graceful shutdown
    for i in $(seq 1 10); do
        if ! kill -0 "$pid" 2>/dev/null; then
            return
        fi
        sleep 0.5
    done
    # Force kill if still alive
    echo "[$(date)] Server didn't stop gracefully, force killing..." | tee -a "$LOG_FILE"
    kill -9 "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
}

echo "AI Radio Show Server Runner"
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo ""

# Handle Ctrl+C
trap 'echo ""; echo "[$(date)] Interrupted" | tee -a "$LOG_FILE"; kill_server $SERVER_PID; exit 0' INT TERM

while true; do
    echo "[$(date)] Starting server on port $PORT..." | tee -a "$LOG_FILE"

    # Start uvicorn directly (not through tee pipe so we get the real PID)
    python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT >> "$LOG_FILE" 2>&1 &
    SERVER_PID=$!

    # Wait for server to exit or restart signal
    while kill -0 $SERVER_PID 2>/dev/null; do
        if [ -f "$RESTART_FLAG" ]; then
            echo "[$(date)] Restart requested..." | tee -a "$LOG_FILE"
            rm -f "$RESTART_FLAG"
            kill_server $SERVER_PID
            sleep 1
            break
        fi

        if [ -f "$STOP_FLAG" ]; then
            echo "[$(date)] Stop requested..." | tee -a "$LOG_FILE"
            rm -f "$STOP_FLAG"
            kill_server $SERVER_PID
            echo "[$(date)] Server stopped." | tee -a "$LOG_FILE"
            exit 0
        fi

        sleep 1
    done

    # Check if we should restart or exit
    if [ -f "$STOP_FLAG" ]; then
        rm -f "$STOP_FLAG"
        echo "[$(date)] Server stopped." | tee -a "$LOG_FILE"
        exit 0
    fi

    echo "[$(date)] Restarting in 2 seconds..." | tee -a "$LOG_FILE"
    sleep 2
done
