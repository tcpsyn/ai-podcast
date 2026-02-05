#!/bin/bash
# AI Radio Show - Server Runner with restart support

LOG_FILE="/tmp/ai-radio-show.log"
RESTART_FLAG="/tmp/ai-radio-show.restart"
STOP_FLAG="/tmp/ai-radio-show.stop"

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Cleanup old flags
rm -f "$RESTART_FLAG" "$STOP_FLAG"

echo "AI Radio Show Server Runner"
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    echo "[$(date)] Starting server..." | tee -a "$LOG_FILE"

    # Start uvicorn with output to both console and log file
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 2>&1 | tee -a "$LOG_FILE" &
    SERVER_PID=$!

    # Wait for server to exit or restart signal
    while kill -0 $SERVER_PID 2>/dev/null; do
        if [ -f "$RESTART_FLAG" ]; then
            echo "[$(date)] Restart requested..." | tee -a "$LOG_FILE"
            rm -f "$RESTART_FLAG"
            kill $SERVER_PID 2>/dev/null
            wait $SERVER_PID 2>/dev/null
            sleep 1
            break
        fi

        if [ -f "$STOP_FLAG" ]; then
            echo "[$(date)] Stop requested..." | tee -a "$LOG_FILE"
            rm -f "$STOP_FLAG"
            kill $SERVER_PID 2>/dev/null
            wait $SERVER_PID 2>/dev/null
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
