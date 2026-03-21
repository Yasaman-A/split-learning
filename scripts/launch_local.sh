#!/usr/bin/bash
# Launch all CC-SFL v1 Custom Cut processes from a single terminal.
# Usage:  bash scripts/launch_local.sh
#
# Reads cut layers and client count from config.yaml.
# Starts: data HTTP server, split server, fed server, N clients.
# Ctrl+C kills everything cleanly.

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$REPO_ROOT/src"
CONFIG="$SRC_DIR/split-learning/splitfed_v1_custom_cut/config.yaml"

# ─── Python: use venv if present, otherwise whatever is on PATH ───────────────
if [ -f "$REPO_ROOT/venv/Scripts/python" ]; then
    PYTHON="$REPO_ROOT/venv/Scripts/python"
elif [ -f "$REPO_ROOT/venv/bin/python3" ]; then
    PYTHON="$REPO_ROOT/venv/bin/python3"
else
    PYTHON="python"
fi

# ─── Convert config path to Windows format (needed when using Windows Python
#     from Git Bash, which otherwise passes /c/... paths that Python can't open)
if command -v cygpath &>/dev/null; then
    CONFIG_PY=$(cygpath -w "$CONFIG")
else
    CONFIG_PY="$CONFIG"
fi

# ─── Read config values ───────────────────────────────────────────────────────
NUM_CLIENTS=$("$PYTHON" -c "import yaml; c=yaml.safe_load(open(r'$CONFIG_PY')); print(c['client_total'])")
CUT_LAYER=$("$PYTHON"  -c "import yaml; c=yaml.safe_load(open(r'$CONFIG_PY')); print(c['cut_layer'])")

# Build cut layer list: e.g. cut_layer=4, 5 clients → "4,4,4,4,4"
CUT_LAYERS=$("$PYTHON" -c "print(','.join(['$CUT_LAYER']*$NUM_CLIENTS))")

MODE="splitfed_v1_custom_cut"

echo "============================================================"
echo "  CC-SFL Local Launcher"
echo "  Model : $("$PYTHON" -c "import yaml; c=yaml.safe_load(open(r'$CONFIG_PY')); print(c.get('model_architecture','?'))")"
echo "  Loss  : $("$PYTHON" -c "import yaml; c=yaml.safe_load(open(r'$CONFIG_PY')); print(c.get('loss_function','?'))")"
echo "  Data  : $("$PYTHON" -c "import yaml; c=yaml.safe_load(open(r'$CONFIG_PY')); print(c['data_server']['output_file'])")"
echo "  Clients: $NUM_CLIENTS  |  Cut layers: $CUT_LAYERS"
echo "============================================================"
echo ""

# ─── Process tracking ─────────────────────────────────────────────────────────
PIDS=()

cleanup() {
    echo ""
    echo "Shutting down all processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "All processes stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

# All Python processes run from REPO_ROOT so that the config path
# "./src/split-learning/..." resolves correctly. SRC_DIR is added to
# PYTHONPATH so that "-m split-learning" finds the package.
export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

# ─── 1. Data HTTP server (serves .pkl files from repo root) ───────────────────
echo "[1/3] Starting data HTTP server on port 8000..."
"$PYTHON" -m http.server 8000 --directory "$REPO_ROOT" > "$REPO_ROOT/logs/http_server.log" 2>&1 &
PIDS+=($!)
echo "      PID ${PIDS[-1]} | log: logs/http_server.log"

sleep 1

# ─── 2. Split server ──────────────────────────────────────────────────────────
echo "[2/3] Starting split server..."
"$PYTHON" -m split-learning --mode $MODE --server --extra "$CUT_LAYERS" &
PIDS+=($!)
SPLIT_PID=${PIDS[-1]}
echo "      PID $SPLIT_PID"

# ─── 3. Fed server ────────────────────────────────────────────────────────────
echo "[3/3] Starting fed server..."
"$PYTHON" -m split-learning --mode $MODE --fed &
PIDS+=($!)
echo "      PID ${PIDS[-1]}"

# Wait for servers to bind their ports before launching clients
echo ""
echo "Waiting 5 seconds for servers to initialize..."
sleep 5

# ─── 4. Clients ───────────────────────────────────────────────────────────────
echo "Starting $NUM_CLIENTS clients..."
for i in $(seq 1 $NUM_CLIENTS); do
    "$PYTHON" -m split-learning --mode $MODE --client $i --extra $CUT_LAYER &
    PIDS+=($!)
    echo "  Client $i  PID ${PIDS[-1]}"
    sleep 0.5
done

echo ""
echo "All processes running. Waiting for experiment to complete..."
echo "Press Ctrl+C to abort early."
echo ""

# Wait for the split server (the process that controls rounds + patience)
wait $SPLIT_PID
echo ""
echo "Experiment complete."
echo "Logs saved to: logs/$(cat "$REPO_ROOT/logs/.run_id" 2>/dev/null || echo '(see logs/)')"

cleanup
