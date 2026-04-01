#!/usr/bin/env bash
# =============================================================================
# run_slicing.sh
# Launches ns-3 5G slicing simulation + Python BWP Manager controller
#
# Usage:
#   chmod +x run_slicing.sh
#   ./run_slicing.sh [SIM_TIME] [NS3_ROOT]
#
# Examples:
#   ./run_slicing.sh 50
#   ./run_slicing.sh 100 /opt/ns-allinone-3.40/ns-3.40
# =============================================================================

set -e

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SIM_TIME=${1:-50}
NS3_ROOT=${2:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}

GYM_PORT=5555
STEP_INTERVAL=0.1
N_URLLC=5
N_EMBB=10
N_MMTC=30
OUTPUT_DIR="./results"

SCRATCH_NAME="sdn_bwp"

# ──────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ──────────────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log_info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_section() { echo -e "\n${BOLD}${CYAN}=== $* ===${NC}\n"; }

# ──────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ──────────────────────────────────────────────────────────────────────────────
log_section "5G Network Slicing — SDN-BWP Simulation"
log_info "Architecture   : Single gNB, 3 BWPs (mixed numerology)"
log_info "Algorithms     : BWP Manager (Python) + Pre-processor + BWP Mux (C++)"
log_info "Sim time       : ${SIM_TIME}s"
log_info "ns-3 root      : $NS3_ROOT"
log_info "Gym port       : $GYM_PORT"
log_info "UEs            : URLLC=$N_URLLC  eMBB=$N_EMBB  mMTC=$N_MMTC"

if [ ! -d "$NS3_ROOT" ]; then
    log_error "ns-3 root not found: $NS3_ROOT"
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    log_error "python3 not found. Install Python 3.8+."
    exit 1
fi

if ! python3 -c "import ns3gym" &>/dev/null; then
    log_warn "ns3gym Python package not found."
    log_warn "Install with: pip install ns3gym"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTPUT_DIR"

# Detect ns3gym venv (if available, use it for Python)
VENV_DIR="$NS3_ROOT/contrib/opengym/ns3gym-venv"
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python3" ]; then
    PYTHON="$VENV_DIR/bin/python3"
    log_info "Using ns3gym venv: $PYTHON"
else
    PYTHON="python3"
    log_info "Using system Python: $PYTHON"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Build
# ──────────────────────────────────────────────────────────────────────────────
log_section "Building ns-3 simulation"

pushd "$NS3_ROOT" > /dev/null

if [ ! -f "cmake-cache/build.ninja" ] && [ ! -f "build/build.ninja" ]; then
    log_info "Configuring ns-3 build..."
    python3 ns3 configure --enable-examples --enable-logs 2>&1 | tail -20
fi

# Build using cmake directly with the exact target name
# (ns3 script has issues resolving scratch subdirectory targets)
CMAKE_TARGET="scratch_sdn-bwp_sdn_bwp"
log_info "Building target: $CMAKE_TARGET ..."
cd cmake-cache && cmake --build . -j $(nproc) --target "$CMAKE_TARGET" 2>&1 | tail -30 && cd ..

BUILD_EXIT=$?
if [ $BUILD_EXIT -ne 0 ]; then
    log_error "Build failed with exit code $BUILD_EXIT"
    exit 1
fi
log_info "Build successful."
popd > /dev/null

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Launch ns-3 + Python controller
# ──────────────────────────────────────────────────────────────────────────────
log_section "Starting simulation + BWP Manager controller"

NS3_CMD="python3 $NS3_ROOT/ns3 run \
    \"scratch/sdn-bwp/sdn_bwp \
    --gymPort=$GYM_PORT \
    --simTime=$SIM_TIME \
    --stepInterval=$STEP_INTERVAL \
    --nUrllc=$N_URLLC \
    --nEmbb=$N_EMBB \
    --nMmtc=$N_MMTC \
    --verbose=true\" \
    --no-build"

PYTHON_CMD="$PYTHON $SCRIPT_DIR/agent.py \
    --port $GYM_PORT \
    --sim-time $SIM_TIME \
    --step-interval $STEP_INTERVAL \
    --n-urllc $N_URLLC \
    --n-embb $N_EMBB \
    --n-mmtc $N_MMTC \
    --output-dir $OUTPUT_DIR \
    --plot"

log_info "ns-3 command  : $NS3_CMD"
log_info "Python command: $PYTHON_CMD"

# Start ns-3 in background
log_info "Starting ns-3 simulation (background)..."
eval "$NS3_CMD" > "$OUTPUT_DIR/ns3_output.log" 2>&1 &
NS3_PID=$!
log_info "ns-3 PID: $NS3_PID"

# Wait for ns3-gym socket to open
log_info "Waiting for ns3-gym socket on port $GYM_PORT ..."
for i in $(seq 1 60); do
    if python3 -c "
import socket
s = socket.socket()
s.settimeout(1)
try:
    s.connect(('127.0.0.1', $GYM_PORT))
    s.close()
    exit(0)
except:
    exit(1)
" 2>/dev/null; then
        log_info "ns3-gym socket ready."
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# Start Python controller (foreground)
log_info "Starting Python BWP Manager controller..."
eval "$PYTHON_CMD"
PYTHON_EXIT=$?

# Wait for ns-3
wait $NS3_PID 2>/dev/null || true
NS3_EXIT=$?

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Summary
# ──────────────────────────────────────────────────────────────────────────────
log_section "Results"

if [ $PYTHON_EXIT -eq 0 ]; then
    log_info "BWP Manager controller finished successfully."
else
    log_warn "Controller exited with code $PYTHON_EXIT"
fi

if [ $NS3_EXIT -eq 0 ]; then
    log_info "ns-3 simulation finished successfully."
else
    log_warn "ns-3 exited with code $NS3_EXIT (check $OUTPUT_DIR/ns3_output.log)"
fi

log_info "Output files in: $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/" 2>/dev/null || true

echo ""
log_info "Done! Check results/ for metrics CSV, weight JSON, and plots."
