#!/usr/bin/env bash
# =============================================================================
# run_slicing.sh
# Launches ns-3 simulation + Python heuristic controller
#
# Usage:
#   chmod +x run_slicing.sh
#   ./run_slicing.sh [STRATEGY] [SIM_TIME] [NS3_ROOT]
#
#   STRATEGY  : threshold_reactive | priority_based |
#               proportional_deficit | weighted_fairness
#   SIM_TIME  : simulation duration in seconds (default 50)
#   NS3_ROOT  : path to your ns-3 root directory (default ~/ns-3-dev)
#
# Examples:
#   ./run_slicing.sh proportional_deficit 100
#   ./run_slicing.sh weighted_fairness 50 /opt/ns-3.40
# =============================================================================

set -e   # exit on error

# -----------------------------------------------------------------------
# Configuration — adjust to match your installation
# -----------------------------------------------------------------------
STRATEGY=${1:-"proportional_deficit"}
SIM_TIME=${2:-50}
NS3_ROOT=${3:-"$HOME/ns-3-dev"}

GYM_PORT=5555
STEP_INTERVAL=0.1
N_URLLC=5
N_EMBB=10
N_MMTC=20
OUTPUT_DIR="./results"

SCRATCH_NAME="network_slicing"   # matches scratch sub-folder name

# -----------------------------------------------------------------------
# Colour helpers
# -----------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log_info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_section() { echo -e "\n${BOLD}${CYAN}=== $* ===${NC}\n"; }

# -----------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------
log_section "5G Network Slicing Simulation"
log_info "Strategy     : $STRATEGY"
log_info "Sim time     : ${SIM_TIME}s"
log_info "ns-3 root    : $NS3_ROOT"
log_info "Gym port     : $GYM_PORT"
log_info "UEs          : URLLC=$N_URLLC  eMBB=$N_EMBB  mMTC=$N_MMTC"

if [ ! -d "$NS3_ROOT" ]; then
    log_error "ns-3 root not found: $NS3_ROOT"
    log_error "Set NS3_ROOT or pass as 3rd argument."
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

mkdir -p "$OUTPUT_DIR"

# -----------------------------------------------------------------------
# Step 1: Copy simulation files to ns-3 scratch (if not already there)
# -----------------------------------------------------------------------
log_section "Copying files to ns-3 scratch"

SCRATCH_TARGET="$NS3_ROOT/scratch/$SCRATCH_NAME"
mkdir -p "$SCRATCH_TARGET"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp -v "$SCRIPT_DIR/slice-simulation.cc"          "$SCRATCH_TARGET/"
cp -v "$SCRIPT_DIR/slice-gym-env.h"              "$SCRATCH_TARGET/"
cp -v "$SCRIPT_DIR/slice-gym-env.cc"             "$SCRATCH_TARGET/"
cp -v "$SCRIPT_DIR/nr-mac-scheduler-weighted-pf.h"  "$SCRATCH_TARGET/"
cp -v "$SCRIPT_DIR/nr-mac-scheduler-weighted-pf.cc" "$SCRATCH_TARGET/"
cp -v "$SCRIPT_DIR/CMakeLists.txt"               "$SCRATCH_TARGET/"

log_info "Files copied to $SCRATCH_TARGET"

# -----------------------------------------------------------------------
# Step 2: Build ns-3 simulation
# -----------------------------------------------------------------------
log_section "Building ns-3 simulation"

pushd "$NS3_ROOT" > /dev/null

# Configure if needed (only if build dir missing)
if [ ! -f "cmake_cache/build.ninja" ] && [ ! -f "build/build.ninja" ]; then
    log_info "Configuring ns-3 build..."
    python3 ns3 configure --enable-examples --enable-logs 2>&1 | tail -20
fi

log_info "Building scratch/network_slicing ..."
python3 ns3 build "$SCRATCH_NAME" 2>&1 | tail -30

log_info "Build successful."
popd > /dev/null

# -----------------------------------------------------------------------
# Step 3: Launch ns-3 in background, then Python controller
# -----------------------------------------------------------------------
log_section "Starting simulation + controller"

NS3_CMD="python3 $NS3_ROOT/ns3 run \
    \"scratch/$SCRATCH_NAME/network_slicing \
    --gymPort=$GYM_PORT \
    --simTime=$SIM_TIME \
    --stepInterval=$STEP_INTERVAL \
    --nUrllc=$N_URLLC \
    --nEmbb=$N_EMBB \
    --nMmtc=$N_MMTC \
    --verbose=true\" \
    --no-build"

PYTHON_CMD="python3 $SCRIPT_DIR/heuristic_controller.py \
    --strategy $STRATEGY \
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
for i in $(seq 1 30); do
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

# Start Python controller (foreground so we see output)
log_info "Starting Python heuristic controller..."
eval "$PYTHON_CMD"
PYTHON_EXIT=$?

# Wait for ns-3 to finish
wait $NS3_PID 2>/dev/null || true
NS3_EXIT=$?

# -----------------------------------------------------------------------
# Step 4: Summary
# -----------------------------------------------------------------------
log_section "Results"

if [ $PYTHON_EXIT -eq 0 ]; then
    log_info "Controller finished successfully."
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
log_info "To compare all four strategies, run:"
echo "  for s in threshold_reactive priority_based proportional_deficit weighted_fairness; do"
echo "    ./run_slicing.sh \$s $SIM_TIME $NS3_ROOT"
echo "  done"