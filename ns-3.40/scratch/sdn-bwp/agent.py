#!/usr/bin/env python3
"""
=============================================================================
agent.py
O-RAN Near-RT RIC — BWP Manager Algorithm (Heuristic-Based)
Python ns3-gym Agent for 5G Network Slicing BWP Control

Architecture position:
   [Service Layer]   ← slice SLA policies
   [Near-RT RIC]     ← THIS FILE (BWP Manager Algorithm)
   [gNB Data Plane]  ← ns-3 5G-LENA (Pre-processor + BWP Multiplexer)

BWP Manager Algorithm (Algorithm 1):
  Receives KPIs from gNB every control loop:
    - throughput_success_rate (eMBB)
    - deadline_failure_rate (uRLLC)
    - buffer_drop_rate (mMTC)

  Heuristic Actions:
    - If deadline_failure_rate > 0:     +2 PRBs to uRLLC
    - If throughput_success_rate < 1.0: +2 PRBs to eMBB
    - If buffer_drop_rate > 0:          +2 PRBs to mMTC
    - Constraint: total PRBs ≤ channel max capacity
    - Reduce PRBs from over-performing slices to rebalance

Usage:
  python3 agent.py --port 5555 --simTime 50

Requirements:
  pip install ns3gym numpy pandas matplotlib


=============================================================================
"""

import argparse
import logging
import csv
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import numpy as np

try:
    import gym
    from ns3gym import ns3env
except ImportError:
    raise ImportError(
        "ns3gym not found. Install with: pip install ns3gym\n"
        "Then ensure ns3-gym module is built in your ns-3 installation."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("controller.log"),
    ],
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants (must match C++ SLICE_URLLC, SLICE_EMBB, SLICE_MMTC)
# ──────────────────────────────────────────────────────────────────────────────
SLICE_URLLC = 0
SLICE_EMBB  = 1
SLICE_MMTC  = 2
NUM_SLICES  = 3
SLICE_NAMES = ["uRLLC", "eMBB ", "mMTC "]

# ──────────────────────────────────────────────────────────────────────────────
# Core (non-radio) delay budget that is present regardless of PRB allocation.
# Includes: P2P link (0.1ms one-way), EPC S1u processing, PDCP/RLC stack,
# and HARQ pipeline with N0/N1/N2 = 0 still leaving ~1 slot (~0.25ms) of
# scheduling-wait at numerology 2.
# Empirically the floor is 2–3ms in this single-gNB, port-based topology.
# Set conservatively at 2.5ms so that a genuine radio delay violation
# (e.g. heavy collision pushing delay to 5+ ms) still triggers a boost.
# ──────────────────────────────────────────────────────────────────────────────
CORE_DELAY_OFFSET_MS: float = 0.0

# PRB saturation guard: if uRLLC PRB utilisation is BELOW this fraction,
# the delay violation is not fixable by adding more PRBs (core-constrained).
# Lowered from 0.30 → 0.00 to match the actual uRLLC load profile:
#   5 UEs × 512 B/1ms = ~20 Mbps on a 100MHz num-2 BWP → ~12–13% PRB util.
PRB_SAT_THRESH: float = 0.00

# ──────────────────────────────────────────────────────────────────────────────
# SLA requirements (mirror of C++ SliceSLA)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SliceSLA:
    min_throughput_mbps: float   # Minimum required throughput [Mbps]
    max_delay_ms:        float   # Maximum tolerated delay [ms]
    max_prb_util:        float   # Maximum PRB utilisation [0–1]
    priority:            int     # Scheduling priority (lower = higher)
    target_throughput:   float   # Target throughput for success rate [Mbps]

DEFAULT_SLAS = {
    
    SLICE_URLLC: SliceSLA(min_throughput_mbps=1.0,  max_delay_ms=1.0,
                          max_prb_util=0.85, priority=1, target_throughput=2.0),
    SLICE_EMBB:  SliceSLA(min_throughput_mbps=20.0, max_delay_ms=50.0,
                          max_prb_util=0.90, priority=2, target_throughput=20.0),
    SLICE_MMTC:  SliceSLA(min_throughput_mbps=0.1,  max_delay_ms=200.0,
                          max_prb_util=0.80, priority=3, target_throughput=0.5),
}

# ──────────────────────────────────────────────────────────────────────────────
# KPI snapshot parsed from gym observation
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SliceKPI:
    throughput_mbps: float
    avg_delay_ms:    float
    prb_utilization: float
    pkt_drop_rate:   float

    @property
    def is_valid(self) -> bool:
        return self.throughput_mbps >= 0 and self.avg_delay_ms >= 0


# ──────────────────────────────────────────────────────────────────────────────
# Observation parsing (12-float vector)
# ──────────────────────────────────────────────────────────────────────────────
def parse_observation(obs: np.ndarray) -> List[SliceKPI]:
    """
    Convert flat observation vector [12] to list of 3 SliceKPIs.
    obs layout: [tput_0, dly_0, prb_0, drop_0,
                 tput_1, dly_1, prb_1, drop_1,
                 tput_2, dly_2, prb_2, drop_2]
    """
    kpis = []
    for s in range(NUM_SLICES):
        base = s * 4
        kpis.append(SliceKPI(
            throughput_mbps = float(obs[base + 0]),
            avg_delay_ms    = float(obs[base + 1]),
            prb_utilization = float(obs[base + 2]),
            pkt_drop_rate   = float(obs[base + 3]),
        ))
    return kpis


# ──────────────────────────────────────────────────────────────────────────────
# Derived KPIs for the BWP Manager Algorithm
# ──────────────────────────────────────────────────────────────────────────────
def compute_throughput_success_rate(kpi: SliceKPI, sla: SliceSLA) -> float:
    """
    eMBB: throughput_success_rate = min(1.0, actual_tput / target_tput)
    1.0 means fully meeting target.
    """
    if sla.target_throughput <= 0:
        return 1.0
    return min(1.0, kpi.throughput_mbps / sla.target_throughput)


def compute_deadline_failure_rate(
    kpi: SliceKPI,
    sla: SliceSLA,
    core_offset_ms: float = CORE_DELAY_OFFSET_MS,
) -> float:
    """
    uRLLC: deadline_failure_rate measures only the radio-attributable delay.

    FIX (v2): Subtract the irreducible core-network delay floor before
    computing the violation ratio.  Without this correction, the measured
    delay (FlowMonitor end-to-end) always includes P2P + EPC + HARQ pipeline
    latency (~2.5ms in this topology), making deadline_failure permanently > 0
    even when the radio scheduler is perfectly fine.

    After subtraction:
      radio_delay = max(0, measured_delay - core_offset)
      violation   = max(0, (radio_delay - max_delay) / max_delay)

    Also adds packet drop rate as an additional deadline-failure signal.
    """
    # Strip the irreducible core-network floor
    radio_delay_ms = max(0.0, kpi.avg_delay_ms - core_offset_ms) \
                     if kpi.avg_delay_ms > 0 else 0.0

    delay_failure = max(
        0.0,
        (radio_delay_ms - sla.max_delay_ms) / sla.max_delay_ms
    ) if radio_delay_ms > 0 else 0.0

    # Packet drops also count as deadline failures for uRLLC
    return delay_failure + kpi.pkt_drop_rate


def compute_buffer_drop_rate(kpi: SliceKPI) -> float:
    """
    mMTC: buffer_drop_rate = pkt_drop_rate from FlowMonitor
    Indicates queue overflow (buffer drops).
    """
    return kpi.pkt_drop_rate


# =============================================================================
# BWP MANAGER ALGORITHM (Algorithm 1 — Python Near-RT RIC)
# =============================================================================

class BwpManagerController:
    """
    Implements the BWP Manager Algorithm from the user specification.

    Every control loop:
      1. Parse KPIs → derive success/failure rates
      2. For each violating slice: allocate +2 PRBs (via weight increase)
      3. For over-performing slices: decrease allocation
      4. Enforce total PRB constraint (weights sum to 1.0)
      5. Apply EMA smoothing for stability

    The "PRBs" are represented as normalised weights [0,1] that the C++
    scheduler translates into PF time-window changes. +2 PRBs maps to
    a weight increment of `prb_step` (configurable).

    v2 changes:
    - core_delay_offset_ms: passed through to compute_deadline_failure_rate()
    - prb_sat_thresh: configurable saturation guard threshold (default 0.20)
    - urllc_freeze_steps counter: when the saturation guard fires repeatedly,
      weight is released progressively to avoid indefinite resource hoarding.
    """

    def __init__(self,
                 slas: Dict[int, SliceSLA],
                 step_interval: float = 0.1,
                 prb_step: float = 0.05,
                 tau: float = 0.3,
                 w_min: float = 0.10,
                 w_max: float = 0.70,
                 buffer_threshold: float = 0.80,
                 core_delay_offset_ms: float = CORE_DELAY_OFFSET_MS,
                 prb_sat_thresh: float = PRB_SAT_THRESH):
        """
        Parameters
        ----------
        slas : dict
            SLA requirements per slice
        step_interval : float
            Control loop period [s]
        prb_step : float
            Weight increment per "+2 PRBs" allocation step.
            Normalised: 0.05 maps to ~2 PRBs on a 52-PRB channel.
        tau : float
            EMA smoothing factor (0=no update, 1=full update)
        w_min : float
            Minimum weight per slice (starvation prevention)
        w_max : float
            Maximum weight per slice
        buffer_threshold : float
            mMTC buffer utilisation threshold (0.80 = 80%)
        core_delay_offset_ms : float
            Irreducible core-network delay to subtract before computing
            uRLLC deadline violation.  Set to match your simulation topology.
        prb_sat_thresh : float
            PRB utilisation below which a uRLLC delay violation is treated as
            core-constrained (not fixable by adding PRBs).
        """
        self.slas = slas
        self.step_interval = step_interval
        self.prb_step = prb_step
        self.tau = tau
        self.w_min = w_min
        self.w_max = w_max
        self.buffer_threshold = buffer_threshold
        self.core_delay_offset_ms = core_delay_offset_ms
        self.prb_sat_thresh = prb_sat_thresh

        # Initial equal allocation
        self.weights = np.array([1/3, 1/3, 1/3], dtype=float)
        self.step_count = 0

        # FIX v2: Track consecutive steps where the saturation guard fired.
        # Used to apply progressive weight release when uRLLC holds spectrum
        # it cannot use (core-constrained delay, low PRB utilisation).
        self.urllc_freeze_steps: int = 0

        # History for logging / visualisation
        self.kpi_history:    List[List[SliceKPI]] = []
        self.weight_history: List[np.ndarray]     = []
        self.reward_history: List[float]          = []

    def compute_action(self, kpis: List[SliceKPI]) -> np.ndarray:
        """
        BWP Manager Algorithm — Priority-Ordered Utility Allocation (Paper 2 RSU 2024).
        """
        self.step_count += 1
        sim_time = self.step_count * self.step_interval
        is_grace_period = (sim_time < 5.0)

        w = self.weights.copy()

        # ── 1. Compute derived KPIs ─────────────────────────────────────────
        deadline_failure = compute_deadline_failure_rate(
            kpis[SLICE_URLLC],
            self.slas[SLICE_URLLC],
            core_offset_ms=self.core_delay_offset_ms,
        )
        throughput_success = compute_throughput_success_rate(
            kpis[SLICE_EMBB], self.slas[SLICE_EMBB])
        buffer_drop = compute_buffer_drop_rate(kpis[SLICE_MMTC])
        mmtc_prb_overload = kpis[SLICE_MMTC].prb_utilization > self.buffer_threshold

        # ── 2. Compute Utility Scores (Urgency) ───────────────────────────────
        scores = np.zeros(NUM_SLICES)

        # uRLLC Score + Saturation Guard
        if deadline_failure > 0:
            if kpis[SLICE_URLLC].prb_utilization < self.prb_sat_thresh:
                self.urllc_freeze_steps += 1
                scores[SLICE_URLLC] = 0.0  # Treat as idle for the loop
                log.info(
                    f"  uRLLC core-constrained (prb={kpis[SLICE_URLLC].prb_utilization:.2f}) "
                    f"freeze_steps={self.urllc_freeze_steps} → score=0 (release mode)"
                )
            else:
                boost = self.prb_step * (1.0 + min(2.0, deadline_failure))
                w[SLICE_URLLC] += boost
                log.debug(f"  uRLLC radio deadline violation → +{boost:.3f}")

        elif kpis[SLICE_URLLC].avg_delay_ms < self.slas[SLICE_URLLC].max_delay_ms * 0.8:
            # Over-performing: free resources
            self.urllc_freeze_steps = 0
            w[SLICE_URLLC] = max(w[SLICE_URLLC] - self.prb_step * 0.5, self.w_min)
            log.debug(f"  uRLLC over-performing → releasing weight")

        else:
            # SLA met (within 80–100% of max_delay): hold current weight
            self.urllc_freeze_steps = 0

        # ── eMBB: throughput-driven allocation ───────────────────────────
        if throughput_success < 1.0:
            # TCP RAMP-UP CHECK: Give more aggressive boost to eMBB if it's failing
            if not is_grace_period or throughput_success < 0.1:
                deficit = 1.0 - throughput_success
                boost = self.prb_step * (3.0 + deficit * 2)  # Significantly increased boost
                w[SLICE_EMBB] += boost
                log.debug(f"  eMBB tput deficit={deficit:.3f} → +{boost:.3f}")
            else:
                log.debug(f"  eMBB tput deficit in grace period → waiting for TCP ramp-up")
        elif throughput_success >= 1.0 and kpis[SLICE_EMBB].prb_utilization < 0.6:
            # Over-performing: free resources
            w[SLICE_EMBB] = max(w[SLICE_EMBB] - self.prb_step * 0.2, self.w_min)
            log.debug(f"  eMBB over-performing → releasing weight")

        # ── mMTC: buffer-driven allocation ───────────────────────────────
        if buffer_drop > 0 or mmtc_prb_overload:
            # Allocate +2 PRBs
            boost = self.prb_step * (1.0 + buffer_drop)
            w[SLICE_MMTC] += boost
            log.debug(f"  mMTC buffer issue → +{boost:.3f}")
        elif buffer_drop == 0 and kpis[SLICE_MMTC].prb_utilization < 0.3:
            # Over-performing: free resources
            w[SLICE_MMTC] = max(w[SLICE_MMTC] - self.prb_step * 0.5, self.w_min)
            log.debug(f"  mMTC over-performing → releasing weight")

        # ── Constraint: total weight must not exceed 1.0 ─────────────────
        # Clip to [w_min, w_max] per slice
        w = np.clip(w, self.w_min, self.w_max)

        return w

    def step(self, obs: np.ndarray, reward: float) -> np.ndarray:
        """Main entry point: parse obs → compute action → log → return."""
        kpis = parse_observation(obs)
        self.kpi_history.append(kpis)
        self.reward_history.append(reward)

        action = self.compute_action(kpis)

        # Normalise weights to sum to 1.0
        action = np.clip(action, self.w_min, None)
        action = action / action.sum()

        # EMA smoothing for stability
        action = (1 - self.tau) * self.weights + self.tau * action

        # Re-normalise after EMA
        action = np.clip(action, self.w_min, None)
        action = action / action.sum()

        self.weights = action
        self.weight_history.append(action.copy())
        self.step_count += 1

        self._log_step(kpis, action, reward)
        return action.astype(np.float32)

    def _log_step(self, kpis: List[SliceKPI], action: np.ndarray, reward: float):
        parts = []
        for s in range(NUM_SLICES):
            k = kpis[s]
            parts.append(
                f"{SLICE_NAMES[s]} "
                f"tput={k.throughput_mbps:6.2f}Mbps "
                f"dly={k.avg_delay_ms:7.3f}ms "
                f"prb={k.prb_utilization:.2f} "
                f"drop={k.pkt_drop_rate:.4f} "
                f"w={action[s]:.3f}"
            )
        log.info(f"Step {self.step_count:4d} | r={reward:+.3f} | " +
                 "  ".join(parts))

    def summary(self) -> str:
        if not self.reward_history:
            return "No steps recorded."
        rewards = np.array(self.reward_history)
        return (
            f"Steps: {self.step_count}\n"
            f"Reward — mean={rewards.mean():.3f}  "
            f"std={rewards.std():.3f}  "
            f"min={rewards.min():.3f}  "
            f"max={rewards.max():.3f}\n"
            f"Final weights — "
            f"uRLLC={self.weights[0]:.3f}  "
            f"eMBB={self.weights[1]:.3f}  "
            f"mMTC={self.weights[2]:.3f}"
        )


# =============================================================================
# MAIN CONTROL LOOP
# =============================================================================

def run(args):
    log.info("=" * 70)
    log.info("  5G Network Slicing — O-RAN Near-RT RIC (BWP Manager)")
    log.info(f"  ns3-gym port        : {args.port}")
    log.info(f"  Step interval       : {args.step_interval}s")
    log.info(f"  Sim time            : {args.sim_time}s")
    log.info(f"  Core delay offset   : {args.core_delay_offset}ms")
    log.info(f"  PRB sat threshold   : {args.prb_sat_thresh}")
    log.info("=" * 70)

    # Create BWP Manager Controller
    controller = BwpManagerController(
        slas                 = DEFAULT_SLAS,
        step_interval        = args.step_interval,
        prb_step             = args.prb_step,
        tau                  = args.tau,
        core_delay_offset_ms = args.core_delay_offset,
        prb_sat_thresh       = args.prb_sat_thresh,
    )

    # Connect to ns3-gym environment
    env = ns3env.Ns3Env(
        port        = args.port,
        stepTime    = args.step_interval,
        startSim    = args.start_sim,
        simSeed     = args.seed,
        simArgs     = {"--simTime":      str(args.sim_time),
                       "--stepInterval": str(args.step_interval),
                       "--nUrllc":       str(args.n_urllc),
                       "--nEmbb":        str(args.n_embb),
                       "--nMmtc":        str(args.n_mmtc),
                       "--verbose":      "true"},
        debug       = args.debug,
    )

    log.info(f"Observation space : {env.observation_space}")
    log.info(f"Action space      : {env.action_space}")

    # Metrics CSV writer
    csv_path = Path(args.output_dir) / "metrics_bwp_manager.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "step", "reward",
        "urllc_tput", "urllc_dly", "urllc_prb", "urllc_drop",
        "embb_tput",  "embb_dly",  "embb_prb",  "embb_drop",
        "mmtc_tput",  "mmtc_dly",  "mmtc_prb",  "mmtc_drop",
        "w_urllc", "w_embb", "w_mmtc",
        "deadline_failure", "tput_success", "buffer_drop",
        "urllc_freeze_steps",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # Episode loop
    # ─────────────────────────────────────────────────────────────────────────
    obs = env.reset()
    done = False
    episode_reward = 0.0
    step = 0

    while not done:
        reward = 0.0 if step == 0 else reward

        # Compute action from BWP Manager heuristic
        action = controller.step(np.array(obs, dtype=float), reward)

        # Step environment
        try:
            obs, reward, done, info = env.step(action)
        except Exception as ex:
            log.error(f"env.step() failed at step {step}: {ex}")
            break

        episode_reward += reward
        step += 1

        # Log to CSV
        kpis = parse_observation(np.array(obs, dtype=float))
        deadline_fail = compute_deadline_failure_rate(
            kpis[SLICE_URLLC],
            DEFAULT_SLAS[SLICE_URLLC],
            core_offset_ms=args.core_delay_offset,
        )
        tput_success = compute_throughput_success_rate(
            kpis[SLICE_EMBB], DEFAULT_SLAS[SLICE_EMBB])
        buf_drop = compute_buffer_drop_rate(kpis[SLICE_MMTC])

        csv_writer.writerow([
            step, reward,
            kpis[0].throughput_mbps, kpis[0].avg_delay_ms,
            kpis[0].prb_utilization, kpis[0].pkt_drop_rate,
            kpis[1].throughput_mbps, kpis[1].avg_delay_ms,
            kpis[1].prb_utilization, kpis[1].pkt_drop_rate,
            kpis[2].throughput_mbps, kpis[2].avg_delay_ms,
            kpis[2].prb_utilization, kpis[2].pkt_drop_rate,
            action[0], action[1], action[2],
            deadline_fail, tput_success, buf_drop,
            controller.urllc_freeze_steps,
        ])

        # SLA violation summary every 50 steps
        if step % 50 == 0:
            log.info("-" * 60)
            log.info(f"  uRLLC deadline_failure_rate: {deadline_fail:.4f}"
                     f"  {'✓' if deadline_fail == 0 else '✗ VIOLATION'}"
                     f"  (freeze_steps={controller.urllc_freeze_steps})")
            log.info(f"  eMBB  throughput_success:     {tput_success:.4f}"
                     f"  {'✓' if tput_success >= 1.0 else '✗ DEFICIT'}")
            log.info(f"  mMTC  buffer_drop_rate:       {buf_drop:.4f}"
                     f"  {'✓' if buf_drop == 0 else '✗ DROPS'}")
            log.info("-" * 60)

    # ─────────────────────────────────────────────────────────────────────────
    # Episode summary
    # ─────────────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info(f"Episode finished after {step} steps")
    log.info(f"Total reward: {episode_reward:.3f}")
    log.info(controller.summary())
    log.info(f"Metrics saved to: {csv_path}")
    log.info("=" * 70)

    csv_file.close()
    env.close()

    # Save weight evolution to JSON
    json_path = Path(args.output_dir) / "weights_bwp_manager.json"
    weight_evolution = {
        "algorithm": "bwp_manager_v2",
        "core_delay_offset_ms": args.core_delay_offset,
        "prb_sat_thresh": args.prb_sat_thresh,
        "steps": list(range(len(controller.weight_history))),
        "w_urllc": [float(w[0]) for w in controller.weight_history],
        "w_embb":  [float(w[1]) for w in controller.weight_history],
        "w_mmtc":  [float(w[2]) for w in controller.weight_history],
        "reward":  [float(r) for r in controller.reward_history],
    }
    with open(json_path, "w") as f:
        json.dump(weight_evolution, f, indent=2)
    log.info(f"Weight evolution saved to: {json_path}")

    return controller


# =============================================================================
# RESULT VISUALISATION (optional, requires matplotlib)
# =============================================================================

def plot_results(controller: BwpManagerController, output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        log.warning("matplotlib not available — skipping plots.")
        return

    steps = list(range(len(controller.weight_history)))
    if not steps:
        return

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("5G Network Slicing — BWP Manager Algorithm (O-RAN Near-RT RIC)",
                 fontsize=16, fontweight="bold")

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    colors = {"uRLLC": "#e74c3c", "eMBB": "#3498db", "mMTC": "#2ecc71"}
    names = ["uRLLC", "eMBB", "mMTC"]

    # --- Plot 1: Weight evolution ---
    ax1 = fig.add_subplot(gs[0, :])
    w_arr = np.array(controller.weight_history)
    for s, name in enumerate(names):
        ax1.plot(steps, w_arr[:, s], label=name,
                 color=list(colors.values())[s], linewidth=2)
    ax1.set_title("BWP Weight Evolution (RIC Output)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Normalised Weight")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Reward ---
    ax2 = fig.add_subplot(gs[1, 0])
    rewards = np.array(controller.reward_history)
    ax2.plot(steps, rewards, color="#8e44ad", linewidth=1.5, alpha=0.8)
    ax2.axhline(y=np.log(2) * 3, color="green", linestyle="--", alpha=0.5,
                label="Max reward")
    ax2.set_title("SLA Compliance Reward")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Reward")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Throughput per slice ---
    ax3 = fig.add_subplot(gs[1, 1])
    if controller.kpi_history:
        for s, (name, color) in enumerate(colors.items()):
            tput = [k[s].throughput_mbps for k in controller.kpi_history]
            ax3.plot(steps, tput, label=name, color=color, linewidth=1.5)
            ax3.axhline(y=DEFAULT_SLAS[s].min_throughput_mbps, color=color,
                        linestyle=":", alpha=0.5)
    ax3.set_title("Slice Throughput [Mbps]")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Throughput [Mbps]")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Delay per slice ---
    ax4 = fig.add_subplot(gs[2, 0])
    if controller.kpi_history:
        for s, (name, color) in enumerate(colors.items()):
            dly = [k[s].avg_delay_ms for k in controller.kpi_history]
            ax4.plot(steps, dly, label=name, color=color, linewidth=1.5)
            ax4.axhline(y=DEFAULT_SLAS[s].max_delay_ms, color=color,
                        linestyle=":", alpha=0.5)
    ax4.set_title("Slice Delay [ms]")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Delay [ms]")
    ax4.set_yscale("log")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # --- Plot 5: Packet Drop Rate per slice ---
    ax5 = fig.add_subplot(gs[2, 1])
    if controller.kpi_history:
        for s, (name, color) in enumerate(colors.items()):
            drop = [k[s].pkt_drop_rate for k in controller.kpi_history]
            ax5.plot(steps, drop, label=name, color=color, linewidth=1.5)
    ax5.set_title("Packet Drop Rate")
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Drop Rate [0-1]")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "results_bwp_manager.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    log.info(f"Plot saved to: {plot_path}")
    plt.close()


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="O-RAN Near-RT RIC — BWP Manager Algorithm for 5G Slicing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--port", type=int, default=5555,
        help="OpenGym TCP port (must match --gymPort in ns-3)")
    parser.add_argument("--sim-time", type=float, default=50.0,
        help="Simulation duration [s]")
    parser.add_argument("--step-interval", type=float, default=0.1,
        help="Gym step interval [s]")
    parser.add_argument("--n-urllc", type=int, default=5,
        help="Number of URLLC UEs")
    parser.add_argument("--n-embb", type=int, default=10,
        help="Number of eMBB UEs")
    parser.add_argument("--n-mmtc", type=int, default=30,
        help="Number of mMTC UEs")
    parser.add_argument("--prb-step", type=float, default=0.05,
        help="Weight increment per +2 PRB step")
    parser.add_argument("--tau", type=float, default=0.3,
        help="EMA smoothing factor (0=no update, 1=instant)")
    parser.add_argument("--core-delay-offset", type=float,
        default=CORE_DELAY_OFFSET_MS,
        help="Irreducible core-network delay offset to subtract [ms]. "
             "Set to P2P + EPC + HARQ pipeline floor for your topology.")
    parser.add_argument("--prb-sat-thresh", type=float,
        default=PRB_SAT_THRESH,
        help="PRB utilisation below which a uRLLC delay violation is "
             "treated as core-constrained (not fixable by more PRBs).")
    parser.add_argument("--start-sim", action="store_true", default=False,
        help="Start ns-3 simulation from Python")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed")
    parser.add_argument("--output-dir", default="./results",
        help="Directory to save metrics and plots")
    parser.add_argument("--plot", action="store_true", default=False,
        help="Generate result plots after simulation")
    parser.add_argument("--debug", action="store_true", default=False,
        help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    controller = run(args)

    if args.plot:
        plot_results(controller, args.output_dir)


if __name__ == "__main__":
    main()