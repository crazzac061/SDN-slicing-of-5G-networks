#!/usr/bin/env python3
"""
=============================================================================
heuristic_controller.py
SDN / NFV Controller — Heuristic-Based 5G Network Slice Manager
Control Layer of the 5G Slicing Architecture

Architecture position:
   [Service Layer]  ← slice policies defined here
   [Control Layer]  ← THIS FILE  (Python VNF / SDN Controller)
   [Infrastructure] ← ns-3 5G-LENA (via ns3-gym)

Heuristic strategies implemented:
  1. THRESHOLD_REACTIVE   — simple threshold checks, proportional adjustment
  2. PRIORITY_BASED       — static priority order: URLLC > eMBB > mMTC
  3. PROPORTIONAL_DEFICIT — redistribute proportional to SLA violation severity
  4. WEIGHTED_FAIRNESS    — balance SLA compliance across slices equally

Usage:
  python heuristic_controller.py --strategy proportional_deficit \
      --port 5555 --simTime 50

Requirements:
  pip install ns3gym numpy pandas matplotlib
=============================================================================
"""

import argparse
import logging
import time
import json
import csv
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum

import numpy as np

try:
    import gym
    from ns3gym import ns3env
except ImportError:
    raise ImportError(
        "ns3gym not found. Install with: pip install ns3gym\n"
        "Then ensure ns3-gym module is built in your ns-3 installation."
    )

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("controller.log"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Slice constants (must match C++ constants)
# ---------------------------------------------------------------------------
SLICE_URLLC = 0
SLICE_EMBB  = 1
SLICE_MMTC  = 2
NUM_SLICES  = 3
SLICE_NAMES = ["URLLC", "eMBB ", "mMTC "]

# ---------------------------------------------------------------------------
# SLA requirements (mirror of C++ SliceSLA)
# ---------------------------------------------------------------------------
@dataclass
class SliceSLA:
    min_throughput_mbps: float  # Minimum required throughput [Mbps]
    max_delay_ms:        float  # Maximum tolerated delay [ms]
    max_prb_util:        float  # Maximum PRB utilisation [0–1]
    priority:            int    # Scheduling priority (lower = higher priority)

DEFAULT_SLAS = {
    SLICE_URLLC: SliceSLA(min_throughput_mbps=1.0,  max_delay_ms=1.0,   max_prb_util=0.85, priority=1),
    SLICE_EMBB:  SliceSLA(min_throughput_mbps=20.0, max_delay_ms=50.0,  max_prb_util=0.90, priority=2),
    SLICE_MMTC:  SliceSLA(min_throughput_mbps=0.1,  max_delay_ms=200.0, max_prb_util=0.80, priority=3),
}

# ---------------------------------------------------------------------------
# KPI snapshot parsed from gym observation
# ---------------------------------------------------------------------------
@dataclass
class SliceKPI:
    throughput_mbps: float
    avg_delay_ms:    float
    prb_utilization: float

    @property
    def is_valid(self) -> bool:
        return self.throughput_mbps >= 0 and self.avg_delay_ms >= 0

# ---------------------------------------------------------------------------
# Observation parsing
# ---------------------------------------------------------------------------
def parse_observation(obs: np.ndarray) -> List[SliceKPI]:
    """
    Convert flat observation vector [9] to list of 3 SliceKPIs.
    obs layout: [tput_0, dly_0, prb_0,  tput_1, dly_1, prb_1,  tput_2, dly_2, prb_2]
    """
    kpis = []
    for s in range(NUM_SLICES):
        base = s * 3
        kpis.append(SliceKPI(
            throughput_mbps = float(obs[base + 0]),
            avg_delay_ms    = float(obs[base + 1]),
            prb_utilization = float(obs[base + 2]),
        ))
    return kpis

# ---------------------------------------------------------------------------
# Violation scoring helpers
# ---------------------------------------------------------------------------
def throughput_violation(kpi: SliceKPI, sla: SliceSLA) -> float:
    """Returns ≥0 fraction of SLA deficit (0 = no violation)."""
    if sla.min_throughput_mbps <= 0:
        return 0.0
    deficit = sla.min_throughput_mbps - kpi.throughput_mbps
    return max(0.0, deficit / sla.min_throughput_mbps)

def delay_violation(kpi: SliceKPI, sla: SliceSLA) -> float:
    """Returns ≥0 normalised excess delay (0 = no violation)."""
    if kpi.avg_delay_ms <= 0:
        return 0.0
    excess = kpi.avg_delay_ms - sla.max_delay_ms
    return max(0.0, excess / sla.max_delay_ms)

def prb_violation(kpi: SliceKPI, sla: SliceSLA) -> float:
    """Returns ≥0 normalised PRB excess (0 = no violation)."""
    excess = kpi.prb_utilization - sla.max_prb_util
    return max(0.0, excess / (1.0 - sla.max_prb_util + 1e-6))

def overall_violation(kpi: SliceKPI, sla: SliceSLA) -> float:
    """Composite violation score ∈ [0, ∞). Higher means worse."""
    return throughput_violation(kpi, sla) + delay_violation(kpi, sla)


# =============================================================================
# HEURISTIC CONTROLLER BASE CLASS
# =============================================================================

class HeuristicController:
    """Base class for heuristic slice controllers."""

    def __init__(self,
                 slas: Dict[int, SliceSLA],
                 step_interval: float = 0.1,
                 epsilon: float = 0.05):
        self.slas = slas
        self.step_interval = step_interval
        self.epsilon = epsilon
        self.weights = np.array([1/3, 1/3, 1/3], dtype=float)
        self.step_count = 0

        # History for logging / visualisation
        self.kpi_history:    List[List[SliceKPI]] = []
        self.weight_history: List[np.ndarray]     = []
        self.reward_history: List[float]          = []

    def compute_action(self, kpis: List[SliceKPI]) -> np.ndarray:
        """
        Override in subclasses to produce weight vector [w0, w1, w2] ∈ [0,1]³.
        The vector need not sum to 1; the simulator will normalise it.
        """
        raise NotImplementedError

    def step(self, obs: np.ndarray, reward: float) -> np.ndarray:
        """Main entry point: parse obs → compute action → log → return action."""
        kpis = parse_observation(obs)
        self.kpi_history.append(kpis)
        self.reward_history.append(reward)

        action = self.compute_action(kpis)

        # IMPROVEMENT: epsilon-greedy noise (Recommendation 2)
        if self.epsilon > 0:
            noise = np.random.uniform(-self.epsilon, self.epsilon, size=NUM_SLICES)
            action = action + noise

        # Clip and normalise
        action = np.clip(action, 0.1, None) # IMPROVEMENT: 0.1 floor
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
            sla = self.slas[s]
            viol = overall_violation(k, sla)
            parts.append(
                f"{SLICE_NAMES[s]} "
                f"tput={k.throughput_mbps:6.2f}Mbps "
                f"dly={k.avg_delay_ms:7.3f}ms "
                f"prb={k.prb_utilization:.2f} "
                f"viol={viol:.3f} "
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
            f"URLLC={self.weights[0]:.3f}  "
            f"eMBB={self.weights[1]:.3f}  "
            f"mMTC={self.weights[2]:.3f}"
        )


# =============================================================================
# Strategy 1: THRESHOLD REACTIVE
# =============================================================================

class ThresholdReactiveController(HeuristicController):
    """
    Simple rule engine:
      - For each slice, check if any KPI violates its SLA.
      - If violated, increase that slice's weight by delta_up.
      - If compliant, decrease weight slightly by delta_down (to free resources).
      - Clip weights to [w_min, w_max].

    Parameters:
      delta_up   — weight increment on violation   (default 0.10)
      delta_down — weight decrement on compliance  (default 0.02)
      w_min      — minimum weight per slice        (default 0.05)
      w_max      — maximum weight per slice        (default 0.70)
    """

    def __init__(self,
                 slas: Dict[int, SliceSLA],
                 step_interval: float = 0.1,
                 delta_up:   float = 0.10,
                 delta_down: float = 0.02,
                 w_min: float = 0.10, # IMPROVEMENT: 0.1 floor
                 w_max: float = 0.70,
                 epsilon: float = 0.05):
        super().__init__(slas, step_interval, epsilon)
        self.delta_up   = delta_up
        self.delta_down = delta_down
        self.w_min      = w_min
        self.w_max      = w_max
        self.weights    = np.array([1/3, 1/3, 1/3])

    def compute_action(self, kpis: List[SliceKPI]) -> np.ndarray:
        w = self.weights.copy()

        for s in range(NUM_SLICES):
            viol = overall_violation(kpis[s], self.slas[s])
            if viol > 0.0:
                # Proportional increase: bigger violation → bigger jump
                w[s] += self.delta_up * (1.0 + viol)
            else:
                # Slight release of resources when SLA is met
                w[s] = max(w[s] - self.delta_down, self.w_min)

        # Enforce per-slice max
        w = np.clip(w, self.w_min, self.w_max)
        return w


# =============================================================================
# Strategy 2: PRIORITY BASED
# =============================================================================

class PriorityBasedController(HeuristicController):
    """
    Fixed priority allocation:
      URLLC always gets its SLA first.
      Remaining resources split between eMBB and mMTC.

    Allocation logic:
      1. Compute minimum resource fraction needed per slice to meet SLA.
      2. Assign in priority order.
      3. Remaining budget goes to eMBB (throughput-hungry).
    """

    def compute_action(self, kpis: List[SliceKPI]) -> np.ndarray:
        # Need-fractions: violation severity → how much extra weight needed
        needs = np.zeros(NUM_SLICES)
        base  = np.array([0.20, 0.50, 0.30])  # baseline allocation

        for s in range(NUM_SLICES):
            viol = overall_violation(kpis[s], self.slas[s])
            needs[s] = viol

        # URLLC is highest priority: give it whatever it needs first
        w = base.copy()

        # Priority order: URLLC=1, eMBB=2, mMTC=3
        priority_order = sorted(range(NUM_SLICES),
                                key=lambda s: self.slas[s].priority)

        for s in priority_order:
            if needs[s] > 0:
                boost = min(needs[s] * 0.3, 0.20)
                w[s] += boost
                # Take from lowest priority slices
                for other in reversed(priority_order):
                    if other == s:
                        continue
                    take = min(boost / (NUM_SLICES - 1), w[other] - 0.05)
                    if take > 0:
                        w[other] -= take
                        w[s]     += take
                        boost    -= take
                        if boost <= 0:
                            break

        return np.clip(w, 0.10, 0.80) # IMPROVEMENT: 0.1 floor


# =============================================================================
# Strategy 3: PROPORTIONAL DEFICIT (recommended for general use)
# =============================================================================

class ProportionalDeficitController(HeuristicController):
    """
    Redistributes weights proportional to each slice's SLA violation deficit.

    Algorithm:
      1. Compute weighted violation score per slice:
           score_s = α * throughput_violation + β * delay_violation
      2. New weight proposal:
           w_s = baseline_s + γ * score_s
      3. Normalise.
      4. Apply exponential moving average for stability:
           w_s ← (1-τ) * w_old + τ * w_new

    Parameters:
      alpha  — weight of throughput violation in score  (default 0.5)
      beta   — weight of delay violation in score       (default 0.5)
      gamma  — step gain for reallocation               (default 0.3)
      tau    — EMA smoothing factor (0=no update, 1=full update) (default 0.4)
    """

    def __init__(self,
                 slas: Dict[int, SliceSLA],
                 step_interval: float = 0.1,
                 alpha: float = 0.4,
                 beta:  float = 0.6,
                 gamma: float = 0.30,
                 tau:   float = 0.40):
        super().__init__(slas, step_interval)
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.tau   = tau

        # Baseline allocation based on SLA priorities
        self.baseline = np.array([0.30, 0.50, 0.20])  # URLLC, eMBB, mMTC

    def compute_action(self, kpis: List[SliceKPI]) -> np.ndarray:
        scores = np.zeros(NUM_SLICES)

        for s in range(NUM_SLICES):
            tv = throughput_violation(kpis[s], self.slas[s])
            dv = delay_violation(kpis[s],      self.slas[s])
            scores[s] = self.alpha * tv + self.beta * dv

        # Proposed weight = baseline + γ * violation_score
        w_new = self.baseline + self.gamma * scores

        # Normalise
        w_new = np.clip(w_new, 0.10, None) # IMPROVEMENT: 0.1 floor
        w_new /= w_new.sum()

        # Exponential moving average for smooth updates
        w_smooth = (1 - self.tau) * self.weights + self.tau * w_new

        return w_smooth


# =============================================================================
# Strategy 4: WEIGHTED FAIRNESS (Max-Min fairness across SLA compliance)
# =============================================================================

class WeightedFairnessController(HeuristicController):
    """
    Max-Min SLA Fairness:
    Aims to equalise the SLA compliance ratio across slices.

    Compliance ratio per slice ∈ [0, 1]:
      compliance_s = 1 - overall_violation_s  (clamped to [0,1])

    The controller identifies the slice with the worst compliance and
    increases its weight, taking from the best-compliant slice.
    This is inspired by max-min fairness in network resource allocation.

    Parameters:
      step_size — weight transfer per step   (default 0.05)
      tol       — compliance difference to trigger rebalancing (default 0.1)
    """

    def __init__(self,
                 slas: Dict[int, SliceSLA],
                 step_interval: float = 0.1,
                 step_size: float = 0.05,
                 tol:       float = 0.10):
        super().__init__(slas, step_interval)
        self.step_size = step_size
        self.tol       = tol

    def compute_action(self, kpis: List[SliceKPI]) -> np.ndarray:
        compliance = np.zeros(NUM_SLICES)
        for s in range(NUM_SLICES):
            viol = overall_violation(kpis[s], self.slas[s])
            compliance[s] = max(0.0, 1.0 - viol)

        w = self.weights.copy()

        worst_s = int(np.argmin(compliance))
        best_s  = int(np.argmax(compliance))

        if (compliance[best_s] - compliance[worst_s]) > self.tol:
            transfer = min(self.step_size, w[best_s] - 0.05)
            if transfer > 0:
                w[best_s]  -= transfer
                w[worst_s] += transfer
                log.debug(
                    f"Fairness transfer: {SLICE_NAMES[best_s]} → "
                    f"{SLICE_NAMES[worst_s]}  Δw={transfer:.3f}  "
                    f"compliance gap={compliance[best_s]-compliance[worst_s]:.3f}"
                )

        return np.clip(w, 0.10, 0.80) # IMPROVEMENT: 0.1 floor


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

STRATEGIES = {
    "threshold_reactive":    ThresholdReactiveController,
    "priority_based":        PriorityBasedController,
    "proportional_deficit":  ProportionalDeficitController,
    "weighted_fairness":     WeightedFairnessController,
}

def create_controller(strategy: str,
                      slas: Dict[int, SliceSLA],
                      step_interval: float,
                      **kwargs) -> HeuristicController:
    cls = STRATEGIES.get(strategy)
    if cls is None:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         f"Choose from: {list(STRATEGIES.keys())}")
    return cls(slas=slas, step_interval=step_interval, **kwargs)


# =============================================================================
# MAIN CONTROL LOOP
# =============================================================================

def run(args):
    log.info("=" * 70)
    log.info("  5G Network Slicing — Heuristic SDN/NFV Controller")
    log.info(f"  Strategy      : {args.strategy}")
    log.info(f"  ns3-gym port  : {args.port}")
    log.info(f"  Step interval : {args.step_interval}s")
    log.info("=" * 70)

    # Create controller
    controller = create_controller(
        strategy      = args.strategy,
        slas          = DEFAULT_SLAS,
        step_interval = args.step_interval,
    )

    # Connect to ns3-gym environment
    env = ns3env.Ns3Env(
        port        = args.port,
        stepTime    = args.step_interval,
        startSim    = args.start_sim,   # True = start ns-3 from Python
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
    csv_path = Path(args.output_dir) / f"metrics_{args.strategy}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "step", "reward",
        "urllc_tput", "urllc_dly", "urllc_prb",
        "embb_tput",  "embb_dly",  "embb_prb",
        "mmtc_tput",  "mmtc_dly",  "mmtc_prb",
        "w_urllc", "w_embb", "w_mmtc",
    ])

    # -----------------------------------------------------------------------
    # Episode loop
    # -----------------------------------------------------------------------
    obs = env.reset()
    done = False
    episode_reward = 0.0
    step = 0

    while not done:
        # Initial dummy reward of 0 for first step
        reward = 0.0 if step == 0 else reward  # noqa (reward from previous)

        # Compute action from heuristic
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
        csv_writer.writerow([
            step, reward,
            kpis[0].throughput_mbps, kpis[0].avg_delay_ms, kpis[0].prb_utilization,
            kpis[1].throughput_mbps, kpis[1].avg_delay_ms, kpis[1].prb_utilization,
            kpis[2].throughput_mbps, kpis[2].avg_delay_ms, kpis[2].prb_utilization,
            action[0], action[1], action[2],
        ])

        # SLA violation summary every 50 steps
        if step % 50 == 0:
            log.info("-" * 60)
            for s in range(NUM_SLICES):
                viol = overall_violation(kpis[s], DEFAULT_SLAS[s])
                status = "✓ OK" if viol == 0 else f"✗ VIOLATION ({viol:.3f})"
                log.info(f"  {SLICE_NAMES[s]} SLA: {status}")
            log.info("-" * 60)

    # -----------------------------------------------------------------------
    # Episode summary
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info(f"Episode finished after {step} steps")
    log.info(f"Total reward: {episode_reward:.3f}")
    log.info(controller.summary())
    log.info(f"Metrics saved to: {csv_path}")
    log.info("=" * 70)

    csv_file.close()
    env.close()

    # Save final weight evolution to JSON
    json_path = Path(args.output_dir) / f"weights_{args.strategy}.json"
    weight_evolution = {
        "strategy": args.strategy,
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

def plot_results(controller: HeuristicController,
                 output_dir: str,
                 strategy: str):
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        log.warning("matplotlib not available — skipping plots.")
        return

    steps = list(range(len(controller.weight_history)))
    if not steps:
        return

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"5G Network Slicing — {strategy.replace('_', ' ').title()}",
        fontsize=16, fontweight="bold"
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    colors = {"URLLC": "#e74c3c", "eMBB": "#3498db", "mMTC": "#2ecc71"}
    slice_names_clean = ["URLLC", "eMBB", "mMTC"]

    # --- Plot 1: Weight evolution ---
    ax1 = fig.add_subplot(gs[0, :])
    w_arr = np.array(controller.weight_history)
    for s, name in enumerate(slice_names_clean):
        ax1.plot(steps, w_arr[:, s], label=name,
                 color=list(colors.values())[s], linewidth=2)
    ax1.set_title("Slice Weight Evolution (SDN Controller Output)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Normalised Weight")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Reward ---
    ax2 = fig.add_subplot(gs[1, 0])
    rewards = np.array(controller.reward_history)
    ax2.plot(steps, rewards, color="#8e44ad", linewidth=1.5, alpha=0.8)
    ax2.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Max reward")
    ax2.axhline(y=0.0, color="orange", linestyle="--", alpha=0.5, label="SLA threshold")
    ax2.fill_between(steps, rewards, 0,
                     where=(rewards >= 0), alpha=0.15, color="green")
    ax2.fill_between(steps, rewards, 0,
                     where=(rewards < 0), alpha=0.15, color="red")
    ax2.set_title("SLA Compliance Reward")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Reward")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Plot 3-5: KPI per slice ---
    for s, (name, color) in enumerate(colors.items()):
        row, col = divmod(s, 2)
        ax = fig.add_subplot(gs[1 + (s == 2), 0 + (s % 2 == 1 or s == 2)])
        # Fix layout: URLLC top-right, eMBB bottom-left, mMTC bottom-right
        ax = fig.add_subplot(gs[2, s % 2 + (1 if s == 2 else 0)])

        if controller.kpi_history:
            kpi_arr_tput = [k[s].throughput_mbps for k in controller.kpi_history]
            kpi_arr_dly  = [k[s].avg_delay_ms    for k in controller.kpi_history]

            ax2b = ax.twinx()
            ax.plot(steps, kpi_arr_tput, color=color,
                    linewidth=2, label="Throughput (Mbps)")
            ax2b.plot(steps, kpi_arr_dly, color=color,
                      linestyle="--", linewidth=1, alpha=0.7, label="Delay (ms)")

            sla = DEFAULT_SLAS[s]
            ax.axhline(y=sla.min_throughput_mbps, color=color,
                       linestyle=":", alpha=0.6, label=f"SLA tput min")
            ax2b.axhline(y=sla.max_delay_ms, color="gray",
                         linestyle=":", alpha=0.6, label=f"SLA dly max")

            ax.set_title(f"{name} Slice KPIs")
            ax.set_xlabel("Step")
            ax.set_ylabel("Throughput [Mbps]", color=color)
            ax2b.set_ylabel("Delay [ms]", color="gray")
            ax.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, f"results_{strategy}.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    log.info(f"Plot saved to: {plot_path}")
    plt.close()


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Heuristic SDN/NFV Controller for 5G Network Slicing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--strategy",
        choices=list(STRATEGIES.keys()),
        default="proportional_deficit",
        help="Heuristic resource allocation strategy")

    parser.add_argument("--port",
        type=int, default=5555,
        help="OpenGym TCP port (must match --gymPort in ns-3)")

    parser.add_argument("--sim-time",
        type=float, default=50.0,
        help="Simulation duration [s]")

    parser.add_argument("--step-interval",
        type=float, default=0.1,
        help="Gym step interval [s]")

    parser.add_argument("--n-urllc", type=int, default=5,
        help="Number of URLLC UEs")
    parser.add_argument("--n-embb",  type=int, default=10,
        help="Number of eMBB UEs")
    parser.add_argument("--n-mmtc",  type=int, default=20,
        help="Number of mMTC UEs")

    parser.add_argument("--start-sim",
        action="store_true", default=False,
        help="Start ns-3 simulation from Python (requires ns3env startSim)")

    parser.add_argument("--seed",
        type=int, default=42,
        help="Random seed for reproducibility")

    parser.add_argument("--output-dir",
        default="./results",
        help="Directory to save metrics CSV and plots")

    parser.add_argument("--plot",
        action="store_true", default=False,
        help="Generate result plots after simulation")

    parser.add_argument("--debug",
        action="store_true", default=False,
        help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    controller = run(args)

    if args.plot:
        plot_results(controller, args.output_dir, args.strategy)


if __name__ == "__main__":
    main()