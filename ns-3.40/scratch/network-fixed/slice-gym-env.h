// =============================================================================
// slice-gym-env.h
// OpenGym Environment for 5G Network Slicing Control
// Compatible with ns3-gym (opengym module) + ns-3 v3.40 + 5G-LENA NR
//
// OBSERVATION (9 floats, one row per slice):
//   [throughput_mbps, avg_delay_ms, prb_utilization]  × 3 slices
//
// ACTION (3 floats in [0,1]):
//   [weight_urllc, weight_embb, weight_mmtc]
//   These are passed to NrMacSchedulerWeightedPF via registered callbacks.
//
// REWARD:
//   SLA compliance score  ∈ (-∞, 1]
//   +1 when ALL slices meet SLA, decreases with each violation.
// =============================================================================

#pragma once

#include "ns3/callback.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/ipv4-flow-classifier.h"
#include "ns3/net-device-container.h"
#include "ns3/node-container.h"
#include "ns3/nr-phy-mac-common.h"
#include "ns3/opengym-module.h"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <string>

namespace ns3
{

class NrMacSchedulerTwoLevelPF;

// ---------------------------------------------------------------------------
// Slice index constants — used everywhere for clarity
// ---------------------------------------------------------------------------
constexpr uint8_t SLICE_URLLC = 0;
constexpr uint8_t SLICE_EMBB = 1;
constexpr uint8_t SLICE_MMTC = 2;
constexpr uint8_t NUM_SLICES = 3;

// ---------------------------------------------------------------------------
// Per-slice KPI snapshot (populated each step)
// ---------------------------------------------------------------------------
struct SliceKPI
{
    double throughputMbps{0.0};    ///< Aggregate DL throughput [Mbps]
    double avgThroughputMbps{0.0}; ///< Per-UE mean throughput [Mbps]
    double avgDelayMs{0.0};        ///< Mean end-to-end delay [ms]
    double prbUtilization{0.0};    ///< PRB utilisation fraction [0,1]
    uint32_t rxPackets{0};         ///< Packets successfully received
    uint32_t lostPackets{0};       ///< Packets lost
    uint32_t activeUes{0};         ///< Number of active UEs detected in this slice
};

// ---------------------------------------------------------------------------
// Per-slice SLA requirements (used for reward computation)
// ---------------------------------------------------------------------------
struct SliceSLA
{
    double minThroughputMbps{0.0}; ///< Minimum required throughput
    double maxDelayMs{1e9};        ///< Maximum tolerated delay
    double maxPrbUtil{1.0};        ///< Maximum PRB utilisation (0–1)
};

// ---------------------------------------------------------------------------
// SliceGymEnv — main gym environment class
// ---------------------------------------------------------------------------

/**
 * \brief OpenGym environment for heuristic-based 5G network slice management.
 *
 * The simulation creates one instance of this class and calls
 * ScheduleNextStateRead() to set up the periodic telemetry-collection loop.
 *
 * Every step_interval seconds the environment:
 *   1. Reads FlowMonitor stats to compute per-slice KPIs.
 *   2. Reports observation + reward to the Python controller.
 *   3. Waits for an action (weight vector) from the Python controller.
 *   4. Applies weights to the registered scheduler via the weight-setter
 *      callback.
 */
class SliceGymEnv : public OpenGymEnv
{
  public:
    static TypeId GetTypeId();

    SliceGymEnv();
    ~SliceGymEnv() override;

    // -----------------------------------------------------------------------
    // Configuration — call before Simulator::Run()
    // -----------------------------------------------------------------------

    /** Attach the FlowMonitor so we can read per-flow statistics. */
    void SetFlowMonitor(Ptr<FlowMonitor> fm, Ptr<Ipv4FlowClassifier> classifier);

    /** Set SLA requirements for a slice. */
    void SetSliceSLA(uint8_t sliceId,
                     double minThroughMbps,
                     double maxDelayMs,
                     double maxPrbUtil = 1.0);

    /**
     * Register the IP subnet for a slice so flows can be classified.
     * \param sliceId   0/1/2
     * \param network   e.g. "10.1.1.0"
     * \param mask      e.g. "255.255.255.0"
     */
    void RegisterSliceSubnet(uint8_t sliceId, Ipv4Address network, Ipv4Mask mask);

    /**
     * Register a callback that the gym env will invoke to apply new weights.
     * Signature:  void ApplyWeights(double wUrllc, double wEmbb, double wMmtc)
     */
    using WeightSetterCb = std::function<void(double, double, double)>;
    void SetWeightSetterCallback(WeightSetterCb cb);

    /** Start periodic telemetry collection (must call after all config). */
    void ScheduleNextStateRead();

    // -----------------------------------------------------------------------
    // Public KPI accessors (for simulation logging)
    // -----------------------------------------------------------------------
    const SliceKPI& GetKPI(uint8_t sliceId) const;

    // -----------------------------------------------------------------------
    // OpenGymEnv interface (overrides)
    // -----------------------------------------------------------------------
    Ptr<OpenGymSpace> GetObservationSpace() override;
    Ptr<OpenGymSpace> GetActionSpace() override;
    Ptr<OpenGymDataContainer> GetObservation() override;
    float GetReward() override;
    bool GetGameOver() override;
    std::string GetExtraInfo() override;
    bool ExecuteActions(Ptr<OpenGymDataContainer> action) override;

    // Enable direct telemetry polling from the Two-Level MAC Scheduler
    void SetMacScheduler(Ptr<NrMacSchedulerTwoLevelPF> sched);

  private:
    Ptr<NrMacSchedulerTwoLevelPF> m_scheduler;
    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------
    void CollectTelemetry();
    bool IsInSlice(uint8_t sliceId, Ipv4Address src, Ipv4Address dst) const;

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    Ptr<FlowMonitor> m_flowMonitor;
    Ptr<Ipv4FlowClassifier> m_classifier;

    std::array<SliceKPI, NUM_SLICES> m_kpis;
    std::array<SliceSLA, NUM_SLICES> m_sla;

    // Subnet info for flow classification
    std::array<Ipv4Address, NUM_SLICES> m_sliceNet;
    std::array<Ipv4Mask, NUM_SLICES> m_sliceMask;

    std::array<uint64_t, NUM_SLICES> m_usedPrbReg;      // Cumulative RB-symbols used
    std::array<uint64_t, NUM_SLICES> m_availablePrbReg; // Cumulative RB-symbols available

    // Previous flow stats to compute deltas
    std::map<FlowId, FlowMonitor::FlowStats> m_prevStats;
    double m_prevTime{0.0}; ///< Simulation time at previous step [s]

    // Interval between gym steps [s]
    double m_stepInterval{0.01};

    // Weight setter callback
    WeightSetterCb m_weightSetterCb;

    // Current weights (for logging)
    std::array<double, NUM_SLICES> m_currentWeights{{1.0 / 3, 1.0 / 3, 1.0 / 3}};

    bool m_done{false};
    uint32_t m_stepCount{0};
    uint32_t m_maxSteps{5000}; ///< Episode length

    static constexpr uint32_t OBS_SIZE = NUM_SLICES * 3; // 9 floats
};

} // namespace ns3