// =============================================================================
// slice-gym-env.h
// OpenGym Environment for 5G Network Slicing BWP Control
// Compatible with ns3-gym (opengym module) + ns-3 v3.40 + 5G-LENA NR
//
// OBSERVATION (12 floats, one row per slice):
//   [throughput_mbps, avg_delay_ms, prb_utilization, pkt_drop_rate] × 3 slices
//
// ACTION (3 floats in [0,1]):
//   [weight_urllc, weight_embb, weight_mmtc]
//   These are passed to NrMacSchedulerWeightedPF via registered callbacks.
//
// REWARD:
//   SLA compliance score ∈ (-∞, 1]
//   +1 when ALL slices meet SLA, decreases with each violation.
//
// DIFFERENCES FROM network-slicing/ VERSION:
//   - Single gNB with 3 BWPs (port-based flow classification)
//   - 12-float observation (adds pkt_drop_rate per slice)
//   - Tracks deadline_failure_rate (uRLLC), throughput_success_rate (eMBB),
//     buffer_drop_rate (mMTC) for the Python RIC heuristic
//   - Pre-processor invocation for per-TTI PRB estimation
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

// ──────────────────────────────────────────────────────────────────────────────
// Slice index constants
// ──────────────────────────────────────────────────────────────────────────────
constexpr uint8_t SLICE_URLLC = 0;
constexpr uint8_t SLICE_EMBB = 1;
constexpr uint8_t SLICE_MMTC = 2;
constexpr uint8_t NUM_SLICES = 3;

// ──────────────────────────────────────────────────────────────────────────────
// Per-slice KPI snapshot (populated each step)
// ──────────────────────────────────────────────────────────────────────────────
struct SliceKPI
{
    double throughputMbps{0.0}; ///< DL throughput [Mbps]
    double avgDelayMs{0.0};     ///< Mean end-to-end delay [ms]
    double prbUtilization{0.0}; ///< PRB utilisation fraction [0,1]
    double pktDropRate{0.0};    ///< Packet drop rate [0,1]
    uint32_t rxPackets{0};      ///< Packets successfully received
    uint32_t lostPackets{0};    ///< Packets lost (forwarded but not received)
    uint32_t txPackets{0};      ///< Packets transmitted
};

// ──────────────────────────────────────────────────────────────────────────────
// Per-slice SLA requirements (used for reward computation)
// ──────────────────────────────────────────────────────────────────────────────
struct SliceSLA
{
    double minThroughputMbps{0.0}; ///< Minimum required throughput
    double maxDelayMs{1e9};        ///< Maximum tolerated delay
    double maxPrbUtil{1.0};        ///< Maximum PRB utilisation (0–1)
};

// ──────────────────────────────────────────────────────────────────────────────
// Port range for flow classification (single-gNB, port-based routing)
// ──────────────────────────────────────────────────────────────────────────────
struct PortRange
{
    uint16_t start{0};
    uint16_t end{0};
};

// ──────────────────────────────────────────────────────────────────────────────
// SliceGymEnv — main OpenGym environment class
// ──────────────────────────────────────────────────────────────────────────────
class SliceGymEnv : public OpenGymEnv
{
  public:
    static TypeId GetTypeId();

    SliceGymEnv();
    ~SliceGymEnv() override;

    // ─────────────────────────────────────────────────────────────────────────
    // Configuration — call before Simulator::Run()
    // ─────────────────────────────────────────────────────────────────────────

    /** Attach the FlowMonitor so we can read per-flow statistics. */
    void SetFlowMonitor(Ptr<FlowMonitor> fm, Ptr<Ipv4FlowClassifier> classifier);

    /** Set SLA requirements for a slice. */
    void SetSliceSLA(uint8_t sliceId,
                     double minThroughMbps,
                     double maxDelayMs,
                     double maxPrbUtil = 1.0);

    /**
     * Register the port range for a slice so flows can be classified.
     * (Single-gNB design: all UEs share same subnet, we classify by port)
     * \param sliceId   0/1/2
     * \param portStart Starting port (inclusive)
     * \param portEnd   Ending port (inclusive)
     */
    void RegisterSlicePorts(uint8_t sliceId, uint16_t portStart, uint16_t portEnd);

    /**
     * Register a callback that the gym env will invoke to apply new weights.
     * Signature: void ApplyWeights(double wUrllc, double wEmbb, double wMmtc)
     */
    using WeightSetterCb = std::function<void(double, double, double)>;
    void SetWeightSetterCallback(WeightSetterCb cb);

    /** Start periodic telemetry collection (must call after all config). */
    void ScheduleNextStateRead();

    // ─────────────────────────────────────────────────────────────────────────
    // Public KPI accessors (for simulation logging / NetAnim)
    // ─────────────────────────────────────────────────────────────────────────
    const SliceKPI& GetKPI(uint8_t sliceId) const;

    // ─────────────────────────────────────────────────────────────────────────
    // OpenGymEnv interface (overrides)
    // ─────────────────────────────────────────────────────────────────────────
    Ptr<OpenGymSpace> GetObservationSpace() override;
    Ptr<OpenGymSpace> GetActionSpace() override;
    Ptr<OpenGymDataContainer> GetObservation() override;
    float GetReward() override;
    bool GetGameOver() override;
    std::string GetExtraInfo() override;
    bool ExecuteActions(Ptr<OpenGymDataContainer> action) override;

    /**
     * Trace sink for SlotDataStats and SlotCtrlStats (PRB utilization).
     */
    void SlotStatsTraceSink(uint32_t sliceId,
                            const SfnSf& sfnSf,
                            uint32_t scheduledUe,
                            uint32_t usedReg,
                            uint32_t usedSym,
                            uint32_t availableRb,
                            uint32_t availableSym,
                            uint16_t bwpId,
                            uint16_t cellId);

  private:
    // ─────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────────
    void CollectTelemetry();
    int8_t ClassifyFlowToSlice(uint16_t srcPort, uint16_t dstPort) const;

    // ─────────────────────────────────────────────────────────────────────────
    // State
    // ─────────────────────────────────────────────────────────────────────────
    Ptr<FlowMonitor> m_flowMonitor;
    Ptr<Ipv4FlowClassifier> m_classifier;

    std::array<SliceKPI, NUM_SLICES> m_kpis;
    std::array<SliceSLA, NUM_SLICES> m_sla;

    // Port range info for flow classification
    std::array<PortRange, NUM_SLICES> m_slicePorts;

    // PRB usage tracking from PHY traces
    std::array<uint64_t, NUM_SLICES> m_usedPrbReg;
    std::array<uint64_t, NUM_SLICES> m_availablePrbReg;

    // Previous flow stats to compute deltas
    std::map<FlowId, FlowMonitor::FlowStats> m_prevStats;
    double m_prevTime{0.0};

    // Interval between gym steps [s]
    double m_stepInterval{0.1};

    // Weight setter callback
    WeightSetterCb m_weightSetterCb;

    // Current weights (for logging)
    std::array<double, NUM_SLICES> m_currentWeights{{1.0 / 3, 1.0 / 3, 1.0 / 3}};

    bool m_done{false};
    uint32_t m_stepCount{0};
    uint32_t m_maxSteps{500};

    // 12 floats: [tput, dly, prb, drop] × 3 slices
    static constexpr uint32_t OBS_SIZE = NUM_SLICES * 4;
};

} // namespace ns3