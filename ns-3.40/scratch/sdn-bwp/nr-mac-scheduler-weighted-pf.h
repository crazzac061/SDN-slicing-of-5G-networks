// =============================================================================
// nr-mac-scheduler-weighted-pf.h
// Weight-Based Proportional Fair Scheduler with BWP Multiplexer & Pre-processor
// Compatible with ns-3 v3.40 + 5G-LENA NR Module
//
// This scheduler implements THREE algorithms from the O-RAN architecture:
//
// 1. WEIGHTED PF SCHEDULING:
//    Weight → PF time-window scaling (higher weight = shorter window = more
//    aggressive). Single weight per BWP/scheduler instance, adjusted by the
//    Python RIC via the OpenGym action callback.
//
// 2. PRE-PROCESSOR (per-TTI resource estimation):
//    - eMBB:  PRBs = min(R_throughput, R_queue) based on CQI
//    - uRLLC: EDF sort → only allocate for packets within 2 TTIs of deadline
//    - mMTC:  Aggregate queue, allocate bulk PRBs using robust base MCS
//
// 3. BWP MULTIPLEXER (per-UE BWP activation):
//    Multi-slice UEs use strict priority: uRLLC > eMBB > mMTC
//    Starvation counters (α) with force-activation thresholds (αmax)
//    prevent lower-priority slices from being permanently starved.
// =============================================================================

#pragma once

#include "ns3/nr-mac-scheduler-ofdma-pf.h"
#include "ns3/object.h"

#include <cstdint>
#include <map>
#include <array>

namespace ns3
{

// ──────────────────────────────────────────────────────────────────────────────
// Slice identifiers (must match slice-gym-env.h)
// ──────────────────────────────────────────────────────────────────────────────
constexpr uint8_t BWP_URLLC = 0;
constexpr uint8_t BWP_EMBB  = 1;
constexpr uint8_t BWP_MMTC  = 2;
constexpr uint8_t NUM_BWP_SLICES = 3;

// ──────────────────────────────────────────────────────────────────────────────
// Per-UE starvation tracking for BWP Multiplexer
// ──────────────────────────────────────────────────────────────────────────────
struct StarvationCounter
{
    uint32_t embbCounter {0};   ///< How many slots eMBB was blocked
    uint32_t mmtcCounter {0};   ///< How many slots mMTC was blocked
};

/**
 * \ingroup scheduler
 * \brief Weight-Based Proportional Fair Scheduler with BWP Mux & Pre-processor
 *
 * Extends NrMacSchedulerOfdmaPF. Each BWP/slice gets one scheduler instance.
 * The SDN/NFV controller (Python RIC) adjusts the per-BWP weight, which is
 * translated into a PF time-window to control scheduling aggressiveness.
 *
 * Additionally implements:
 * - Pre-processor: per-TTI PRB estimation per slice type
 * - BWP Multiplexer: per-UE BWP activation with starvation thresholds
 */
class NrMacSchedulerWeightedPF : public NrMacSchedulerOfdmaPF
{
  public:
    static TypeId GetTypeId();

    NrMacSchedulerWeightedPF();
    ~NrMacSchedulerWeightedPF() override = default;

    // ─────────────────────────────────────────────────────────────────────────
    // Primary API: single weight per scheduler instance
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * \brief Set the scheduling weight for this slice's scheduler.
     * \param weight Positive float. 1.0 = neutral. 3.0 = triple priority.
     *
     * Weight is applied by scaling the PF time window:
     *   timeWindow = BASE_WINDOW_MS / weight
     */
    void SetWeight(double weight);

    /** \return Current raw weight */
    double GetWeight() const { return m_weight; }

    // ─────────────────────────────────────────────────────────────────────────
    // Backward-compatible multi-index API
    // ─────────────────────────────────────────────────────────────────────────
    void SetSliceWeight(uint8_t bwpId, double weight);
    double GetSliceWeight(uint8_t bwpId) const;

    // Attribute-friendly wrappers
    void SetSliceWeightAttr(double weight);
    double GetSliceWeightAttr() const;

    // ─────────────────────────────────────────────────────────────────────────
    // PRE-PROCESSOR ALGORITHM (Section 2)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * \brief Set the slice type for this scheduler instance.
     * Determines which pre-processor logic is applied.
     * 0=URLLC, 1=eMBB, 2=mMTC
     */
    void SetSliceType(uint8_t sliceType);
    uint8_t GetSliceType() const { return m_sliceType; }

    /**
     * \brief Compute effective PRBs needed for this slice in the current TTI.
     * Uses slice-type-specific pre-processor logic.
     *
     * \param queueBytes     Total bytes queued across all UEs in this BWP
     * \param avgCqi         Average CQI across active UEs (0-15)
     * \param numActiveUes   Number of UEs with data to send
     * \param targetTputBps  Target throughput in bps (for eMBB)
     * \param availablePrbs  Total PRBs available in this BWP
     * \return               Effective PRBs recommended by pre-processor
     */
    uint32_t PreprocessPrbs(uint64_t queueBytes,
                            double avgCqi,
                            uint32_t numActiveUes,
                            double targetTputBps,
                            uint32_t availablePrbs) const;

    // ─────────────────────────────────────────────────────────────────────────
    // BWP MULTIPLEXER ALGORITHM (Section 3)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * \brief Determine which BWP should be activated for a given UE.
     *
     * Priority: uRLLC > eMBB > mMTC with starvation prevention.
     *
     * \param ueId           Node ID of the UE
     * \param hasUrllcData   True if UE has pending URLLC data
     * \param hasEmbbData    True if UE has pending eMBB data
     * \param hasMmtcData    True if UE has pending mMTC data
     * \return               BWP index to activate (0=URLLC, 1=eMBB, 2=mMTC)
     */
    static uint8_t MultiplexBwp(uint32_t ueId,
                                bool hasUrllcData,
                                bool hasEmbbData,
                                bool hasMmtcData);

    /**
     * \brief Reset starvation counters for a specific UE.
     */
    static void ResetStarvationCounters(uint32_t ueId);

    // Starvation thresholds (configurable)
    static uint32_t s_alphaMaxEmbb;   ///< Max consecutive slots eMBB can be starved
    static uint32_t s_alphaMaxMmtc;   ///< Max consecutive slots mMTC can be starved

    // Non-static getter/setter wrappers for ns-3 attribute system
    void SetAlphaMaxEmbb(uint32_t v) { s_alphaMaxEmbb = v; }
    uint32_t GetAlphaMaxEmbb() const { return s_alphaMaxEmbb; }
    void SetAlphaMaxMmtc(uint32_t v) { s_alphaMaxMmtc = v; }
    uint32_t GetAlphaMaxMmtc() const { return s_alphaMaxMmtc; }

    static constexpr double BASE_WINDOW_MS = 99.0; ///< Default PF time window [ms]

  private:
    double  m_weight    {1.0};   ///< Current slice weight
    uint8_t m_sliceType {1};     ///< 0=URLLC, 1=eMBB, 2=mMTC

    /** Translate m_weight → PF TimeWindow attribute and apply it. */
    void ApplyWeightAsTimeWindow();

    /**
     * \brief Estimate spectral efficiency from CQI.
     * Maps CQI (0-15) to bits/PRB using 3GPP table approximation.
     */
    static double CqiToSpectralEfficiency(double cqi);

    // Static starvation counter map shared across all BWP scheduler instances
    static std::map<uint32_t, StarvationCounter> s_starvationMap;
};

} // namespace ns3
