// =============================================================================
// nr-mac-scheduler-weighted-pf.h
// Weight-Based Proportional Fair Scheduler for 5G Network Slicing
// Compatible with ns-3 v3.40 + 5G-LENA NR Module
// =============================================================================
//
// DESIGN: Each BWP/slice gets its own scheduler INSTANCE. So instead of
// storing 3 per-slice weights here, this class holds a SINGLE weight that
// represents this slice's "aggressiveness" relative to a neutral baseline.
//
// Weight effect: applied via PF time-window scaling.
//   weight > 1.0  → shorter time window → more aggressive (more reactive PF)
//   weight < 1.0  → longer  time window → more conservative
//   weight = 1.0  → neutral (default 99-subframe window unchanged)
//
// The SDN controller normalises its output over 3 slices to sum to 1.
// ApplyWeights() in slice-simulation.cc multiplies by 3 so the neutral
// value maps to weight = 1.0 for each scheduler instance.
//
// score(UE) = achievableRate / avgThroughput(window)
//   where window = BASE_WINDOW / max(weight, 0.01)
// =============================================================================

#pragma once

#include "ns3/nr-mac-scheduler-ofdma-pf.h"
#include "ns3/object.h"

#include <cstdint>

namespace ns3
{

/**
 * \ingroup scheduler
 * \brief Weight-Based Proportional Fair Scheduler
 *
 * Extends NrMacSchedulerOfdmaPF. Each slice (BWP) gets one instance.
 * The SDN/NFV controller adjusts the single slice weight, which is then
 * translated into a PF time-window to make the scheduler more or less
 * aggressive in serving its UEs.
 *
 * Slice-to-BWP mapping (set in slice-simulation.cc):
 *   BWP 0  →  URLLC  (Ultra-Reliable Low Latency)
 *   BWP 1  →  eMBB   (Enhanced Mobile Broadband)
 *   BWP 2  →  mMTC   (Massive Machine-Type Communications)
 */
class NrMacSchedulerWeightedPF : public NrMacSchedulerOfdmaPF
{
  public:
    static TypeId GetTypeId();

    NrMacSchedulerWeightedPF();
    ~NrMacSchedulerWeightedPF() override = default;

    // -----------------------------------------------------------------------
    // Primary API: single weight per scheduler instance
    // -----------------------------------------------------------------------

    /**
     * \brief Set the scheduling weight for this slice's scheduler.
     * \param weight  Positive float. 1.0 = neutral. 3.0 = triple priority.
     *
     * Weight is applied by scaling the PF time window:
     *   timeWindow = BASE_WINDOW_MS / weight
     * A shorter window makes the scheduler more reactive to instantaneous
     * channel conditions, effectively granting more frequent high-rate
     * allocations.
     */
    void SetWeight(double weight);

    /** \return Current raw weight (before clamping) */
    double GetWeight() const
    {
        return m_weight;
    }

    // -----------------------------------------------------------------------
    // Backward-compatible multi-index API (used by existing ApplyWeights code)
    // bwpId is accepted but ignored — this instance serves exactly one slice.
    // -----------------------------------------------------------------------

    /**
     * \brief Set weight; bwpId is accepted for API compatibility but ignored.
     * The caller should only pass the index matching this scheduler's BWP.
     */
    void SetSliceWeight(uint8_t bwpId, double weight);
    double GetSliceWeight(uint8_t bwpId) const;

    // Attribute-friendly wrappers (no index parameter)
    void SetSliceWeightAttr(double weight);
    double GetSliceWeightAttr() const;

    static constexpr double BASE_WINDOW_MS = 99.0; // default PF time window [ms]

  private:
    double m_weight{1.0}; ///< Current slice weight (1.0 = neutral)

    /** Translate m_weight → PF TimeWindow attribute and apply it. */
    void ApplyWeightAsTimeWindow();
};

} // namespace ns3