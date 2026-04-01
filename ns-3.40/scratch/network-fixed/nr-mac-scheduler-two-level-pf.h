// =============================================================================
// nr-mac-scheduler-two-level-pf.h
// Two-Level Slicing Proportional Fair Scheduler
// =============================================================================

#pragma once

#include "ns3/nr-mac-scheduler-ofdma-pf.h"
#include <map>

namespace ns3
{

/**
 * \ingroup scheduler
 * \brief Two-Level Hierarchy MAC Scheduler using Proportional Fair algorithms.
 * 
 * 1. Inter-Slice Level: Sets hard/soft limits on RBG allocation per slice.
 * 2. Intra-Slice Level: Runs base Proportional Fair among UEs belonging to the slice.
 * 3. Idle-PRB Sharing: Pass 1 enforces limits. Pass 2 freely distributes idle PRBs.
 */
class NrMacSchedulerTwoLevelPF : public NrMacSchedulerOfdmaPF
{
  public:
    static TypeId GetTypeId();
    NrMacSchedulerTwoLevelPF();
    ~NrMacSchedulerTwoLevelPF() override;

    // SLA quota APIs called by Gym Controller
    void SetSliceQuota(uint8_t sliceId, double quota);
    double GetSliceQuota(uint8_t sliceId) const;

    // Simulation mapping API
    void SetUeSlice(uint16_t rnti, uint8_t sliceId);

    // Telemetry API for the Gym Environment
    uint32_t GetAndResetUsedRbg(uint8_t sliceId);
    uint32_t GetAndResetTotalRbg();

  protected:
    BeamSymbolMap AssignDLRBG(uint32_t symAvail, const ActiveUeMap& activeDl) const override;
    BeamSymbolMap AssignULRBG(uint32_t symAvail, const ActiveUeMap& activeUl) const override;

  private:
    double m_sliceQuota[3];
    std::map<uint16_t, uint8_t> m_ueSliceMap;

    // Telemetry accumulators
    mutable uint32_t m_usedRbg[3];
    mutable uint32_t m_totalRbg;
};

} // namespace ns3
