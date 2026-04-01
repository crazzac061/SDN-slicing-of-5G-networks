// =============================================================================
// nr-mac-scheduler-two-level-pf.cc
// Two-Level Slicing Proportional Fair Scheduler
// =============================================================================

#include "nr-mac-scheduler-two-level-pf.h"
#include "ns3/log.h"
#include <algorithm>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("NrMacSchedulerTwoLevelPF");
NS_OBJECT_ENSURE_REGISTERED(NrMacSchedulerTwoLevelPF);

TypeId
NrMacSchedulerTwoLevelPF::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::NrMacSchedulerTwoLevelPF")
            .SetParent<NrMacSchedulerOfdmaPF>()
            .SetGroupName("nr")
            .AddConstructor<NrMacSchedulerTwoLevelPF>();
    return tid;
}

NrMacSchedulerTwoLevelPF::NrMacSchedulerTwoLevelPF()
    : NrMacSchedulerOfdmaPF()
{
    NS_LOG_FUNCTION(this);
    m_sliceQuota[0] = 0.33;
    m_sliceQuota[1] = 0.33;
    m_sliceQuota[2] = 0.33;

    m_usedRbg[0] = 0;
    m_usedRbg[1] = 0;
    m_usedRbg[2] = 0;
    m_totalRbg = 0;
}

NrMacSchedulerTwoLevelPF::~NrMacSchedulerTwoLevelPF()
{
}

void
NrMacSchedulerTwoLevelPF::SetSliceQuota(uint8_t sliceId, double quota)
{
    if (sliceId < 3)
    {
        m_sliceQuota[sliceId] = quota;
    }
}

double
NrMacSchedulerTwoLevelPF::GetSliceQuota(uint8_t sliceId) const
{
    return (sliceId < 3) ? m_sliceQuota[sliceId] : 0.0;
}

void
NrMacSchedulerTwoLevelPF::SetUeSlice(uint16_t rnti, uint8_t sliceId)
{
    m_ueSliceMap[rnti] = sliceId;
}

uint32_t
NrMacSchedulerTwoLevelPF::GetAndResetUsedRbg(uint8_t sliceId)
{
    if (sliceId >= 3) return 0;
    uint32_t val = m_usedRbg[sliceId];
    m_usedRbg[sliceId] = 0;
    return val;
}

uint32_t
NrMacSchedulerTwoLevelPF::GetAndResetTotalRbg()
{
    uint32_t val = m_totalRbg;
    m_totalRbg = 0;
    return val;
}

// ---------------------------------------------------------------------------
// DOWNLINK ASSIGNMENT (TWO-PASS ALGORITHM)
// ---------------------------------------------------------------------------
NrMacSchedulerNs3::BeamSymbolMap
NrMacSchedulerTwoLevelPF::AssignDLRBG(uint32_t symAvail, const ActiveUeMap& activeDl) const
{
    NS_LOG_FUNCTION(this);
    GetFirst GetBeamId;
    GetSecond GetUeVector;
    BeamSymbolMap symPerBeam = GetSymPerBeam(symAvail, activeDl);

    for (const auto& el : activeDl)
    {
        uint32_t beamSym = symPerBeam.at(GetBeamId(el));
        uint32_t rbgAssignable = 1 * beamSym;
        std::vector<UePtrAndBufferReq> ueVector;
        FTResources assigned(0, 0);
        
        const std::vector<uint8_t> dlNotchedRBGsMask = GetDlNotchedRbgMask();
        uint32_t resources = dlNotchedRBGsMask.size() > 0
                                 ? std::count(dlNotchedRBGsMask.begin(), dlNotchedRBGsMask.end(), 1)
                                 : GetBandwidthInRbg();
        NS_ASSERT(resources > 0);

        m_totalRbg += resources; // Accumulate telemetry

        for (const auto& ue : GetUeVector(el))
        {
            ueVector.emplace_back(ue);
            BeforeDlSched(ue, FTResources(rbgAssignable * beamSym, beamSym));
        }

        // Calculate Quotas in RBGs
        std::map<uint8_t, uint32_t> sliceRbgQuota;
        std::map<uint8_t, uint32_t> sliceRbgAssigned;
        for (uint8_t i = 0; i < 3; ++i) { 
            sliceRbgQuota[i] = static_cast<uint32_t>(resources * m_sliceQuota[i]);
            sliceRbgAssigned[i] = 0;
        }

        // =========================================================================
        // PASS 1: Strict Quota Enforcement (SLA Guarantee)
        // =========================================================================
        bool pass1Active = true;
        while (resources > 0 && pass1Active)
        {
            pass1Active = false; // Assume pass 1 is dead unless we make an assignment
            
            GetFirst GetUe;
            auto pfSort = GetUeCompareDlFn();
            auto customSort = [this, pfSort](const UePtrAndBufferReq& a, const UePtrAndBufferReq& b) {
                GetFirst GetUe;
                uint16_t rntiA = GetUe(a)->m_rnti;
                uint16_t rntiB = GetUe(b)->m_rnti;
                uint8_t sliceA = m_ueSliceMap.count(rntiA) ? m_ueSliceMap.at(rntiA) : 3;
                uint8_t sliceB = m_ueSliceMap.count(rntiB) ? m_ueSliceMap.at(rntiB) : 3;
                if (sliceA != sliceB) return sliceA < sliceB;
                return pfSort(a, b);
            };
            std::sort(ueVector.begin(), ueVector.end(), customSort);
            auto schedInfoIt = ueVector.begin();

            while (schedInfoIt != ueVector.end())
            {
                uint32_t bufQueueSize = schedInfoIt->second;
                uint32_t tbSize = 0;
                for (const auto& it : GetUe(*schedInfoIt)->m_dlTbSize) tbSize += it;

                // Standard OFDMA buffer satisfaction check
                if (tbSize >= std::max(bufQueueSize, 10U)) {
                    if (GetUe(*schedInfoIt)->m_dlTbSize.size() > 1) {
                        uint8_t streamCounter = 0;
                        uint32_t copyBufQueueSize = bufQueueSize;
                        auto dlTbSizeIt = GetUe(*schedInfoIt)->m_dlTbSize.begin();
                        while (dlTbSizeIt != GetUe(*schedInfoIt)->m_dlTbSize.end()) {
                            if (copyBufQueueSize != 0) {
                                if (*dlTbSizeIt >= copyBufQueueSize) copyBufQueueSize = 0;
                                else copyBufQueueSize = copyBufQueueSize - *dlTbSizeIt;
                                streamCounter++;
                                dlTbSizeIt++;
                            } else {
                                *dlTbSizeIt = 0;
                                streamCounter++;
                                dlTbSizeIt++;
                            }
                        }
                    }
                    schedInfoIt++;
                    continue; // Check next UE
                }

                // Buffer needs data! 
                // Check if the UE's slice has consumed its quota
                uint16_t rnti = GetUe(*schedInfoIt)->m_rnti;
                uint8_t sliceId = m_ueSliceMap.count(rnti) ? m_ueSliceMap.at(rnti) : 0;
                
                if (sliceRbgAssigned[sliceId] >= sliceRbgQuota[sliceId]) {
                    schedInfoIt++; // Skip UE, its slice quota is full
                    continue;
                }

                // SLA met! UE selected.
                break;
            }

            if (schedInfoIt != ueVector.end())
            {
                // Assign 1 RBG
                GetUe(*schedInfoIt)->m_dlRBG += rbgAssignable;
                assigned.m_rbg += rbgAssignable;
                GetUe(*schedInfoIt)->m_dlSym = beamSym;
                assigned.m_sym = beamSym;

                uint16_t rnti = GetUe(*schedInfoIt)->m_rnti;
                uint8_t sliceId = m_ueSliceMap.count(rnti) ? m_ueSliceMap.at(rnti) : 0;
                
                sliceRbgAssigned[sliceId] += 1;
                m_usedRbg[sliceId] += 1; // Update telemetry
                resources -= 1;
                pass1Active = true; // We made an assignment, keep going!

                AssignedDlResources(*schedInfoIt, FTResources(rbgAssignable, beamSym), assigned);
                for (auto& ue : ueVector)
                {
                    if (GetUe(ue)->m_rnti != rnti)
                    {
                        NotAssignedDlResources(ue, FTResources(rbgAssignable, beamSym), assigned);
                    }
                }
            }
        }

        // =========================================================================
        // PASS 2: Idle PRB Sharing (No Limits)
        // =========================================================================
        while (resources > 0)
        {
            GetFirst GetUe;
            auto pfSort = GetUeCompareDlFn();
            auto customSort = [this, pfSort](const UePtrAndBufferReq& a, const UePtrAndBufferReq& b) {
                GetFirst GetUe;
                uint16_t rntiA = GetUe(a)->m_rnti;
                uint16_t rntiB = GetUe(b)->m_rnti;
                uint8_t sliceA = m_ueSliceMap.count(rntiA) ? m_ueSliceMap.at(rntiA) : 3;
                uint8_t sliceB = m_ueSliceMap.count(rntiB) ? m_ueSliceMap.at(rntiB) : 3;
                if (sliceA != sliceB) return sliceA < sliceB;
                return pfSort(a, b);
            };
            std::sort(ueVector.begin(), ueVector.end(), customSort);
            auto schedInfoIt = ueVector.begin();

            while (schedInfoIt != ueVector.end())
            {
                uint32_t bufQueueSize = schedInfoIt->second;
                uint32_t tbSize = 0;
                for (const auto& it : GetUe(*schedInfoIt)->m_dlTbSize) tbSize += it;

                if (tbSize >= std::max(bufQueueSize, 10U)) {
                    if (GetUe(*schedInfoIt)->m_dlTbSize.size() > 1) {
                        // same MIMO reset
                        uint8_t streamCounter = 0;
                        uint32_t copyBufQueueSize = bufQueueSize;
                        auto dlTbSizeIt = GetUe(*schedInfoIt)->m_dlTbSize.begin();
                        while (dlTbSizeIt != GetUe(*schedInfoIt)->m_dlTbSize.end()) {
                            if (copyBufQueueSize != 0) {
                                if (*dlTbSizeIt >= copyBufQueueSize) copyBufQueueSize = 0;
                                else copyBufQueueSize = copyBufQueueSize - *dlTbSizeIt;
                                streamCounter++;
                                dlTbSizeIt++;
                            } else {
                                *dlTbSizeIt = 0;
                                streamCounter++;
                                dlTbSizeIt++;
                            }
                        }
                    }
                    schedInfoIt++;
                } else {
                    break;
                }
            }

            if (schedInfoIt == ueVector.end()) {
                break; // No active UEs left across all slices
            }

            GetUe(*schedInfoIt)->m_dlRBG += rbgAssignable;
            assigned.m_rbg += rbgAssignable;
            GetUe(*schedInfoIt)->m_dlSym = beamSym;
            assigned.m_sym = beamSym;

            uint16_t rnti = GetUe(*schedInfoIt)->m_rnti;
            uint8_t sliceId = m_ueSliceMap.count(rnti) ? m_ueSliceMap.at(rnti) : 0;
            m_usedRbg[sliceId] += 1;
            resources -= 1;

            AssignedDlResources(*schedInfoIt, FTResources(rbgAssignable, beamSym), assigned);
            for (auto& ue : ueVector)
            {
                if (GetUe(ue)->m_rnti != rnti)
                {
                    NotAssignedDlResources(ue, FTResources(rbgAssignable, beamSym), assigned);
                }
            }
        }
    }

    return symPerBeam;
}

// ---------------------------------------------------------------------------
// UPLINK ASSIGNMENT (TWO-PASS ALGORITHM)
// ---------------------------------------------------------------------------
NrMacSchedulerNs3::BeamSymbolMap
NrMacSchedulerTwoLevelPF::AssignULRBG(uint32_t symAvail, const ActiveUeMap& activeUl) const
{
    NS_LOG_FUNCTION(this);
    GetFirst GetBeamId;
    GetSecond GetUeVector;
    BeamSymbolMap symPerBeam = GetSymPerBeam(symAvail, activeUl);

    for (const auto& el : activeUl)
    {
        uint32_t beamSym = symPerBeam.at(GetBeamId(el));
        uint32_t rbgAssignable = 1 * beamSym;
        std::vector<UePtrAndBufferReq> ueVector;
        FTResources assigned(0, 0);
        
        const std::vector<uint8_t> ulNotchedRBGsMask = GetUlNotchedRbgMask();
        uint32_t resources = ulNotchedRBGsMask.size() > 0
                                 ? std::count(ulNotchedRBGsMask.begin(), ulNotchedRBGsMask.end(), 1)
                                 : GetBandwidthInRbg();
        NS_ASSERT(resources > 0);

        for (const auto& ue : GetUeVector(el))
        {
            ueVector.emplace_back(ue);
            BeforeUlSched(ue, FTResources(rbgAssignable * beamSym, beamSym));
        }

        std::map<uint8_t, uint32_t> sliceRbgQuota;
        std::map<uint8_t, uint32_t> sliceRbgAssigned;
        for (uint8_t i = 0; i < 3; ++i) { 
            sliceRbgQuota[i] = static_cast<uint32_t>(resources * m_sliceQuota[i]);
            sliceRbgAssigned[i] = 0;
        }

        // PASS 1: Strict Quotas
        bool pass1Active = true;
        while (resources > 0 && pass1Active)
        {
            pass1Active = false;
            GetFirst GetUe;
            auto pfSort = GetUeCompareUlFn();
            auto customSort = [this, pfSort](const UePtrAndBufferReq& a, const UePtrAndBufferReq& b) {
                GetFirst GetUe;
                uint16_t rntiA = GetUe(a)->m_rnti;
                uint16_t rntiB = GetUe(b)->m_rnti;
                uint8_t sliceA = m_ueSliceMap.count(rntiA) ? m_ueSliceMap.at(rntiA) : 3;
                uint8_t sliceB = m_ueSliceMap.count(rntiB) ? m_ueSliceMap.at(rntiB) : 3;
                if (sliceA != sliceB) return sliceA < sliceB;
                return pfSort(a, b);
            };
            std::sort(ueVector.begin(), ueVector.end(), customSort);
            auto schedInfoIt = ueVector.begin();

            while (schedInfoIt != ueVector.end())
            {
                uint32_t bufQueueSize = schedInfoIt->second;
                if (GetUe(*schedInfoIt)->m_ulTbSize >= std::max(bufQueueSize, 12U)) {
                    schedInfoIt++;
                    continue;
                }

                uint16_t rnti = GetUe(*schedInfoIt)->m_rnti;
                uint8_t sliceId = m_ueSliceMap.count(rnti) ? m_ueSliceMap.at(rnti) : 0;
                if (sliceRbgAssigned[sliceId] >= sliceRbgQuota[sliceId]) {
                    schedInfoIt++;
                    continue;
                }
                break;
            }

            if (schedInfoIt != ueVector.end())
            {
                GetUe(*schedInfoIt)->m_ulRBG += rbgAssignable;
                assigned.m_rbg += rbgAssignable;
                GetUe(*schedInfoIt)->m_ulSym = beamSym;
                assigned.m_sym = beamSym;

                uint16_t rnti = GetUe(*schedInfoIt)->m_rnti;
                uint8_t sliceId = m_ueSliceMap.count(rnti) ? m_ueSliceMap.at(rnti) : 0;
                
                sliceRbgAssigned[sliceId] += 1;
                // No telemetry tracking for UL needed right now
                resources -= 1;
                pass1Active = true;

                AssignedUlResources(*schedInfoIt, FTResources(rbgAssignable, beamSym), assigned);
                for (auto& ue : ueVector)
                {
                    if (GetUe(ue)->m_rnti != rnti)
                    {
                        NotAssignedUlResources(ue, FTResources(rbgAssignable, beamSym), assigned);
                    }
                }
            }
        }

        // PASS 2: Idle PRB Sharing
        while (resources > 0)
        {
            GetFirst GetUe;
            auto pfSort = GetUeCompareUlFn();
            auto customSort = [this, pfSort](const UePtrAndBufferReq& a, const UePtrAndBufferReq& b) {
                GetFirst GetUe;
                uint16_t rntiA = GetUe(a)->m_rnti;
                uint16_t rntiB = GetUe(b)->m_rnti;
                uint8_t sliceA = m_ueSliceMap.count(rntiA) ? m_ueSliceMap.at(rntiA) : 3;
                uint8_t sliceB = m_ueSliceMap.count(rntiB) ? m_ueSliceMap.at(rntiB) : 3;
                if (sliceA != sliceB) return sliceA < sliceB;
                return pfSort(a, b);
            };
            std::sort(ueVector.begin(), ueVector.end(), customSort);
            auto schedInfoIt = ueVector.begin();

            while (schedInfoIt != ueVector.end())
            {
                uint32_t bufQueueSize = schedInfoIt->second;
                if (GetUe(*schedInfoIt)->m_ulTbSize >= std::max(bufQueueSize, 12U)) {
                    schedInfoIt++;
                } else {
                    break;
                }
            }

            if (schedInfoIt == ueVector.end()) {
                break;
            }

            GetUe(*schedInfoIt)->m_ulRBG += rbgAssignable;
            assigned.m_rbg += rbgAssignable;
            GetUe(*schedInfoIt)->m_ulSym = beamSym;
            assigned.m_sym = beamSym;

            uint16_t rnti = GetUe(*schedInfoIt)->m_rnti;
            uint8_t sliceId = m_ueSliceMap.count(rnti) ? m_ueSliceMap.at(rnti) : 0;
            
            resources -= 1;

            AssignedUlResources(*schedInfoIt, FTResources(rbgAssignable, beamSym), assigned);
            for (auto& ue : ueVector)
            {
                if (GetUe(ue)->m_rnti != rnti)
                {
                    NotAssignedUlResources(ue, FTResources(rbgAssignable, beamSym), assigned);
                }
            }
        }
    }

    return symPerBeam;
}

} // namespace ns3
