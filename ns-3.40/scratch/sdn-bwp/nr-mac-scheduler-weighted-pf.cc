// =============================================================================
// nr-mac-scheduler-weighted-pf.cc
// Weight-Based PF Scheduler + Pre-processor + BWP Multiplexer — Implementation
//
// Implements all three gNB-side algorithms:
//   1. Weighted PF scheduling (PF window scaling by weight)
//   2. Pre-processor: per-slice, per-TTI PRB estimation
//   3. BWP Multiplexer: per-UE BWP activation with starvation prevention
// =============================================================================

#include "nr-mac-scheduler-weighted-pf.h"

#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/uinteger.h"

#include <algorithm>
#include <cmath>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("NrMacSchedulerWeightedPF");
NS_OBJECT_ENSURE_REGISTERED(NrMacSchedulerWeightedPF);

// ──────────────────────────────────────────────────────────────────────────────
// Static member initialization
// ──────────────────────────────────────────────────────────────────────────────
std::map<uint32_t, StarvationCounter> NrMacSchedulerWeightedPF::s_starvationMap;
uint32_t NrMacSchedulerWeightedPF::s_alphaMaxEmbb = 8;   // Force eMBB activation after 8 slots
uint32_t NrMacSchedulerWeightedPF::s_alphaMaxMmtc = 16;  // Force mMTC activation after 16 slots

// ──────────────────────────────────────────────────────────────────────────────
// TypeId registration
// ──────────────────────────────────────────────────────────────────────────────
TypeId
NrMacSchedulerWeightedPF::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::NrMacSchedulerWeightedPF")
            .SetParent<NrMacSchedulerOfdmaPF>()
            .SetGroupName("nr")
            .AddConstructor<NrMacSchedulerWeightedPF>()
            .AddAttribute("SliceWeight",
                          "Scheduling aggressiveness weight for this slice. "
                          "1.0 = neutral. >1.0 = more aggressive (shorter PF window). "
                          "<1.0 = more conservative (longer PF window).",
                          DoubleValue(1.0),
                          MakeDoubleAccessor(&NrMacSchedulerWeightedPF::SetSliceWeightAttr,
                                            &NrMacSchedulerWeightedPF::GetSliceWeightAttr),
                          MakeDoubleChecker<double>(0.01, 100.0))
            .AddAttribute("AlphaMaxEmbb",
                          "Max consecutive slots eMBB can be starved by uRLLC "
                          "before forced activation.",
                          UintegerValue(8),
                          MakeUintegerAccessor(&NrMacSchedulerWeightedPF::SetAlphaMaxEmbb,
                                              &NrMacSchedulerWeightedPF::GetAlphaMaxEmbb),
                          MakeUintegerChecker<uint32_t>(1, 100))
            .AddAttribute("AlphaMaxMmtc",
                          "Max consecutive slots mMTC can be starved by higher "
                          "priority slices before forced activation.",
                          UintegerValue(16),
                          MakeUintegerAccessor(&NrMacSchedulerWeightedPF::SetAlphaMaxMmtc,
                                              &NrMacSchedulerWeightedPF::GetAlphaMaxMmtc),
                          MakeUintegerChecker<uint32_t>(1, 200));
    return tid;
}

NrMacSchedulerWeightedPF::NrMacSchedulerWeightedPF()
    : NrMacSchedulerOfdmaPF()
{
    NS_LOG_FUNCTION(this);
}

// ──────────────────────────────────────────────────────────────────────────────
// ALGORITHM 1: Weight-based PF scheduling
// ──────────────────────────────────────────────────────────────────────────────

void
NrMacSchedulerWeightedPF::SetWeight(double weight)
{
    NS_LOG_FUNCTION(this << weight);
    NS_ASSERT_MSG(weight > 0.0, "Slice weight must be strictly positive");

    m_weight = weight;
    ApplyWeightAsTimeWindow();

    NS_LOG_INFO("Slice weight set to " << weight
                 << "  → PF TimeWindow = "
                 << std::max(1.0, BASE_WINDOW_MS / weight) << " ms");
}

void
NrMacSchedulerWeightedPF::SetSliceWeight(uint8_t bwpId, double weight)
{
    NS_LOG_FUNCTION(this << +bwpId << weight);
    SetWeight(weight);
}

double
NrMacSchedulerWeightedPF::GetSliceWeight(uint8_t /*bwpId*/) const
{
    return m_weight;
}

void
NrMacSchedulerWeightedPF::SetSliceWeightAttr(double weight)
{
    SetWeight(weight);
}

double
NrMacSchedulerWeightedPF::GetSliceWeightAttr() const
{
    return m_weight;
}

void
NrMacSchedulerWeightedPF::ApplyWeightAsTimeWindow()
{
    // PF metric = achievableRate / avgThroughput(timeWindow)
    // Shorter window → avg decays faster → metric is more sensitive to
    // current channel quality → scheduler serves good-channel UEs harder.
    //
    // Mapping: timeWindow = BASE_WINDOW / weight
    //   weight=1.0 → timeWindow=99  (default, neutral)
    //   weight=3.0 → timeWindow=33  (3× more reactive)
    //   weight=0.5 → timeWindow=198 (2× more conservative)

    double tw = std::max(1.0, BASE_WINDOW_MS / m_weight);

    // "LastAvgTPutWeight" is the PF time-window attribute registered in
    // NrMacSchedulerOfdmaPF::GetTypeId().
    SetAttribute("LastAvgTPutWeight", DoubleValue(tw));
}

// ──────────────────────────────────────────────────────────────────────────────
// ALGORITHM 2: Pre-processor (per-slice, per-TTI PRB estimation)
// ──────────────────────────────────────────────────────────────────────────────

void
NrMacSchedulerWeightedPF::SetSliceType(uint8_t sliceType)
{
    NS_ASSERT_MSG(sliceType < NUM_BWP_SLICES, "Invalid slice type");
    m_sliceType = sliceType;
    NS_LOG_INFO("Scheduler slice type set to " << +sliceType);
}

double
NrMacSchedulerWeightedPF::CqiToSpectralEfficiency(double cqi)
{
    // 3GPP TS 38.214 Table 5.2.2.1-2 (simplified mapping)
    // CQI → bits/RE (approximate spectral efficiency)
    // Each PRB has 12 subcarriers × 14 symbols = 168 REs (Numerology 0)
    static const double cqiTable[16] = {
        0.0,     // CQI 0: out of range
        0.1523,  // CQI 1:  QPSK, code rate ~78/1024
        0.2344,  // CQI 2:  QPSK, code rate ~120/1024
        0.3770,  // CQI 3:  QPSK, code rate ~193/1024
        0.6016,  // CQI 4:  QPSK, code rate ~308/1024
        0.8770,  // CQI 5:  QPSK, code rate ~449/1024
        1.1758,  // CQI 6:  QPSK, code rate ~602/1024
        1.4766,  // CQI 7:  16QAM, code rate ~378/1024
        1.9141,  // CQI 8:  16QAM, code rate ~490/1024
        2.4063,  // CQI 9:  16QAM, code rate ~616/1024
        2.7305,  // CQI 10: 64QAM, code rate ~466/1024
        3.3223,  // CQI 11: 64QAM, code rate ~567/1024
        3.9023,  // CQI 12: 64QAM, code rate ~666/1024
        4.5234,  // CQI 13: 64QAM, code rate ~772/1024
        5.1152,  // CQI 14: 64QAM, code rate ~873/1024
        5.5547,  // CQI 15: 64QAM, code rate ~948/1024
    };

    int idx = std::max(0, std::min(15, static_cast<int>(std::round(cqi))));
    return cqiTable[idx];
}

uint32_t
NrMacSchedulerWeightedPF::PreprocessPrbs(uint64_t queueBytes,
                                          double avgCqi,
                                          uint32_t numActiveUes,
                                          double targetTputBps,
                                          uint32_t availablePrbs) const
{
    NS_LOG_FUNCTION(this << queueBytes << avgCqi << numActiveUes
                         << targetTputBps << availablePrbs);

    if (numActiveUes == 0 || queueBytes == 0)
    {
        NS_LOG_DEBUG("No active UEs or empty queue → 0 PRBs");
        return 0;
    }

    switch (m_sliceType)
    {
    case BWP_URLLC:
    {
        // ──────────────────────────────────────────────────────────────────
        // uRLLC Pre-processor: Earliest Deadline First (EDF)
        //
        // Sort LCs by deadline. Only allocate PRBs for PDUs whose deadline
        // expires within the next 2 TTIs. This avoids over-allocating
        // resources for packets with distant deadlines.
        //
        // Since we cannot directly access LC deadlines from the scheduler
        // level, we estimate: all queued bytes are urgent (typical for
        // uRLLC with 1ms latency budget ≈ 2 TTIs at numerology 2).
        //
        // PRBs = ceil(urgentBytes * 8 / bitsPerPrb)
        // ──────────────────────────────────────────────────────────────────
        double se = CqiToSpectralEfficiency(avgCqi);
        if (se <= 0.0)
            se = 0.1523; // Worst-case QPSK

        // Bits per PRB: spectral_efficiency * 12 subcarriers * 14 symbols
        // For numerology 2 (60 kHz SCS): 12 × 14 = 168 REs per PRB
        double bitsPerPrb = se * 168.0;

        // Assume all uRLLC bytes are within 2-TTI deadline (conservative)
        uint64_t urgentBytes = queueBytes;
        uint32_t prbsNeeded = static_cast<uint32_t>(
            std::ceil(static_cast<double>(urgentBytes) * 8.0 / bitsPerPrb));

        uint32_t result = std::min(prbsNeeded, availablePrbs);
        NS_LOG_DEBUG("uRLLC EDF: urgentBytes=" << urgentBytes
                     << " bitsPerPRB=" << bitsPerPrb
                     << " → PRBs=" << result);
        return result;
    }

    case BWP_EMBB:
    {
        // ──────────────────────────────────────────────────────────────────
        // eMBB Pre-processor: CQI/Throughput-aware
        //
        // Calculate R (PRBs needed for target throughput based on CQI)
        // and N (PRBs needed to empty the queue in this TTI).
        // Allocate min(R, N) to prevent overallocating PRBs to empty queues.
        //
        // R = targetTputBps / (bitsPerPrb * slotsPerSecond)
        // N = ceil(queueBytes * 8 / bitsPerPrb)
        // ──────────────────────────────────────────────────────────────────
        double se = CqiToSpectralEfficiency(avgCqi);
        if (se <= 0.0)
            se = 1.4766; // Default to CQI 7 (16QAM)

        // For numerology 1 (30 kHz SCS): 2000 slots/sec
        double bitsPerPrb = se * 168.0;
        double slotsPerSec = 2000.0; // Numerology 1: 2^1 * 1000

        // R: PRBs for target throughput
        uint32_t R = (targetTputBps > 0 && bitsPerPrb > 0)
            ? static_cast<uint32_t>(std::ceil(targetTputBps / (bitsPerPrb * slotsPerSec)))
            : availablePrbs;

        // N: PRBs to drain queue this TTI
        uint32_t N = static_cast<uint32_t>(
            std::ceil(static_cast<double>(queueBytes) * 8.0 / bitsPerPrb));

        uint32_t result = std::min({R, N, availablePrbs});
        NS_LOG_DEBUG("eMBB: R=" << R << " N=" << N
                     << " avail=" << availablePrbs
                     << " → PRBs=" << result);
        return result;
    }

    case BWP_MMTC:
    {
        // ──────────────────────────────────────────────────────────────────
        // mMTC Pre-processor: Bulk Round-Robin
        //
        // Do NOT track per-UE CQI (too expensive for massive nodes).
        // Aggregate total queue size of active mMTC nodes.
        // Allocate PRBs based on aggregated queue using robust base MCS
        // (QPSK, code rate ~1/3 → spectral efficiency ~0.3770, CQI 3).
        // ──────────────────────────────────────────────────────────────────
        constexpr double ROBUST_SE = 0.3770; // CQI 3: QPSK, robust modulation

        // For numerology 0 (15 kHz SCS): 168 REs per PRB
        double bitsPerPrb = ROBUST_SE * 168.0;

        // Aggregate queue across all mMTC nodes
        uint32_t prbsNeeded = static_cast<uint32_t>(
            std::ceil(static_cast<double>(queueBytes) * 8.0 / bitsPerPrb));

        uint32_t result = std::min(prbsNeeded, availablePrbs);
        NS_LOG_DEBUG("mMTC bulk: totalQueue=" << queueBytes
                     << " numUEs=" << numActiveUes
                     << " → PRBs=" << result);
        return result;
    }

    default:
        NS_LOG_WARN("Unknown slice type " << +m_sliceType << "; returning all available PRBs");
        return availablePrbs;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ALGORITHM 3: BWP Multiplexer (per-UE BWP activation with starvation control)
// ──────────────────────────────────────────────────────────────────────────────

uint8_t
NrMacSchedulerWeightedPF::MultiplexBwp(uint32_t ueId,
                                        bool hasUrllcData,
                                        bool hasEmbbData,
                                        bool hasMmtcData)
{
    NS_LOG_FUNCTION(ueId << hasUrllcData << hasEmbbData << hasMmtcData);

    // Get or create starvation counters for this UE
    auto& sc = s_starvationMap[ueId];

    // ── Check for starvation force-activation FIRST ──────────────────────
    //
    // If eMBB has been starved beyond threshold, force eMBB activation
    if (hasEmbbData && sc.embbCounter >= s_alphaMaxEmbb)
    {
        NS_LOG_INFO("BWP Mux: UE " << ueId << " FORCE eMBB activation"
                     << " (α_embb=" << sc.embbCounter
                     << " >= αmax=" << s_alphaMaxEmbb << ")");
        sc.embbCounter = 0; // Reset eMBB starvation counter
        // mMTC still increments since it wasn't served
        if (hasMmtcData)
            sc.mmtcCounter++;
        return BWP_EMBB;
    }

    // If mMTC has been starved beyond threshold, force mMTC activation
    if (hasMmtcData && sc.mmtcCounter >= s_alphaMaxMmtc)
    {
        NS_LOG_INFO("BWP Mux: UE " << ueId << " FORCE mMTC activation"
                     << " (α_mmtc=" << sc.mmtcCounter
                     << " >= αmax=" << s_alphaMaxMmtc << ")");
        sc.mmtcCounter = 0; // Reset mMTC starvation counter
        // eMBB still increments since it wasn't served
        if (hasEmbbData)
            sc.embbCounter++;
        return BWP_MMTC;
    }

    // ── Standard priority: uRLLC > eMBB > mMTC ─────────────────────────
    if (hasUrllcData)
    {
        // uRLLC wins. Increment starvation counters for eMBB and mMTC.
        if (hasEmbbData)
            sc.embbCounter++;
        if (hasMmtcData)
            sc.mmtcCounter++;

        NS_LOG_DEBUG("BWP Mux: UE " << ueId << " → uRLLC"
                     << " (α_embb=" << sc.embbCounter
                     << ", α_mmtc=" << sc.mmtcCounter << ")");
        return BWP_URLLC;
    }

    if (hasEmbbData)
    {
        // eMBB served → reset its starvation counter
        sc.embbCounter = 0;
        // mMTC still starved
        if (hasMmtcData)
            sc.mmtcCounter++;

        NS_LOG_DEBUG("BWP Mux: UE " << ueId << " → eMBB"
                     << " (α_mmtc=" << sc.mmtcCounter << ")");
        return BWP_EMBB;
    }

    if (hasMmtcData)
    {
        // mMTC served → reset its starvation counter
        sc.mmtcCounter = 0;

        NS_LOG_DEBUG("BWP Mux: UE " << ueId << " → mMTC");
        return BWP_MMTC;
    }

    // No data → default to eMBB (idle BWP)
    return BWP_EMBB;
}

void
NrMacSchedulerWeightedPF::ResetStarvationCounters(uint32_t ueId)
{
    auto it = s_starvationMap.find(ueId);
    if (it != s_starvationMap.end())
    {
        it->second.embbCounter = 0;
        it->second.mmtcCounter = 0;
    }
}

} // namespace ns3
