// =============================================================================
// slice-gym-env.cc
// OpenGym Environment for 5G Network Slicing — Implementation
// =============================================================================
// FIXES APPLIED:
//   Bug 4 — Added #include <iomanip> (needed for std::setprecision, std::fixed)
//   Bug 5 — Fixed IsInSlice: dst.CombineMask(mask)==net  (was always false)
//   Bug 6 — Added PRB utilisation estimation from throughput + peak capacity
// =============================================================================

#include "slice-gym-env.h"

#include "ns3/log.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/simulator.h"
#include "ns3/opengym-module.h"

#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>   // FIX Bug 4: needed for std::setprecision / std::fixed

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("SliceGymEnv");
NS_OBJECT_ENSURE_REGISTERED (SliceGymEnv);

// ---------------------------------------------------------------------------
// Peak-capacity estimates used for PRB utilisation approximation (Bug 6 fix).
// These match the BWP configuration in slice-simulation.cc:
//   BWP 0 URLLC:  10 MHz @ 28 GHz  (mmWave, UMi)  → ~50 Mbps peak DL
//   BWP 1 eMBB:   40 MHz @  3.5 GHz (UMa)          → ~200 Mbps peak DL
//   BWP 2 mMTC:   10 MHz @  0.7 GHz (RMa)           → ~30 Mbps peak DL
// Adjust these if you change BW or propagation scenario.
// ---------------------------------------------------------------------------
static constexpr double PEAK_MBPS[NUM_SLICES] = { 50.0, 500.0, 30.0 }; // IMPROVEMENT: eMBB 500 Mbps for 100 MHz

// ---------------------------------------------------------------------------
TypeId
SliceGymEnv::GetTypeId ()
{
    static TypeId tid =
        TypeId ("ns3::SliceGymEnv")
            .SetParent<OpenGymEnv> ()
            .SetGroupName ("OpenGym")
            .AddConstructor<SliceGymEnv> ()
            .AddAttribute ("StepInterval",
                           "Period between gym steps [seconds]",
                           DoubleValue (0.1),
                           MakeDoubleAccessor (&SliceGymEnv::m_stepInterval),
                           MakeDoubleChecker<double> (0.01, 10.0))
            .AddAttribute ("MaxSteps",
                           "Maximum number of steps per episode",
                           UintegerValue (500),
                           MakeUintegerAccessor (&SliceGymEnv::m_maxSteps),
                           MakeUintegerChecker<uint32_t> (1));
    return tid;
}

SliceGymEnv::SliceGymEnv ()
{
    NS_LOG_FUNCTION (this);

    // ---------- Default SLA requirements ----------
    m_sla[SLICE_URLLC] = { .minThroughputMbps = 1.0,
                            .maxDelayMs        = 1.0,
                            .maxPrbUtil        = 0.85 };

    m_sla[SLICE_EMBB]  = { .minThroughputMbps = 20.0,
                            .maxDelayMs        = 50.0,
                            .maxPrbUtil        = 0.90 };

    m_sla[SLICE_MMTC]  = { .minThroughputMbps = 0.1,
                            .maxDelayMs        = 200.0,
                            .maxPrbUtil        = 0.80 };

    m_usedPrbReg.fill (0);
    m_availablePrbReg.fill (0);
}

SliceGymEnv::~SliceGymEnv ()
{
    NS_LOG_FUNCTION (this);
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

void
SliceGymEnv::SetFlowMonitor (Ptr<FlowMonitor> fm, Ptr<Ipv4FlowClassifier> classifier)
{
    m_flowMonitor  = fm;
    m_classifier   = classifier;
}

void
SliceGymEnv::SetSliceSLA (uint8_t sliceId,
                           double minThroughMbps,
                           double maxDelayMs,
                           double maxPrbUtil)
{
    NS_ASSERT_MSG (sliceId < NUM_SLICES, "Slice ID out of range");
    m_sla[sliceId] = { minThroughMbps, maxDelayMs, maxPrbUtil };
}

void
SliceGymEnv::RegisterSliceSubnet (uint8_t sliceId,
                                   Ipv4Address network,
                                   Ipv4Mask    mask)
{
    NS_ASSERT_MSG (sliceId < NUM_SLICES, "Slice ID out of range");
    m_sliceNet [sliceId] = network;
    m_sliceMask[sliceId] = mask;
}

void
SliceGymEnv::SetWeightSetterCallback (WeightSetterCb cb)
{
    m_weightSetterCb = cb;
}

// ---------------------------------------------------------------------------
// Periodic step scheduling
// ---------------------------------------------------------------------------

void
SliceGymEnv::ScheduleNextStateRead ()
{
    Simulator::Schedule (Seconds (m_stepInterval),
                         &SliceGymEnv::CollectTelemetry,
                         this);
}

// ---------------------------------------------------------------------------
// Telemetry collection — called every step_interval seconds
// ---------------------------------------------------------------------------

void
SliceGymEnv::SlotStatsTraceSink (uint32_t sliceId, const SfnSf& sfnSf, uint32_t scheduledUe,
                                 uint32_t usedReg, uint32_t usedSym, uint32_t availableRb,
                                 uint32_t availableSym, uint16_t bwpId, uint16_t cellId)
{
    if (sliceId < NUM_SLICES)
    {
        m_usedPrbReg[sliceId] += usedReg;
        m_availablePrbReg[sliceId] += (uint64_t)availableRb * availableSym;
    }
}

void
SliceGymEnv::CollectTelemetry ()
{
    NS_LOG_FUNCTION (this << Simulator::Now ().GetSeconds ());

    double now = Simulator::Now ().GetSeconds ();
    double dt  = now - m_prevTime;

    // Reset KPIs (throughput and packets only; prbUtilization computed below)
    for (auto& k : m_kpis) k = SliceKPI{};

    if (m_flowMonitor && dt > 0.0)
        {
            m_flowMonitor->CheckForLostPackets ();
            const auto& stats = m_flowMonitor->GetFlowStats ();

            std::array<double,   NUM_SLICES> totalDelay   {};
            std::array<uint32_t, NUM_SLICES> delayCount   {};

            for (const auto& [fid, fs] : stats)
                {
                    Ipv4FlowClassifier::FiveTuple t =
                        m_classifier->FindFlow (fid);

                    int8_t sliceId = -1;
                    for (uint8_t s = 0; s < NUM_SLICES; ++s)
                        {
                            // FIX Bug 5: was  dst == net.CombineMask(mask)
                            //            now  dst.CombineMask(mask) == net
                            if (IsInSlice (s, t.sourceAddress, t.destinationAddress))
                                {
                                    sliceId = s;
                                    break;
                                }
                        }
                    if (sliceId < 0) continue;

                    uint64_t prevRxBytes  = 0;
                    uint32_t prevRxPkts   = 0;
                    uint32_t prevLostPkts = 0;
                    double   prevDelay    = 0.0;

                    if (m_prevStats.count (fid))
                        {
                            const auto& p  = m_prevStats.at (fid);
                            prevRxBytes    = p.rxBytes;
                            prevRxPkts     = p.rxPackets;
                            prevLostPkts   = p.lostPackets;
                            prevDelay      = p.delaySum.GetSeconds ();
                        }

                    uint64_t deltaBytes = fs.rxBytes - prevRxBytes;
                    double   deltaBits  = static_cast<double> (deltaBytes) * 8.0;

                    m_kpis[sliceId].throughputMbps  += (deltaBits / dt) / 1e6;
                    m_kpis[sliceId].rxPackets       += fs.rxPackets - prevRxPkts;
                    m_kpis[sliceId].lostPackets     += fs.lostPackets - prevLostPkts;

                    uint32_t rxDelta = fs.rxPackets - prevRxPkts;
                    if (rxDelta > 0)
                        {
                            double delayDelta = fs.delaySum.GetSeconds () - prevDelay;
                            totalDelay[sliceId]  += delayDelta;
                            delayCount[sliceId]  += rxDelta;
                        }
                }

            // Average delay per slice [ms]
            for (uint8_t s = 0; s < NUM_SLICES; ++s)
                {
                    if (delayCount[s] > 0)
                        m_kpis[s].avgDelayMs =
                            (totalDelay[s] / delayCount[s]) * 1e3;
                }

            m_prevStats = stats;
            m_prevTime  = now;
        }

    // BUG 6 Fix: Use actual PRB usage from MAC/PHY trace sinks.
    for (uint8_t s = 0; s < NUM_SLICES; ++s)
        {
            if (m_availablePrbReg[s] > 0)
                {
                    m_kpis[s].prbUtilization =
                        static_cast<double> (m_usedPrbReg[s]) / m_availablePrbReg[s];
                }
            else
                {
                    m_kpis[s].prbUtilization = 0.0;
                }
            
            // Reset counters for the next step interval
            m_usedPrbReg[s] = 0;
            m_availablePrbReg[s] = 0;
        }

    // Log KPIs
    for (uint8_t s = 0; s < NUM_SLICES; ++s)
        {
            static const char* names[] = { "URLLC", "eMBB ", "mMTC " };
            NS_LOG_INFO ("[" << names[s] << "] "
                             << "tput=" << std::fixed << std::setprecision(2)
                             << m_kpis[s].throughputMbps << " Mbps  "
                             << "delay=" << m_kpis[s].avgDelayMs << " ms  "
                             << "prb_util=" << m_kpis[s].prbUtilization);
        }

    Notify ();

    if (!m_done)
        ScheduleNextStateRead ();
}

// ---------------------------------------------------------------------------
// Flow-to-slice classification
// ---------------------------------------------------------------------------

bool
SliceGymEnv::IsInSlice (uint8_t sliceId,
                         Ipv4Address /*src*/,
                         Ipv4Address dst) const
{
    // FIX Bug 5: original code was:
    //   return (dst == net.CombineMask(mask));
    // This compared dst against the network address itself, which was wrong
    // because CombineMask() on a /24 network address just returns the same
    // address.  The correct check: mask the destination and compare to the
    // registered network prefix.
    Ipv4Mask    mask = m_sliceMask[sliceId];
    Ipv4Address net  = m_sliceNet[sliceId];
    return dst.CombineMask (mask) == net;
}

// ---------------------------------------------------------------------------
// OpenGymEnv interface
// ---------------------------------------------------------------------------

Ptr<OpenGymSpace>
SliceGymEnv::GetObservationSpace ()
{
    std::vector<uint32_t> shape = { OBS_SIZE };

    float low  [OBS_SIZE];
    float high [OBS_SIZE];

    for (uint32_t i = 0; i < NUM_SLICES; ++i)
        {
            low [i*3+0] = 0.0f;   high[i*3+0] = 1000.0f;  // throughput [Mbps]
            low [i*3+1] = 0.0f;   high[i*3+1] = 1000.0f;  // delay [ms]
            low [i*3+2] = 0.0f;   high[i*3+2] = 1.0f;     // prb_util [0,1]
        }

    Ptr<OpenGymBoxSpace> space =
        CreateObject<OpenGymBoxSpace> (
            std::vector<float> (low,  low  + OBS_SIZE),
            std::vector<float> (high, high + OBS_SIZE),
            shape, TypeNameGet<float> ());
    return space;
}

Ptr<OpenGymSpace>
SliceGymEnv::GetActionSpace ()
{
    std::vector<uint32_t> shape = { NUM_SLICES };
    std::vector<float> low  (NUM_SLICES, 0.0f);
    std::vector<float> high (NUM_SLICES, 1.0f);

    Ptr<OpenGymBoxSpace> space =
        CreateObject<OpenGymBoxSpace> (low, high, shape, TypeNameGet<float> ());
    return space;
}

Ptr<OpenGymDataContainer>
SliceGymEnv::GetObservation ()
{
    std::vector<uint32_t> shape = { OBS_SIZE };
    Ptr<OpenGymBoxContainer<float>> obs =
        CreateObject<OpenGymBoxContainer<float>> (shape);

    for (uint8_t s = 0; s < NUM_SLICES; ++s)
        {
            obs->AddValue (static_cast<float> (m_kpis[s].throughputMbps));
            obs->AddValue (static_cast<float> (m_kpis[s].avgDelayMs));
            obs->AddValue (static_cast<float> (m_kpis[s].prbUtilization));
        }

    return obs;
}

float
SliceGymEnv::GetReward ()
{
    // IMPROVEMENT: Log-sum reward for better multi-slice balance (Recommendation 6)
    // Reward = sum(log1p(compliance_s))
    // compliance_s = max(0, 1 - total_normalized_violation_s)
    double totalReward = 0.0;

    for (uint8_t s = 0; s < NUM_SLICES; ++s)
        {
            const auto& kpi_s = m_kpis[s];
            const auto& sla_s = m_sla[s];
            double sliceViol = 0.0;

            if (sla_s.minThroughputMbps > 0 && kpi_s.throughputMbps < sla_s.minThroughputMbps)
                {
                    sliceViol += (sla_s.minThroughputMbps - kpi_s.throughputMbps) / sla_s.minThroughputMbps;
                }

            if (kpi_s.avgDelayMs > sla_s.maxDelayMs && kpi_s.avgDelayMs > 0)
                {
                    sliceViol += (kpi_s.avgDelayMs - sla_s.maxDelayMs) / sla_s.maxDelayMs;
                }

            if (kpi_s.prbUtilization > sla_s.maxPrbUtil)
                {
                    sliceViol += (kpi_s.prbUtilization - sla_s.maxPrbUtil) / (1.0 - sla_s.maxPrbUtil + 1e-6);
                }

            double compliance = std::max (0.0, 1.0 - sliceViol);
            totalReward += std::log1p (compliance); // log(1 + compliance)
        }

    NS_LOG_INFO ("Step " << m_stepCount << "  reward=" << totalReward);
    return static_cast<float> (totalReward);
}

bool
SliceGymEnv::GetGameOver ()
{
    m_done = (++m_stepCount >= m_maxSteps);
    return m_done;
}

std::string
SliceGymEnv::GetExtraInfo ()
{
    std::ostringstream oss;
    static const char* names[] = { "URLLC", "eMBB", "mMTC" };

    oss << "step=" << m_stepCount << " | ";
    for (uint8_t s = 0; s < NUM_SLICES; ++s)
        {
            oss << names[s]
                << "[tput=" << std::fixed << std::setprecision(2)
                << m_kpis[s].throughputMbps
                << "Mbps,dly=" << m_kpis[s].avgDelayMs << "ms"
                << ",prb=" << std::setprecision(3) << m_kpis[s].prbUtilization
                << ",w=" << m_currentWeights[s] << "] ";
        }
    return oss.str ();
}

bool
SliceGymEnv::ExecuteActions (Ptr<OpenGymDataContainer> action)
{
    Ptr<OpenGymBoxContainer<float>> box =
        DynamicCast<OpenGymBoxContainer<float>> (action);

    if (!box)
        {
            NS_LOG_ERROR ("Action container is null or wrong type");
            return false;
        }

    std::vector<float> vals = box->GetData ();
    if (vals.size () < NUM_SLICES)
        {
            NS_LOG_ERROR ("Action vector too short: " << vals.size ());
            return false;
        }

    // IMPROVEMENT: Enforce 0.1 minimum weight floor to prevent starvation (Recommendation 3)
    double w0 = std::max (0.1, static_cast<double> (vals[0]));
    double w1 = std::max (0.1, static_cast<double> (vals[1]));
    double w2 = std::max (0.1, static_cast<double> (vals[2]));

    double sum = w0 + w1 + w2;
    w0 /= sum;  w1 /= sum;  w2 /= sum;

    m_currentWeights[SLICE_URLLC] = w0;
    m_currentWeights[SLICE_EMBB]  = w1;
    m_currentWeights[SLICE_MMTC]  = w2;

    NS_LOG_INFO ("New weights  URLLC=" << w0
                                       << "  eMBB=" << w1
                                       << "  mMTC=" << w2);

    if (m_weightSetterCb)
        m_weightSetterCb (w0, w1, w2);

    return true;
}

// ---------------------------------------------------------------------------
const SliceKPI&
SliceGymEnv::GetKPI (uint8_t sliceId) const
{
    NS_ASSERT_MSG (sliceId < NUM_SLICES, "Slice ID out of range");
    return m_kpis[sliceId];
}

} // namespace ns3