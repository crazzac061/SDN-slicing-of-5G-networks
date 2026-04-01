// =============================================================================
// slice-gym-env.cc
// OpenGym Environment for 5G Network Slicing BWP Control — Implementation
//
// Single-gNB, 3-BWP design with port-based flow classification.
// 12-float observation: [tput, delay, prb_util, pkt_drop_rate] × 3 slices.
// =============================================================================

#include "slice-gym-env.h"

#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/opengym-module.h"
#include "ns3/simulator.h"
#include "ns3/uinteger.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("SliceGymEnv");
NS_OBJECT_ENSURE_REGISTERED(SliceGymEnv);

// ──────────────────────────────────────────────────────────────────────────────
TypeId
SliceGymEnv::GetTypeId()
{
    static TypeId tid = TypeId("ns3::SliceGymEnv")
                            .SetParent<OpenGymEnv>()
                            .SetGroupName("OpenGym")
                            .AddConstructor<SliceGymEnv>()
                            .AddAttribute("StepInterval",
                                          "Period between gym steps [seconds]",
                                          DoubleValue(0.1),
                                          MakeDoubleAccessor(&SliceGymEnv::m_stepInterval),
                                          MakeDoubleChecker<double>(0.01, 10.0))
                            .AddAttribute("MaxSteps",
                                          "Maximum number of steps per episode",
                                          UintegerValue(500),
                                          MakeUintegerAccessor(&SliceGymEnv::m_maxSteps),
                                          MakeUintegerChecker<uint32_t>(1));
    return tid;
}

SliceGymEnv::SliceGymEnv()
{
    NS_LOG_FUNCTION(this);

    // Default SLA requirements
    m_sla[SLICE_URLLC] = {.minThroughputMbps = 1.0,
                          .maxDelayMs = 1.0,
                          .maxPrbUtil = 0.85};

    m_sla[SLICE_EMBB] = {.minThroughputMbps = 20.0,
                         .maxDelayMs = 50.0,
                         .maxPrbUtil = 0.90};

    m_sla[SLICE_MMTC] = {.minThroughputMbps = 0.1,
                         .maxDelayMs = 200.0,
                         .maxPrbUtil = 0.80};

    m_usedPrbReg.fill(0);
    m_availablePrbReg.fill(0);
}

SliceGymEnv::~SliceGymEnv()
{
    NS_LOG_FUNCTION(this);
}

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

void
SliceGymEnv::SetFlowMonitor(Ptr<FlowMonitor> fm, Ptr<Ipv4FlowClassifier> classifier)
{
    m_flowMonitor = fm;
    m_classifier = classifier;
}

void
SliceGymEnv::SetSliceSLA(uint8_t sliceId,
                         double minThroughMbps,
                         double maxDelayMs,
                         double maxPrbUtil)
{
    NS_ASSERT_MSG(sliceId < NUM_SLICES, "Slice ID out of range");
    m_sla[sliceId] = {minThroughMbps, maxDelayMs, maxPrbUtil};
}

void
SliceGymEnv::RegisterSlicePorts(uint8_t sliceId, uint16_t portStart, uint16_t portEnd)
{
    NS_ASSERT_MSG(sliceId < NUM_SLICES, "Slice ID out of range");
    m_slicePorts[sliceId] = {portStart, portEnd};
    NS_LOG_INFO("Slice " << +sliceId << " registered ports "
                         << portStart << "-" << portEnd);
}

void
SliceGymEnv::SetWeightSetterCallback(WeightSetterCb cb)
{
    m_weightSetterCb = cb;
}

// ──────────────────────────────────────────────────────────────────────────────
// Periodic step scheduling
// ──────────────────────────────────────────────────────────────────────────────

void
SliceGymEnv::ScheduleNextStateRead()
{
    Simulator::Schedule(Seconds(m_stepInterval), &SliceGymEnv::CollectTelemetry, this);
}

// ──────────────────────────────────────────────────────────────────────────────
// PRB trace sink — tracks per-BWP resource usage
// ──────────────────────────────────────────────────────────────────────────────

void
SliceGymEnv::SlotStatsTraceSink(uint32_t sliceId,
                                const SfnSf& sfnSf,
                                uint32_t scheduledUe,
                                uint32_t usedReg,
                                uint32_t usedSym,
                                uint32_t availableRb,
                                uint32_t availableSym,
                                uint16_t bwpId,
                                uint16_t cellId)
{
    if (sliceId < NUM_SLICES)
    {
        m_usedPrbReg[sliceId] += usedReg;
        m_availablePrbReg[sliceId] += (uint64_t)availableRb * availableSym;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Telemetry collection — called every step_interval seconds
// ──────────────────────────────────────────────────────────────────────────────

void
SliceGymEnv::CollectTelemetry()
{
    NS_LOG_FUNCTION(this << Simulator::Now().GetSeconds());

    double now = Simulator::Now().GetSeconds();
    double dt = now - m_prevTime;

    // Reset KPIs
    for (auto& k : m_kpis)
        k = SliceKPI{};

    if (m_flowMonitor && dt > 0.0)
    {
        m_flowMonitor->CheckForLostPackets();
        const auto& stats = m_flowMonitor->GetFlowStats();

        std::array<double, NUM_SLICES> totalDelay{};
        std::array<uint32_t, NUM_SLICES> delayCount{};
        std::array<uint32_t, NUM_SLICES> activeFlows{};

        for (const auto& [fid, fs] : stats)
        {
            Ipv4FlowClassifier::FiveTuple t = m_classifier->FindFlow(fid);

            // Classify flow to slice by port
            int8_t sliceId = ClassifyFlowToSlice(t.sourcePort, t.destinationPort);
            if (sliceId < 0)
                continue;

            // Retrieve previous stats for delta computation
            uint64_t prevRxBytes = 0;
            uint32_t prevRxPkts = 0;
            uint32_t prevLostPkts = 0;
            uint32_t prevTxPkts = 0;
            double prevDelay = 0.0;

            if (m_prevStats.count(fid))
            {
                const auto& p = m_prevStats.at(fid);
                prevRxBytes = p.rxBytes;
                prevRxPkts = p.rxPackets;
                prevLostPkts = p.lostPackets;
                prevTxPkts = p.txPackets;
                prevDelay = p.delaySum.GetSeconds();
            }

            // Delta computation
            uint64_t deltaBytes = fs.rxBytes - prevRxBytes;
            double deltaBits = static_cast<double>(deltaBytes) * 8.0;

            m_kpis[sliceId].throughputMbps += (deltaBits / dt) / 1e6;
            m_kpis[sliceId].rxPackets += fs.rxPackets - prevRxPkts;
            m_kpis[sliceId].lostPackets += fs.lostPackets - prevLostPkts;
            m_kpis[sliceId].txPackets += fs.txPackets - prevTxPkts;

            if (deltaBytes > 0)
            {
                activeFlows[sliceId]++;
            }

            uint32_t rxDelta = fs.rxPackets - prevRxPkts;
            if (rxDelta > 0)
            {
                double delayDelta = fs.delaySum.GetSeconds() - prevDelay;
                totalDelay[sliceId] += delayDelta;
                delayCount[sliceId] += rxDelta;
            }
        }

        // Average delay and throughput per slice
        for (uint8_t s = 0; s < NUM_SLICES; ++s)
        {
            if (delayCount[s] > 0)
                m_kpis[s].avgDelayMs = (totalDelay[s] / delayCount[s]) * 1e3;

            if (activeFlows[s] > 0)
                m_kpis[s].throughputMbps /= activeFlows[s]; // Convert aggregate to per-UE average
        }

        // Packet drop rate per slice
        for (uint8_t s = 0; s < NUM_SLICES; ++s)
        {
            uint32_t totalPkts = m_kpis[s].rxPackets + m_kpis[s].lostPackets;
            if (totalPkts > 0)
            {
                m_kpis[s].pktDropRate = static_cast<double>(m_kpis[s].lostPackets)
                                        / static_cast<double>(totalPkts);
            }
            else
            {
                m_kpis[s].pktDropRate = 0.0;
            }
        }

        m_prevStats = stats;
        m_prevTime = now;
    }

    // PRB utilisation from MAC/PHY trace sinks
    for (uint8_t s = 0; s < NUM_SLICES; ++s)
    {
        if (m_availablePrbReg[s] > 0)
        {
            m_kpis[s].prbUtilization =
                static_cast<double>(m_usedPrbReg[s]) / m_availablePrbReg[s];
        }
        else
        {
            m_kpis[s].prbUtilization = 0.0;
        }
        // Reset counters for next interval
        m_usedPrbReg[s] = 0;
        m_availablePrbReg[s] = 0;
    }

    // Log KPIs
    for (uint8_t s = 0; s < NUM_SLICES; ++s)
    {
        static const char* names[] = {"URLLC", "eMBB ", "mMTC "};
        NS_LOG_INFO("[" << names[s] << "] "
                        << "tput=" << std::fixed << std::setprecision(2)
                        << m_kpis[s].throughputMbps << " Mbps  "
                        << "delay=" << m_kpis[s].avgDelayMs << " ms  "
                        << "prb_util=" << m_kpis[s].prbUtilization << "  "
                        << "drop=" << std::setprecision(4) << m_kpis[s].pktDropRate);
    }

    // Notify OpenGym interface (sends observation to Python, waits for action)
    Notify();

    if (!m_done)
        ScheduleNextStateRead();
}

// ──────────────────────────────────────────────────────────────────────────────
// Flow-to-slice classification (port-based for single-gNB design)
// ──────────────────────────────────────────────────────────────────────────────

int8_t
SliceGymEnv::ClassifyFlowToSlice(uint16_t srcPort, uint16_t dstPort) const
{
    for (uint8_t s = 0; s < NUM_SLICES; ++s)
    {
        uint16_t pStart = m_slicePorts[s].start;
        uint16_t pEnd = m_slicePorts[s].end;
        if (pStart == 0 && pEnd == 0)
            continue;

        // Check if either source or destination port falls in slice range
        if ((dstPort >= pStart && dstPort <= pEnd) ||
            (srcPort >= pStart && srcPort <= pEnd))
        {
            return static_cast<int8_t>(s);
        }
    }
    return -1; // Unclassified
}

// ──────────────────────────────────────────────────────────────────────────────
float SliceGymEnv::GetReward()
{
    // Asymmetric SLA weighting (Paper 3 O-RAN SLA TMC 2025)
    // Assigned based on slice priority: uRLLC (5.0), eMBB (2.0), mMTC (1.0)
    constexpr double w_sla[NUM_SLICES] = {5.0, 2.0, 1.0};
    double totalReward = 0.0;
    double sliceContribs[NUM_SLICES];

    for (uint8_t s = 0; s < NUM_SLICES; ++s)
    {
        const auto& kpi_s = m_kpis[s];
        const auto& sla_s = m_sla[s];
        double sliceViol = 0.0;

        // 1. Throughput violation
        if (sla_s.minThroughputMbps > 0 &&
            kpi_s.throughputMbps < sla_s.minThroughputMbps)
        {
            sliceViol += (sla_s.minThroughputMbps - kpi_s.throughputMbps)
                        / sla_s.minThroughputMbps;
        }

        // 2. Delay violation
        if (kpi_s.avgDelayMs > sla_s.maxDelayMs && kpi_s.avgDelayMs > 0)
        {
            sliceViol += (kpi_s.avgDelayMs - sla_s.maxDelayMs) / sla_s.maxDelayMs;
        }

        // 3. PRB utilisation violation
        if (kpi_s.prbUtilization > sla_s.maxPrbUtil)
        {
            sliceViol += (kpi_s.prbUtilization - sla_s.maxPrbUtil)
                        / (1.0 - sla_s.maxPrbUtil + 1e-6);
        }

        // 4. Packet drop penalty
        if (kpi_s.pktDropRate > 0.0)
        {
            sliceViol += kpi_s.pktDropRate;
        }

        double compliance = std::max(0.0, 1.0 - sliceViol);
        double contrib = w_sla[s] * std::log1p(compliance);
        sliceContribs[s] = contrib;
        totalReward += contrib;
    }

    NS_LOG_INFO("Step " << m_stepCount << "  reward=" << std::fixed << std::setprecision(4) << totalReward
                        << "  [urllc=" << sliceContribs[SLICE_URLLC]
                        << " embb=" << sliceContribs[SLICE_EMBB]
                        << " mmtc=" << sliceContribs[SLICE_MMTC] << "]");

    return static_cast<float>(totalReward);
}

// ──────────────────────────────────────────────────────────────────────────────
// OpenGymEnv interface
// ──────────────────────────────────────────────────────────────────────────────

Ptr<OpenGymSpace>
SliceGymEnv::GetObservationSpace()
{
    std::vector<uint32_t> shape = {OBS_SIZE}; // 12 floats

    float low[OBS_SIZE];
    float high[OBS_SIZE];

    for (uint32_t i = 0; i < NUM_SLICES; ++i)
    {
        low[i * 4 + 0]  = 0.0f;
        high[i * 4 + 0] = 1000.0f;  // throughput [Mbps]
        low[i * 4 + 1]  = 0.0f;
        high[i * 4 + 1] = 1000.0f;  // delay [ms]
        low[i * 4 + 2]  = 0.0f;
        high[i * 4 + 2] = 1.0f;     // prb_util [0,1]
        low[i * 4 + 3]  = 0.0f;
        high[i * 4 + 3] = 1.0f;     // pkt_drop_rate [0,1]
    }

    Ptr<OpenGymBoxSpace> space =
        CreateObject<OpenGymBoxSpace>(std::vector<float>(low, low + OBS_SIZE),
                                      std::vector<float>(high, high + OBS_SIZE),
                                      shape,
                                      TypeNameGet<float>());
    return space;
}

Ptr<OpenGymSpace>
SliceGymEnv::GetActionSpace()
{
    std::vector<uint32_t> shape = {NUM_SLICES};
    std::vector<float> low(NUM_SLICES, 0.0f);
    std::vector<float> high(NUM_SLICES, 1.0f);

    Ptr<OpenGymBoxSpace> space =
        CreateObject<OpenGymBoxSpace>(low, high, shape, TypeNameGet<float>());
    return space;
}

Ptr<OpenGymDataContainer>
SliceGymEnv::GetObservation()
{
    std::vector<uint32_t> shape = {OBS_SIZE};
    Ptr<OpenGymBoxContainer<float>> obs =
        CreateObject<OpenGymBoxContainer<float>>(shape);

    for (uint8_t s = 0; s < NUM_SLICES; ++s)
    {
        obs->AddValue(static_cast<float>(m_kpis[s].throughputMbps));
        obs->AddValue(static_cast<float>(m_kpis[s].avgDelayMs));
        obs->AddValue(static_cast<float>(m_kpis[s].prbUtilization));
        obs->AddValue(static_cast<float>(m_kpis[s].pktDropRate));
    }

    return obs;
}

bool
SliceGymEnv::GetGameOver()
{
    m_done = (++m_stepCount >= m_maxSteps);
    return m_done;
}

std::string
SliceGymEnv::GetExtraInfo()
{
    std::ostringstream oss;
    static const char* names[] = {"URLLC", "eMBB", "mMTC"};

    oss << "step=" << m_stepCount << " | ";
    for (uint8_t s = 0; s < NUM_SLICES; ++s)
    {
        oss << names[s]
            << "[tput=" << std::fixed << std::setprecision(2)
            << m_kpis[s].throughputMbps << "Mbps"
            << ",dly=" << m_kpis[s].avgDelayMs << "ms"
            << ",prb=" << std::setprecision(3) << m_kpis[s].prbUtilization
            << ",drop=" << std::setprecision(4) << m_kpis[s].pktDropRate
            << ",w=" << m_currentWeights[s] << "] ";
    }
    return oss.str();
}

bool
SliceGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
    Ptr<OpenGymBoxContainer<float>> box =
        DynamicCast<OpenGymBoxContainer<float>>(action);

    if (!box)
    {
        NS_LOG_ERROR("Action container is null or wrong type");
        return false;
    }

    std::vector<float> vals = box->GetData();
    if (vals.size() < NUM_SLICES)
    {
        NS_LOG_ERROR("Action vector too short: " << vals.size());
        return false;
    }

    // Enforce 0.1 minimum weight floor to prevent starvation
    double w0 = std::max(0.1, static_cast<double>(vals[0]));
    double w1 = std::max(0.1, static_cast<double>(vals[1]));
    double w2 = std::max(0.1, static_cast<double>(vals[2]));

    // Normalise
    double sum = w0 + w1 + w2;
    w0 /= sum;
    w1 /= sum;
    w2 /= sum;

    m_currentWeights[SLICE_URLLC] = w0;
    m_currentWeights[SLICE_EMBB]  = w1;
    m_currentWeights[SLICE_MMTC]  = w2;

    NS_LOG_INFO("New weights  URLLC=" << w0 << "  eMBB=" << w1 << "  mMTC=" << w2);

    if (m_weightSetterCb)
        m_weightSetterCb(w0, w1, w2);

    return true;
}

// ──────────────────────────────────────────────────────────────────────────────
const SliceKPI&
SliceGymEnv::GetKPI(uint8_t sliceId) const
{
    NS_ASSERT_MSG(sliceId < NUM_SLICES, "Slice ID out of range");
    return m_kpis[sliceId];
}

} // namespace ns3
