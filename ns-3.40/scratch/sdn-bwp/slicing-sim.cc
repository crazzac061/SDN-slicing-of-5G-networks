/* =============================================================================
 * slicing-sim.cc
 * 5G Network Slicing Simulation — Single gNB, 3 BWPs, Mixed Numerologies
 *
 * Architecture:
 *   - 1 gNB with 3 Bandwidth Parts (BWPs), one per slice
 *   - BWP 0 (uRLLC): Numerology 2 (60 kHz SCS), 100 MHz @ 6 GHz
 *   - BWP 1 (eMBB):  Numerology 1 (30 kHz SCS), 100 MHz @ 3.5 GHz
 *   - BWP 2 (mMTC):  Numerology 0 (15 kHz SCS), 10 MHz @ 700 MHz
 *   - O-RAN Near-RT RIC: Python agent via ns3-gym (BWP Manager Algorithm)
 *   - gNB Pre-processor + BWP Multiplexer in C++ MAC scheduler
 *
 * Traffic:
 *   - uRLLC: UDP CBR, 512 B every 1 ms (robotic-arm control)
 *   - eMBB:  PersistentBulkSender TCP (video streaming / file download)
 *   - mMTC:  UDP CBR, 64 B every 100 ms (IoT sensors)
 *
 * ns-3.40 + 5G-LENA (CTTC) + ns3-gym
 * ============================================================================= */

#include "nr-mac-scheduler-weighted-pf.h"
#include "slice-gym-env.h"

#include "ns3/antenna-module.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/opengym-module.h"

// ns-3 core
#include "ns3/applications-module.h"
#include "ns3/config-store.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

// ns-3 NR (5G-LENA)
#include "ns3/nr-module.h"
#include "ns3/traffic-control-module.h"

#include <iomanip>
#include <memory>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("SlicingSimulation");

// ──────────────────────────────────────────────────────────────────────────────
// Global state for scheduler weight application and NetAnim
// ──────────────────────────────────────────────────────────────────────────────
static Ptr<NrMacSchedulerWeightedPF> g_scheduler[3];
static AnimationInterface* g_anim = nullptr;
static uint32_t g_gnbId = 0;
static uint32_t g_urllcResId, g_embbResId, g_mmtcResId;

// ──────────────────────────────────────────────────────────────────────────────
// Annotate NetAnim with current slice weights on the gNB node
// ──────────────────────────────────────────────────────────────────────────────
void
AnnotateWeights(double wUrllc, double wEmbb, double wMmtc)
{
    if (!g_anim)
        return;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << "W_URLLC=" << wUrllc << " W_eMBB=" << wEmbb
        << " W_mMTC=" << wMmtc;
    g_anim->UpdateNodeDescription(g_gnbId, oss.str());
}

// ──────────────────────────────────────────────────────────────────────────────
// Callback from SliceGymEnv::ExecuteActions → applies normalised weights
// Weight SCALE=3 so that equal split (0.333) maps to weight=1.0 per scheduler
// ──────────────────────────────────────────────────────────────────────────────
void
ApplyWeights(double wUrllc, double wEmbb, double wMmtc)
{
    constexpr double SCALE = 3.0;

    if (g_scheduler[SLICE_URLLC])
        g_scheduler[SLICE_URLLC]->SetWeight(wUrllc * SCALE);
    if (g_scheduler[SLICE_EMBB])
        g_scheduler[SLICE_EMBB]->SetWeight(wEmbb * SCALE);
    if (g_scheduler[SLICE_MMTC])
        g_scheduler[SLICE_MMTC]->SetWeight(wMmtc * SCALE);

    AnnotateWeights(wUrllc, wEmbb, wMmtc);
}

// ──────────────────────────────────────────────────────────────────────────────
// Periodically update NetAnim resource counters (Throughput per UE)
// ──────────────────────────────────────────────────────────────────────────────
void
UpdateAnimCounters(Ptr<SliceGymEnv> gymEnv,
                   NodeContainer urllcUes,
                   NodeContainer embbUes,
                   NodeContainer mmtcUes)
{
    if (!g_anim)
        return;

    auto update = [&](uint32_t resId, NodeContainer& nodes, double totalTput) {
        double perUe = totalTput / std::max(1u, nodes.GetN());
        for (uint32_t i = 0; i < nodes.GetN(); ++i)
            g_anim->UpdateNodeCounter(resId, nodes.Get(i)->GetId(), perUe);
    };

    update(g_urllcResId, urllcUes, gymEnv->GetKPI(SLICE_URLLC).throughputMbps);
    update(g_embbResId, embbUes, gymEnv->GetKPI(SLICE_EMBB).throughputMbps);
    update(g_mmtcResId, mmtcUes, gymEnv->GetKPI(SLICE_MMTC).throughputMbps);

    Simulator::Schedule(Seconds(0.1), &UpdateAnimCounters, gymEnv, urllcUes, embbUes, mmtcUes);
}

// ──────────────────────────────────────────────────────────────────────────────
// Application factory: UDP CBR traffic
// ──────────────────────────────────────────────────────────────────────────────
ApplicationContainer
InstallUdpCbr(NodeContainer ueNodes,
              Ptr<Node> remoteHost,
              Ipv4InterfaceContainer& ueIpIface,
              uint16_t basePort,
              double packetSizeBytes,
              double intervalSec,
              double startTime,
              double stopTime)
{
    ApplicationContainer serverApps, clientApps;

    UdpClientHelper udpClient;
    udpClient.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
    udpClient.SetAttribute("Interval", TimeValue(Seconds(intervalSec)));
    udpClient.SetAttribute("PacketSize", UintegerValue(static_cast<uint32_t>(packetSizeBytes)));

    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        uint16_t port = basePort + i;

        UdpServerHelper udpServer(port);
        ApplicationContainer sa = udpServer.Install(ueNodes.Get(i));
        sa.Start(Seconds(startTime));
        sa.Stop(Seconds(stopTime));
        serverApps.Add(sa);

        udpClient.SetAttribute("RemoteAddress",
                               AddressValue(InetSocketAddress(ueIpIface.GetAddress(i), port)));
        ApplicationContainer ca = udpClient.Install(remoteHost);
        ca.Start(Seconds(startTime + 0.01 * i));
        ca.Stop(Seconds(stopTime));
        clientApps.Add(ca);
    }
    return clientApps;
}

// ──────────────────────────────────────────────────────────────────────────────
// PersistentBulkSender — reconnects automatically on socket close
// Mimics real eMBB behaviour (e.g. video streaming, file downloads)
// ──────────────────────────────────────────────────────────────────────────────
class PersistentBulkSender : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::PersistentBulkSender")
                                .SetParent<Application>()
                                .SetGroupName("Applications")
                                .AddConstructor<PersistentBulkSender>();
        return tid;
    }

    PersistentBulkSender()
    {
    }

    virtual ~PersistentBulkSender()
    {
    }

    void Setup(Address remoteAddr, uint32_t packetSize = 1400)
    {
        m_remoteAddr = remoteAddr;
        m_packetSize = packetSize;
    }

  private:
    virtual void StartApplication() override
    {
        m_running = true;
        Connect();
    }

    virtual void StopApplication() override
    {
        m_running = false;
        if (m_socket)
        {
            m_socket->Close();
            m_socket->SetConnectCallback(MakeNullCallback<void, Ptr<Socket>>(),
                                         MakeNullCallback<void, Ptr<Socket>>());
            m_socket->SetCloseCallbacks(MakeNullCallback<void, Ptr<Socket>>(),
                                        MakeNullCallback<void, Ptr<Socket>>());
        }
        m_socket = nullptr;
    }

    void Connect()
    {
        if (!m_running)
            return;
        NS_LOG_INFO("eMBB TCP: attempting connection to " << m_remoteAddr);
        m_socket = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
        m_socket->SetConnectCallback(MakeCallback(&PersistentBulkSender::OnConnected, this),
                                     MakeCallback(&PersistentBulkSender::OnConnectFailed, this));
        m_socket->SetCloseCallbacks(MakeCallback(&PersistentBulkSender::OnClosed, this),
                                    MakeCallback(&PersistentBulkSender::OnClosed, this));
        m_socket->Connect(m_remoteAddr);
    }

    void OnConnected(Ptr<Socket> socket)
    {
        NS_LOG_INFO("eMBB TCP connected at t=" << Simulator::Now().GetSeconds());
        SendData();
    }

    void OnConnectFailed(Ptr<Socket> socket)
    {
        NS_LOG_WARN("eMBB TCP connect failed — retrying in 200ms");
        if (m_running)
            Simulator::Schedule(MilliSeconds(200), &PersistentBulkSender::Connect, this);
    }

    void OnClosed(Ptr<Socket> socket)
    {
        NS_LOG_WARN("eMBB TCP socket closed — reconnecting in 500ms");
        m_socket = nullptr;
        if (m_running)
            Simulator::Schedule(MilliSeconds(500), &PersistentBulkSender::Connect, this);
    }

    void SendData()
    {
        if (!m_socket || !m_running)
            return;
        // Saturate the link by filling the TX buffer
        while (m_socket->GetTxAvailable() > m_packetSize)
        {
            Ptr<Packet> pkt = Create<Packet>(m_packetSize);
            int sent = m_socket->Send(pkt);
            if (sent < 0)
                break;
        }
        // Reschedule to check buffer availability
        Simulator::Schedule(MilliSeconds(5), &PersistentBulkSender::SendData, this);
    }

    Ptr<Socket> m_socket{nullptr};
    Address m_remoteAddr;
    uint32_t m_packetSize{1400};
    bool m_running{false};
};

NS_OBJECT_ENSURE_REGISTERED(PersistentBulkSender);

// =============================================================================
// MAIN
// =============================================================================

int
main(int argc, char* argv[])
{
    // ─────────────────────────────────────────────────────────────────────────
    // 1. CLI parameters
    // ─────────────────────────────────────────────────────────────────────────
    uint32_t gymPort = 5555;
    double simTime = 50.0;
    Packet::EnablePrinting();
    double stepInterval = 0.1;
    uint32_t nUrllc = 5;
    uint32_t nEmbb = 10;
    uint32_t nMmtc = 30;
    bool verbose = false;
    bool pcapEnabled = false;

    CommandLine cmd(__FILE__);
    cmd.AddValue("gymPort", "OpenGym TCP port", gymPort);
    cmd.AddValue("simTime", "Simulation time [s]", simTime);
    cmd.AddValue("stepInterval", "Gym step interval [s]", stepInterval);
    cmd.AddValue("nUrllc", "Number of URLLC UEs", nUrllc);
    cmd.AddValue("nEmbb", "Number of eMBB UEs", nEmbb);
    cmd.AddValue("nMmtc", "Number of mMTC UEs", nMmtc);
    cmd.AddValue("verbose", "Enable verbose logging", verbose);
    cmd.AddValue("pcap", "Enable PCAP traces", pcapEnabled);
    cmd.Parse(argc, argv);

    // Increase TCP window sizes so PRBs aren't starved by TCP buffering
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(1 << 23)); // 8 MB
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(1 << 23)); // 8 MB
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(500));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1400));
    Config::SetDefault("ns3::TcpSocketBase::WindowScaling", BooleanValue(true));

    if (verbose)
    {
        LogComponentEnable("SlicingSimulation", LOG_LEVEL_INFO);
        LogComponentEnable("SliceGymEnv", LOG_LEVEL_INFO);
        LogComponentEnable("NrMacSchedulerWeightedPF", LOG_LEVEL_INFO);
        LogComponentEnable("FqCoDelQueueDisc", LOG_LEVEL_INFO);
    }

    NS_LOG_INFO("=== 5G Network Slicing Simulation (Single gNB, 3 BWPs) ===");
    NS_LOG_INFO("UEs: URLLC=" << nUrllc << " eMBB=" << nEmbb << " mMTC=" << nMmtc);
    NS_LOG_INFO("simTime=" << simTime << "s  step=" << stepInterval << "s");

    // Buffer setting: set to a small finite value (20 KB) to trigger drops during congestion
    Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue(20 * 1024));
    Config::SetDefault("ns3::LteEnbRrc::EpsBearerToRlcMapping", StringValue("RlcUmAlways"));
    Config::SetDefault("ns3::LteEnbRrc::SrsPeriodicity", UintegerValue(320));

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Create nodes — Single gNB + all UEs
    // ─────────────────────────────────────────────────────────────────────────
    NodeContainer gnbNodes;
    gnbNodes.Create(1); // SINGLE gNB

    NodeContainer urllcUes;
    urllcUes.Create(nUrllc);
    NodeContainer embbUes;
    embbUes.Create(nEmbb);
    NodeContainer mmtcUes;
    mmtcUes.Create(nMmtc);

    NodeContainer allUes;
    allUes.Add(urllcUes);
    allUes.Add(embbUes);
    allUes.Add(mmtcUes);

    // ─────────────────────────────────────────────────────────────────────────
    // 3. Mobility
    // ─────────────────────────────────────────────────────────────────────────
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    // gNB: fixed at center
    Ptr<ListPositionAllocator> gnbPos = CreateObject<ListPositionAllocator>();
    gnbPos->Add(Vector(200.0, 200.0, 10.0));
    mobility.SetPositionAllocator(gnbPos);
    mobility.Install(gnbNodes);

    // uRLLC UEs: random scatter + walking
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                              "Bounds",
                              RectangleValue(Rectangle(0, 400, 0, 400)),
                              "Distance",
                              DoubleValue(2.0),
                              "Speed",
                              StringValue("ns3::UniformRandomVariable[Min=1.0|Max=5.0]"));
    Ptr<RandomBoxPositionAllocator> uePos = CreateObject<RandomBoxPositionAllocator>();
    uePos->SetAttribute("X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=400.0]"));
    uePos->SetAttribute("Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=400.0]"));
    uePos->SetAttribute("Z", StringValue("ns3::UniformRandomVariable[Min=1.5|Max=1.5]"));
    mobility.SetPositionAllocator(uePos);
    mobility.Install(urllcUes);
    mobility.Install(mmtcUes);

    // eMBB UEs: constrained to mid-cell for good CQI
    Ptr<RandomBoxPositionAllocator> embbPos = CreateObject<RandomBoxPositionAllocator>();
    embbPos->SetAttribute("X", StringValue("ns3::UniformRandomVariable[Min=150.0|Max=250.0]"));
    embbPos->SetAttribute("Y", StringValue("ns3::UniformRandomVariable[Min=150.0|Max=250.0]"));
    embbPos->SetAttribute("Z", StringValue("ns3::UniformRandomVariable[Min=1.5|Max=1.5]"));

    MobilityHelper embbMobility;
    embbMobility.SetMobilityModel("ns3::RandomWaypointMobilityModel",
                                  "Speed",
                                  StringValue("ns3::UniformRandomVariable[Min=1.0|Max=3.0]"),
                                  "Pause",
                                  StringValue("ns3::UniformRandomVariable[Min=2.0|Max=8.0]"),
                                  "PositionAllocator",
                                  PointerValue(embbPos));
    embbMobility.SetPositionAllocator(embbPos);
    embbMobility.Install(embbUes);

    // ─────────────────────────────────────────────────────────────────────────
    // 4. MEC / Local-UPF topology — Remote host co-located at the edge
    //    Eliminates the core-network delay floor so uRLLC can hit <1 ms.
    //    REMINDER: update agent.py launch args:
    //       --core-delay-offset 0.0   (was 2.5)
    // ─────────────────────────────────────────────────────────────────────────
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);

    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
    epcHelper->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));
    Ptr<Node> pgw = epcHelper->GetPgwNode();

    // MEC link: application server sits at the edge — zero propagation delay
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", StringValue("100Gbps"));
    p2ph.SetChannelAttribute("Delay", StringValue("0ms"));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(1500));
    p2ph.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", StringValue("10000p"));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);

    // Position PGW and RemoteHost
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(pgw);
    pgw->GetObject<MobilityModel>()->SetPosition(Vector(200.0, 60.0, 0.0));
    mobility.Install(remoteHost);
    // Co-locate MEC application server right next to PGW
    remoteHost->GetObject<MobilityModel>()->SetPosition(Vector(200.0, 62.0, 0.0));

    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    // ─────────────────────────────────────────────────────────────────────────
    // 5. NR Helper + 3 Operation Bands (one CC, one BWP each)
    //
    //    BWP 0 — uRLLC: 100 MHz @  6.0 GHz  (FR1-High, UMa)
    //    BWP 1 — eMBB:  100 MHz @  3.5 GHz  (sub-6, UMa)
    //    BWP 2 — mMTC:  10 MHz  @  0.7 GHz  (sub-GHz, RMa)
    // ─────────────────────────────────────────────────────────────────────────
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    nrHelper->SetEpcHelper(epcHelper);

    CcBwpCreator ccBwpCreator;

    // --- uRLLC Band (BWP 0, CC 0) ---
    CcBwpCreator::SimpleOperationBandConf urllcBandConf(6.0e9, 100e6, 1, BandwidthPartInfo::UMa);
    OperationBandInfo urllcBand = ccBwpCreator.CreateOperationBandContiguousCc(urllcBandConf);

    // --- eMBB Band (BWP 1, CC 1) ---
    CcBwpCreator::SimpleOperationBandConf embbBandConf(3.5e9, 100e6, 1, BandwidthPartInfo::UMa);
    OperationBandInfo embbBand = ccBwpCreator.CreateOperationBandContiguousCc(embbBandConf);

    // Fix IDs for eMBB
    auto ccPtr1 = std::move(embbBand.m_cc.at(0));
    embbBand.m_cc.erase(embbBand.m_cc.begin());
    ccPtr1->m_ccId = 1;
    auto bwpPtr1 = std::move(ccPtr1->m_bwp.at(0));
    ccPtr1->m_bwp.erase(ccPtr1->m_bwp.begin());
    bwpPtr1->m_bwpId = 1;
    ccPtr1->m_bwp.push_back(std::move(bwpPtr1));
    embbBand.m_cc.push_back(std::move(ccPtr1));

    // --- mMTC Band (BWP 2, CC 2) ---
    CcBwpCreator::SimpleOperationBandConf mmtcBandConf(700e6, 10e6, 1, BandwidthPartInfo::RMa);
    OperationBandInfo mmtcBand = ccBwpCreator.CreateOperationBandContiguousCc(mmtcBandConf);

    // Fix IDs for mMTC
    auto ccPtr2 = std::move(mmtcBand.m_cc.at(0));
    mmtcBand.m_cc.erase(mmtcBand.m_cc.begin());
    ccPtr2->m_ccId = 2;
    auto bwpPtr2 = std::move(ccPtr2->m_bwp.at(0));
    ccPtr2->m_bwp.erase(ccPtr2->m_bwp.begin());
    bwpPtr2->m_bwpId = 2;
    ccPtr2->m_bwp.push_back(std::move(bwpPtr2));
    mmtcBand.m_cc.push_back(std::move(ccPtr2));

    // ─────────────────────────────────────────────────────────────────────────
    // 5.1 Configure Antennas and Beamforming (REQUIRED for 5G-LENA PHY)
    // ─────────────────────────────────────────────────────────────────────────
    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(4));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));

    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(8));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<IsotropicAntennaModel>()));

    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
    idealBeamformingHelper->SetAttribute("BeamformingMethod",
                                         TypeIdValue(DirectPathBeamforming::GetTypeId()));
    nrHelper->SetBeamformingHelper(idealBeamformingHelper);

    // Initialize all operation bands with unique IDs
    nrHelper->InitializeOperationBand(&urllcBand);
    nrHelper->InitializeOperationBand(&embbBand);
    nrHelper->InitializeOperationBand(&mmtcBand);

    // Combine ALL BWPs into a single vector for the single gNB
    BandwidthPartInfoPtrVector allBwps;
    {
        auto bwp0 = CcBwpCreator::GetAllBwps({urllcBand});
        auto bwp1 = CcBwpCreator::GetAllBwps({embbBand});
        auto bwp2 = CcBwpCreator::GetAllBwps({mmtcBand});
        allBwps.insert(allBwps.end(), bwp0.begin(), bwp0.end());
        allBwps.insert(allBwps.end(), bwp1.begin(), bwp1.end());
        allBwps.insert(allBwps.end(), bwp2.begin(), bwp2.end());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 6. BWP Manager: route QCI → BWP ID
    //    This is the NR-level traffic-to-BWP routing (before it reaches MAC)
    // ─────────────────────────────────────────────────────────────────────────
    nrHelper->SetGnbBwpManagerAlgorithmAttribute("GBR_CONV_VOICE",
                                                 UintegerValue(0)); // uRLLC → BWP 0
    nrHelper->SetGnbBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB",
                                                 UintegerValue(1)); // eMBB → BWP 1
    nrHelper->SetGnbBwpManagerAlgorithmAttribute("NGBR_VIDEO_TCP_DEFAULT",
                                                 UintegerValue(2)); // mMTC → BWP 2

    // Set UE side BWP routing to match
    nrHelper->SetUeBwpManagerAlgorithmAttribute("GBR_CONV_VOICE", UintegerValue(0));
    nrHelper->SetUeBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB", UintegerValue(1));
    nrHelper->SetUeBwpManagerAlgorithmAttribute("NGBR_VIDEO_TCP_DEFAULT", UintegerValue(2));

    // ─────────────────────────────────────────────────────────────────────────
    // 7. Install gNB + UE devices with our custom weighted PF scheduler
    // ─────────────────────────────────────────────────────────────────────────
    nrHelper->SetSchedulerTypeId(NrMacSchedulerWeightedPF::GetTypeId());
    nrHelper->SetSchedulerAttribute("SliceWeight", DoubleValue(1.0));

    // Install SINGLE gNB device with ALL 3 BWPs
    NetDeviceContainer gnbDev = nrHelper->InstallGnbDevice(gnbNodes, allBwps);

    // Apply SrsPeriodicity DIRECTLY to the gNB RRC to ensure it takes effect
    for (uint32_t i = 0; i < gnbDev.GetN(); ++i)
    {
        Ptr<NrGnbNetDevice> gnb = DynamicCast<NrGnbNetDevice>(gnbDev.Get(i));
        UintegerValue val;
        gnb->GetRrc()->GetAttribute("SrsPeriodicity", val);
        std::cout << "DEBUG: GNB " << i << " SrsPeriodicity is " << val.Get() << std::endl;
        gnb->GetRrc()->SetAttribute("SrsPeriodicity", UintegerValue(320));
        gnb->GetRrc()->GetAttribute("SrsPeriodicity", val);
        std::cout << "DEBUG: GNB " << i << " SrsPeriodicity NOW is " << val.Get() << std::endl;
    }

    // Install ALL UEs with ALL 3 BWPs (UE can attach to any BWP)
    NetDeviceContainer ueDev = nrHelper->InstallUeDevice(allUes, allBwps);

    // ─────────────────────────────────────────────────────────────────────────
    // 8. Post-install per-BWP configuration: mixed numerologies + TxPower
    //    BWP 0 (uRLLC): Numerology 3 (120 kHz SCS) — 0.125 ms slots for <1 ms
    //    BWP 1 (eMBB):  Numerology 1 (30 kHz SCS)
    //    BWP 2 (mMTC):  Numerology 0 (15 kHz SCS)
    // ─────────────────────────────────────────────────────────────────────────
    // uRLLC BWP (index 0): Numerology 3 = 120 kHz SCS
    Ptr<NrGnbPhy> urllcPhy = nrHelper->GetGnbPhy(gnbDev.Get(0), 0);
    urllcPhy->SetAttribute("Numerology", UintegerValue(3));
    urllcPhy->SetAttribute("TxPower", DoubleValue(43.0));
    urllcPhy->SetAttribute("Pattern", StringValue("F|F|F|F|F|F|F|F|F|F|"));
    // Latency Surgery: Zero-out processing delays for uRLLC
    urllcPhy->SetAttribute("N0Delay", UintegerValue(0));
    urllcPhy->SetAttribute("N1Delay", UintegerValue(0));
    urllcPhy->SetAttribute("N2Delay", UintegerValue(0));
    urllcPhy->SetAttribute("TbDecodeLatency", TimeValue(MilliSeconds(0)));

    // eMBB BWP (index 1): Numerology 1 = 30 kHz SCS
    nrHelper->GetGnbPhy(gnbDev.Get(0), 1)->SetAttribute("Numerology", UintegerValue(1));
    nrHelper->GetGnbPhy(gnbDev.Get(0), 1)->SetAttribute("TxPower", DoubleValue(43.0));
    nrHelper->GetGnbPhy(gnbDev.Get(0), 1)
        ->SetAttribute("Pattern", StringValue("F|F|F|F|F|F|F|F|F|F|"));

    // mMTC BWP (index 2): Numerology 0 = 15 kHz SCS
    nrHelper->GetGnbPhy(gnbDev.Get(0), 2)->SetAttribute("Numerology", UintegerValue(0));
    nrHelper->GetGnbPhy(gnbDev.Get(0), 2)->SetAttribute("TxPower", DoubleValue(43.0));
    nrHelper->GetGnbPhy(gnbDev.Get(0), 2)
        ->SetAttribute("Pattern", StringValue("F|F|F|F|F|F|F|F|F|F|"));

    // Update device configs after attribute changes
    for (auto it = gnbDev.Begin(); it != gnbDev.End(); ++it)
    {
        DynamicCast<NrGnbNetDevice>(*it)->UpdateConfig();
    }
    for (auto it = ueDev.Begin(); it != ueDev.End(); ++it)
    {
        Ptr<NrUeNetDevice> nrUeDev = DynamicCast<NrUeNetDevice>(*it);
        // Latency Surgery (Part 2): Zero-out decoding delay for all BWPs on this UE
        for (uint32_t bwp = 0; bwp < 3; ++bwp)
        {
            Ptr<NrUePhy> uePhy = nrUeDev->GetPhy(bwp);
            if (uePhy)
            {
                // N0/N1/N2 are GNB-side; for UE we only zero-out TB Decode Latency
                uePhy->SetAttribute("TbDecodeLatency", TimeValue(MilliSeconds(0)));
            }
        }
        nrUeDev->UpdateConfig();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 9. Retrieve scheduler pointers + set slice types for pre-processor
    // ─────────────────────────────────────────────────────────────────────────
    for (uint8_t bwpIdx = 0; bwpIdx < NUM_SLICES; ++bwpIdx)
    {
        Ptr<NrMacScheduler> sched = nrHelper->GetScheduler(gnbDev.Get(0), bwpIdx);
        g_scheduler[bwpIdx] = sched ? DynamicCast<NrMacSchedulerWeightedPF>(sched) : nullptr;

        NS_ABORT_MSG_IF(!g_scheduler[bwpIdx], "Could not retrieve scheduler for BWP " << +bwpIdx);

        // Set slice type so pre-processor applies correct algorithm
        g_scheduler[bwpIdx]->SetSliceType(bwpIdx);
    }

    NS_LOG_INFO("Retrieved 3 scheduler instances for BWPs 0-2");

    // ─────────────────────────────────────────────────────────────────────────
    // 10. Internet stack + IP addressing on UEs
    // ─────────────────────────────────────────────────────────────────────────
    internet.Install(allUes);

    // EPC assigns UE IPs from 7.0.0.0/8
    Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address(ueDev);

    // Set default route for all UEs
    for (uint32_t i = 0; i < allUes.GetN(); ++i)
    {
        Ptr<Ipv4StaticRouting> ueRouting =
            ipv4RoutingHelper.GetStaticRouting(allUes.Get(i)->GetObject<Ipv4>());
        ueRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    // 11. Attach all UEs to the single gNB
    // ─────────────────────────────────────────────────────────────────────────
    nrHelper->AttachToClosestEnb(ueDev, gnbDev);

    // Enforce Active Queue Management (AQM) to prevent UDP Bufferbloat
    auto installAqm = [&](NetDeviceContainer& devices) {
        for (uint32_t i = 0; i < devices.GetN(); ++i)
        {
            Ptr<NetDevice> dev = devices.Get(i);
            Ptr<Node> node = dev->GetNode();
            Ptr<TrafficControlLayer> tc = node->GetObject<TrafficControlLayer>();
            if (tc)
            {
                if (tc->GetRootQueueDiscOnDevice(dev))
                {
                    tc->DeleteRootQueueDiscOnDevice(dev);
                }

                // Manually create and configure FqCoDel to bypass attribute limitations
                Ptr<FqCoDelQueueDisc> qd = CreateObject<FqCoDelQueueDisc>();
                qd->SetAttribute("Target", StringValue("2ms"));
                qd->SetAttribute("Interval", StringValue("20ms"));
                qd->SetQuantum(1514); // Set directly before Initialize()

                tc->SetRootQueueDiscOnDevice(dev, qd);
                qd->Initialize();
            }
        }
    };

    installAqm(gnbDev);
    installAqm(ueDev);
    installAqm(internetDevices);

    // ─────────────────────────────────────────────────────────────────────────
    // 12. EPC bearers: route traffic to correct BWPs using EpcTft filters
    //     uRLLC → port 3000-3099 → GBR_CONV_VOICE       → BWP 0
    //     eMBB  → port 4000-4099 → NGBR_LOW_LAT_EMBB    → BWP 1
    //     mMTC  → port 5000-5099 → NGBR_VIDEO_TCP_DEFAULT → BWP 2
    // ─────────────────────────────────────────────────────────────────────────

    // --- uRLLC bearers ---
    for (uint32_t i = 0; i < urllcUes.GetN(); ++i)
    {
        uint32_t ueGlobalIdx = i; // uRLLC UEs are first in allUes
        uint16_t port = 3000 + i;

        Ptr<EpcTft> tft = Create<EpcTft>();
        EpcTft::PacketFilter pf;
        pf.localPortStart = port;
        pf.localPortEnd = port;
        tft->Add(pf);

        EpsBearer bearer(EpsBearer::GBR_CONV_VOICE);
        nrHelper->ActivateDedicatedEpsBearer(ueDev.Get(ueGlobalIdx), bearer, tft);
    }

    // --- eMBB bearers ---
    for (uint32_t i = 0; i < embbUes.GetN(); ++i)
    {
        uint32_t ueGlobalIdx = nUrllc + i;
        uint16_t port = 4000 + i;

        Ptr<EpcTft> tft = Create<EpcTft>();
        EpcTft::PacketFilter pf;
        pf.localPortStart = port;
        pf.localPortEnd = port;
        tft->Add(pf);

        EpsBearer bearer(EpsBearer::NGBR_LOW_LAT_EMBB);
        nrHelper->ActivateDedicatedEpsBearer(ueDev.Get(ueGlobalIdx), bearer, tft);
    }

    // --- mMTC bearers ---
    for (uint32_t i = 0; i < mmtcUes.GetN(); ++i)
    {
        uint32_t ueGlobalIdx = nUrllc + nEmbb + i;
        uint16_t port = 5000 + i;

        Ptr<EpcTft> tft = Create<EpcTft>();
        EpcTft::PacketFilter pf;
        pf.localPortStart = port;
        pf.localPortEnd = port;
        tft->Add(pf);

        EpsBearer bearer(EpsBearer::NGBR_VIDEO_TCP_DEFAULT);
        nrHelper->ActivateDedicatedEpsBearer(ueDev.Get(ueGlobalIdx), bearer, tft);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 13. Applications
    // ─────────────────────────────────────────────────────────────────────────
    double appStart = 1.0;
    double appStop = simTime;

    // Separate IP interface containers by extracting from global ueIpIface
    Ipv4InterfaceContainer urllcIpIface, embbIpIface, mmtcIpIface;
    for (uint32_t i = 0; i < nUrllc; ++i)
        urllcIpIface.Add(ueIpIface.Get(i));
    for (uint32_t i = 0; i < nEmbb; ++i)
        embbIpIface.Add(ueIpIface.Get(nUrllc + i));
    for (uint32_t i = 0; i < nMmtc; ++i)
        mmtcIpIface.Add(ueIpIface.Get(nUrllc + nEmbb + i));

    // uRLLC: tight-loop UDP CBR (512 B / 1 ms per UE)
    InstallUdpCbr(urllcUes, remoteHost, urllcIpIface, 3000, 512, 0.001, appStart, appStop);

    // eMBB: UDP OnOff firehose (150 Mbps per UE — saturates 100 MHz BWP)
    // Eliminates TCP bufferbloat; tests pure MAC scheduler throughput.
    for (uint32_t i = 0; i < embbUes.GetN(); ++i)
    {
        uint16_t port = 4000 + i;

        // Sink on UE side (UDP)
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApp = sink.Install(embbUes.Get(i));
        sinkApp.Start(Seconds(appStart));
        sinkApp.Stop(Seconds(appStop));

        // OnOff source on remote host (always-on UDP)
        OnOffHelper onoff("ns3::UdpSocketFactory",
                          InetSocketAddress(embbIpIface.GetAddress(i), port));
        onoff.SetAttribute("DataRate", StringValue("50Mbps"));
        onoff.SetAttribute("PacketSize", UintegerValue(1500));
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));

        ApplicationContainer senderApp = onoff.Install(remoteHost);
        senderApp.Start(Seconds(appStart + 0.01 * i));
        senderApp.Stop(Seconds(appStop));
    }

    // mMTC: periodic tiny UDP (64 B / 100 ms per UE)
    InstallUdpCbr(mmtcUes, remoteHost, mmtcIpIface, 5000, 1280, 0.01, appStart, appStop);

    // ─────────────────────────────────────────────────────────────────────────
    // 14. Flow Monitor
    // ─────────────────────────────────────────────────────────────────────────
    FlowMonitorHelper fmHelper;
    Ptr<FlowMonitor> flowMonitor = fmHelper.InstallAll();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(fmHelper.GetClassifier());

    // ─────────────────────────────────────────────────────────────────────────
    // 15. OpenGym / ns3-gym setup
    // ─────────────────────────────────────────────────────────────────────────
    Ptr<SliceGymEnv> gymEnv = CreateObject<SliceGymEnv>();
    gymEnv->SetAttribute("StepInterval", DoubleValue(stepInterval));
    gymEnv->SetAttribute("MaxSteps", UintegerValue(static_cast<uint32_t>(simTime / stepInterval)));

    gymEnv->SetFlowMonitor(flowMonitor, classifier);

    // Register port ranges for flow classification (single-gNB, port-based)
    gymEnv->RegisterSlicePorts(SLICE_URLLC, 3000, 3000 + nUrllc - 1);
    gymEnv->RegisterSlicePorts(SLICE_EMBB, 4000, 4000 + nEmbb - 1);
    gymEnv->RegisterSlicePorts(SLICE_MMTC, 5000, 5000 + nMmtc - 1);

    // SLA definitions (matching agent.py per-UE targets)
    gymEnv->SetSliceSLA(SLICE_URLLC, 1.0, 1.0, 0.85);  // 1.0 Mbps per-UE
    gymEnv->SetSliceSLA(SLICE_EMBB, 20.0, 50.0, 0.90); // 20.0 Mbps per-UE target
    gymEnv->SetSliceSLA(SLICE_MMTC, 0.1, 200.0, 0.80);

    gymEnv->SetWeightSetterCallback(ApplyWeights);

    // Connect gym env to OpenGym TCP interface
    Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface>(gymPort);
    gymEnv->SetOpenGymInterface(openGymInterface);

    gymEnv->ScheduleNextStateRead();

    // ─────────────────────────────────────────────────────────────────────────
    // 16. PRB Telemetry Trace Connections (one per BWP on the single gNB)
    // ─────────────────────────────────────────────────────────────────────────
    {
        Ptr<NrGnbNetDevice> gnb = DynamicCast<NrGnbNetDevice>(gnbDev.Get(0));
        NS_ABORT_MSG_IF(!gnb, "Could not cast gNB NetDevice");

        for (uint8_t bwpIdx = 0; bwpIdx < NUM_SLICES; ++bwpIdx)
        {
            Ptr<NrGnbPhy> phy = DynamicCast<NrGnbPhy>(gnb->GetPhy(bwpIdx));
            if (!phy)
            {
                NS_LOG_WARN("Could not get PHY for BWP " << +bwpIdx);
                continue;
            }

            phy->TraceConnectWithoutContext("SlotDataStats",
                                            MakeCallback(&SliceGymEnv::SlotStatsTraceSink, gymEnv)
                                                .Bind(static_cast<uint32_t>(bwpIdx)));
            phy->TraceConnectWithoutContext("SlotCtrlStats",
                                            MakeCallback(&SliceGymEnv::SlotStatsTraceSink, gymEnv)
                                                .Bind(static_cast<uint32_t>(bwpIdx)));
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 17. Optional PCAP
    // ─────────────────────────────────────────────────────────────────────────
    if (pcapEnabled)
        p2ph.EnablePcapAll("sdn-bwp-slice");

    // ─────────────────────────────────────────────────────────────────────────
    // 18. NetAnim Trace & Configuration
    // ─────────────────────────────────────────────────────────────────────────
    static AnimationInterface anim("sdn-bwp-slicing.xml");
    g_anim = &anim;
    g_gnbId = gnbNodes.Get(0)->GetId();

    anim.EnablePacketMetadata(true);
    anim.SetMaxPktsPerTraceFile(100000);
    anim.SetMobilityPollInterval(Seconds(0.05));

    // Register node counters
    g_urllcResId = anim.AddNodeCounter("URLLC Tput (Mbps)", AnimationInterface::DOUBLE_COUNTER);
    g_embbResId = anim.AddNodeCounter("eMBB Tput (Mbps)", AnimationInterface::DOUBLE_COUNTER);
    g_mmtcResId = anim.AddNodeCounter("mMTC Tput (Mbps)", AnimationInterface::DOUBLE_COUNTER);

    // gNB node (Blue)
    anim.UpdateNodeDescription(gnbNodes.Get(0), "gNB-0 (3BWP)");
    anim.UpdateNodeColor(gnbNodes.Get(0), 0, 0, 255);
    anim.UpdateNodeSize(gnbNodes.Get(0)->GetId(), 3.0, 3.0);

    // Remote Host & PGW (Gray)
    anim.UpdateNodeDescription(remoteHost, "Internet");
    anim.UpdateNodeColor(remoteHost, 128, 128, 128);
    anim.UpdateNodeSize(remoteHost->GetId(), 3.0, 3.0);
    anim.UpdateNodeDescription(pgw, "PGW/UPF");
    anim.UpdateNodeColor(pgw, 128, 128, 128);
    anim.UpdateNodeSize(pgw->GetId(), 2.5, 2.5);

    // UEs by slice
    for (uint32_t i = 0; i < urllcUes.GetN(); ++i)
    {
        anim.UpdateNodeDescription(urllcUes.Get(i), "U" + std::to_string(i));
        anim.UpdateNodeColor(urllcUes.Get(i), 255, 0, 0); // Red = uRLLC
        anim.UpdateNodeSize(urllcUes.Get(i)->GetId(), 2.0, 2.0);
    }
    for (uint32_t i = 0; i < embbUes.GetN(); ++i)
    {
        anim.UpdateNodeDescription(embbUes.Get(i), "E" + std::to_string(i));
        anim.UpdateNodeColor(embbUes.Get(i), 0, 200, 0); // Green = eMBB
        anim.UpdateNodeSize(embbUes.Get(i)->GetId(), 2.0, 2.0);
    }
    for (uint32_t i = 0; i < mmtcUes.GetN(); ++i)
    {
        anim.UpdateNodeDescription(mmtcUes.Get(i), "M" + std::to_string(i));
        anim.UpdateNodeColor(mmtcUes.Get(i), 255, 165, 0); // Orange = mMTC
        anim.UpdateNodeSize(mmtcUes.Get(i)->GetId(), 2.0, 2.0);
    }

    // Schedule periodic counter updates
    Simulator::Schedule(Seconds(appStart + 0.1),
                        &UpdateAnimCounters,
                        gymEnv,
                        urllcUes,
                        embbUes,
                        mmtcUes);

    // ─────────────────────────────────────────────────────────────────────────
    // 19. Run simulation
    // ─────────────────────────────────────────────────────────────────────────
    NS_LOG_INFO("Starting simulation for " << simTime << " seconds...");

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    // ─────────────────────────────────────────────────────────────────────────
    // 20. Post-simulation statistics
    // ─────────────────────────────────────────────────────────────────────────
    NS_LOG_INFO("\n===== Flow Statistics =====");
    flowMonitor->SerializeToXmlFile("sdn-bwp-flow-stats.xml", true, true);

    openGymInterface->NotifySimulationEnd();

    Simulator::Destroy();

    NS_LOG_INFO("Simulation complete.");
    return 0;
}
