/* =============================================================================
 * slice-simulation.cc
 * 5G Network Slicing Simulation — Infrastructure + Control Layer.
 * ============================================================================= */

#include "nr-mac-scheduler-weighted-pf.h"
#include "slice-gym-env.h"

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
#include "ns3/epc-ue-nas.h"
#include "ns3/lte-module.h"
#include "ns3/nr-module.h"

#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <memory>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("SliceSimulation");

static Ptr<NrMacSchedulerWeightedPF> g_scheduler[3];
static AnimationInterface* g_anim = nullptr; // Global pointer for NetAnim annotations
static uint32_t g_gnbId = 0;

// NetAnim Resource Counter IDs
static uint32_t g_urllcResId, g_embbResId, g_mmtcResId;

/**
 * Helper to annotate NetAnim with current slice weights on the gNB node.
 */
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

/**
 * Callback from SliceGymEnv::ExecuteActions → applies normalised weights.
 */
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

    // Update NetAnim annotation
    AnnotateWeights(wUrllc, wEmbb, wMmtc);
}

/**
 * Periodically update NetAnim resource counters (Throughput)
 */
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

// ---------------------------------------------------------------------------
// Application factory helpers
// ---------------------------------------------------------------------------

ApplicationContainer
InstallUdpCbr(NodeContainer ueNodes,
              Ptr<Node> remoteHost,
              Ipv4InterfaceContainer& ueIpIface,
              uint16_t port,
              double packetSizeBytes,
              double intervalSec,
              double startTime,
              double stopTime)
{
    ApplicationContainer serverApps, clientApps;

    UdpServerHelper udpServer(port);
    serverApps = udpServer.Install(ueNodes);
    serverApps.Start(Seconds(startTime));
    serverApps.Stop(Seconds(stopTime));

    UdpClientHelper udpClient;
    udpClient.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
    udpClient.SetAttribute("Interval", TimeValue(Seconds(intervalSec)));
    udpClient.SetAttribute("PacketSize", UintegerValue(static_cast<uint32_t>(packetSizeBytes)));

    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        udpClient.SetAttribute("RemoteAddress",
                               AddressValue(InetSocketAddress(ueIpIface.GetAddress(i), port)));
        ApplicationContainer ac = udpClient.Install(remoteHost);
        ac.Start(Seconds(startTime + 0.01 * i));
        ac.Stop(Seconds(stopTime));
        clientApps.Add(ac);
    }
    return clientApps;
}

ApplicationContainer
InstallTcpBulk(NodeContainer ueNodes,
               Ptr<Node> remoteHost,
               Ipv4InterfaceContainer& ueIpIface,
               uint16_t port,
               double startTime,
               double stopTime)
{
    ApplicationContainer serverApps, clientApps;

    PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    serverApps = sink.Install(ueNodes);
    serverApps.Start(Seconds(startTime));
    serverApps.Stop(Seconds(stopTime));

    BulkSendHelper bulk("ns3::TcpSocketFactory", Address());
    bulk.SetAttribute("MaxBytes", UintegerValue(0));

    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        AddressValue remoteAddr(InetSocketAddress(ueIpIface.GetAddress(i), port));
        bulk.SetAttribute("Remote", remoteAddr);
        ApplicationContainer ac = bulk.Install(remoteHost);
        ac.Start(Seconds(startTime + 0.05 * i));
        ac.Stop(Seconds(stopTime));
        clientApps.Add(ac);
    }
    return clientApps;
}

// ---------------------------------------------------------------------------
// PersistentBulkSender — reconnects automatically on socket close
// This mimics real application behaviour (e.g. YouTube, file downloads)
// where a client reconnects after a temporary network interruption.
// ---------------------------------------------------------------------------
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

        NS_LOG_INFO("Attempting connection to " << m_remoteAddr);
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
        {
            Simulator::Schedule(MilliSeconds(200), &PersistentBulkSender::Connect, this);
        }
    }

    void OnClosed(Ptr<Socket> socket)
    {
        NS_LOG_WARN("eMBB TCP socket closed — reconnecting in 500ms");
        m_socket = nullptr;
        if (m_running)
        {
            Simulator::Schedule(MilliSeconds(500), &PersistentBulkSender::Connect, this);
        }
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

        // Reschedule to check buffer availability soon
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
    // -----------------------------------------------------------------------
    // 1. CLI parameters
    // -----------------------------------------------------------------------
    uint32_t gymPort = 5555;
    double simTime = 50.0;
    Packet::EnablePrinting(); // Enable packet metadata for NetAnim details
    double stepInterval = 0.01;
    uint32_t nUrllc = 5;
    uint32_t nEmbb = 10;
    uint32_t nMmtc = 20;
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

    if (verbose)
    {
        LogComponentEnable("SliceSimulation", LOG_LEVEL_INFO);
        LogComponentEnable("SliceGymEnv", LOG_LEVEL_INFO);
        LogComponentEnable("NrMacSchedulerWeightedPF", LOG_LEVEL_INFO);
    }

    NS_LOG_INFO("=== 5G Network Slicing Simulation ===");
    NS_LOG_INFO("UEs: URLLC=" << nUrllc << " eMBB=" << nEmbb << " mMTC=" << nMmtc);
    NS_LOG_INFO("simTime=" << simTime << "s  step=" << stepInterval << "s");

    // -----------------------------------------------------------------------
    // 2. Create nodes
    // -----------------------------------------------------------------------
    NodeContainer gnbNodes;
    gnbNodes.Create(1); // Single gNB node for all slices (using BWPs)

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

    // -----------------------------------------------------------------------
    // 3. Mobility (Fixed grid layout for NetAnim)
    // -----------------------------------------------------------------------
    // Seed for reproducibility
    Ptr<UniformRandomVariable> rng = CreateObject<UniformRandomVariable>();

    // -- gNBs: Use separate nodes at the SAME POSITION (Co-located) --
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    Ptr<ListPositionAllocator> gnbPos = CreateObject<ListPositionAllocator>();
    gnbPos->Add(Vector(200.0, 200.0, 25.0)); // Single central site for all BWPs/Slices
    mobility.SetPositionAllocator(gnbPos);
    mobility.Install(gnbNodes);

    // -- URLLC and mMTC UEs: Random initial positions with fixed Z=1.5 --
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                              "Bounds",
                              RectangleValue(Rectangle(0, 400, 0, 400)),
                              "Distance",
                              DoubleValue(2.0),
                              "Speed",
                              StringValue("ns3::UniformRandomVariable[Min=1.0|Max=5.0]"));

    Ptr<ListPositionAllocator> uePos = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < nUrllc + nMmtc; i++)
    {
        double x = rng->GetValue(0.0, 399.0);
        double y = rng->GetValue(0.0, 399.0);
        uePos->Add(Vector(x, y, 1.5));
    }
    mobility.SetPositionAllocator(uePos);

    mobility.Install(urllcUes);
    mobility.Install(mmtcUes);

    // -- eMBB: Constrained mobility to the middle of the cell (100-300m range) --
    MobilityHelper embbMobility;

    Ptr<ListPositionAllocator> embbPos = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < nEmbb; i++)
    {
        double x = rng->GetValue(100.0, 300.0); // 100-300 range
        double y = rng->GetValue(100.0, 300.0); // 100-300 range
        embbPos->Add(Vector(x, y, 1.5));
    }
    embbMobility.SetPositionAllocator(embbPos);

    Ptr<RandomRectanglePositionAllocator> embbWaypointAlloc = CreateObject<RandomRectanglePositionAllocator>();
    embbWaypointAlloc->SetX(CreateObjectWithAttributes<UniformRandomVariable>("Min", DoubleValue(100.0), "Max", DoubleValue(300.0)));
    embbWaypointAlloc->SetY(CreateObjectWithAttributes<UniformRandomVariable>("Min", DoubleValue(100.0), "Max", DoubleValue(300.0)));

    embbMobility.SetMobilityModel("ns3::RandomWaypointMobilityModel",
                                  "Speed",
                                  StringValue("ns3::UniformRandomVariable[Min=1.0|Max=3.0]"),
                                  "Pause",
                                  StringValue("ns3::UniformRandomVariable[Min=2.0|Max=8.0]"),
                                  "PositionAllocator", PointerValue(embbWaypointAlloc));

    embbMobility.Install(embbUes);

    // Ensure all eMBB UEs have correct Z coordinate
    for (uint32_t i = 0; i < embbUes.GetN(); ++i)
    {
        Ptr<MobilityModel> mob = embbUes.Get(i)->GetObject<MobilityModel>();
        Vector pos = mob->GetPosition();
        pos.z = 1.5;
        mob->SetPosition(pos);
    }

    // -----------------------------------------------------------------------
    // 4. Remote host + EPC
    // -----------------------------------------------------------------------
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);

    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<Node> pgw = epcHelper->GetPgwNode();

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", StringValue("10Gbps"));
    p2ph.SetChannelAttribute(
        "Delay",
        StringValue("0.1ms")); // IMPROVEMENT: 0.1ms to allow sub-ms URLLC latency
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(1500));

    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);

    // Position PGW and RemoteHost (Stationary)
    MobilityHelper mobilityConst;
    mobilityConst.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobilityConst.Install(pgw);
    pgw->GetObject<MobilityModel>()->SetPosition(Vector(200.0, 60.0, 1.5));
    mobilityConst.Install(remoteHost);
    remoteHost->GetObject<MobilityModel>()->SetPosition(Vector(200.0, 10.0, 1.5));

    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    // -----------------------------------------------------------------------
    // 5. NR Helper
    // -----------------------------------------------------------------------
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    nrHelper->SetEpcHelper(epcHelper);

    // -----------------------------------------------------------------------
    // 6. Bandwidth Parts (one per slice)
    //    BWP 0 — URLLC:  100 MHz @ 6.0 GHz  (sub-6, FR1-High, UMa)
    //    BWP 1 — eMBB:   100 MHz @ 3.5 GHz  (sub-6, FR1, UMa)
    //    BWP 2 — mMTC:   10 MHz  @ 0.7 GHz  (sub-GHz, FR1, RMa)
    // -----------------------------------------------------------------------
    CcBwpCreator ccBwpCreator;

    CcBwpCreator::SimpleOperationBandConf urllcBandConf(6.0e9, 100e6, 1, BandwidthPartInfo::UMa);
    CcBwpCreator::SimpleOperationBandConf embbBandConf(3.5e9, 100e6, 1, BandwidthPartInfo::UMa);
    CcBwpCreator::SimpleOperationBandConf mmtcBandConf(700e6, 10e6, 1, BandwidthPartInfo::RMa);

    OperationBandInfo urllcBand = ccBwpCreator.CreateOperationBandContiguousCc(urllcBandConf);
    OperationBandInfo embbBand = ccBwpCreator.CreateOperationBandContiguousCc(embbBandConf);
    OperationBandInfo mmtcBand = ccBwpCreator.CreateOperationBandContiguousCc(mmtcBandConf);

    // FIX: set unique BWP IDs BEFORE InitializeOperationBand
    urllcBand.m_cc.at(0)->m_bwp.at(0)->m_bwpId = 0;
    embbBand.m_cc.at(0)->m_bwp.at(0)->m_bwpId = 1;
    mmtcBand.m_cc.at(0)->m_bwp.at(0)->m_bwpId = 2;

    nrHelper->InitializeOperationBand(&urllcBand);
    nrHelper->InitializeOperationBand(&embbBand);
    nrHelper->InitializeOperationBand(&mmtcBand);

    BandwidthPartInfoPtrVector allBwps;
    allBwps.push_back(urllcBand.m_cc.at(0)->m_bwp.at(0));
    allBwps.push_back(embbBand.m_cc.at(0)->m_bwp.at(0));
    allBwps.push_back(mmtcBand.m_cc.at(0)->m_bwp.at(0));

    BandwidthPartInfoPtrVector urllcBwps;
    urllcBwps.push_back(allBwps.at(0));
    BandwidthPartInfoPtrVector embbBwps;
    embbBwps.push_back(allBwps.at(1));
    BandwidthPartInfoPtrVector mmtcBwps;
    mmtcBwps.push_back(allBwps.at(2));

    // Extract individual containers for UE installation (order matches band vector above)
    nrHelper->SetSchedulerTypeId(NrMacSchedulerWeightedPF::GetTypeId());
    nrHelper->SetSchedulerAttribute("SliceWeight", DoubleValue(1.0));
    nrHelper->SetGnbPhyAttribute("TxPower", DoubleValue(43.0));

    // Install single gNB device with 3 BWPs
    NetDeviceContainer gnbDevs = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
    Ptr<NrGnbNetDevice> gnbDev = DynamicCast<NrGnbNetDevice>(gnbDevs.Get(0));

    // Configure per-BWP parameters (Numerology)
    // BWP 0: URLLC (120 kHz SCS)
    gnbDev->GetPhy(0)->SetAttribute("Numerology", UintegerValue(3));
    // BWP 1: eMBB (30 kHz SCS)
    gnbDev->GetPhy(1)->SetAttribute("Numerology", UintegerValue(1));
    // BWP 2: mMTC (15 kHz SCS)
    gnbDev->GetPhy(2)->SetAttribute("Numerology", UintegerValue(0));

    // -----------------------------------------------------------------------
    // 8. Create, Configure and Install UEs
    // -----------------------------------------------------------------------
    nrHelper->SetUeBwpManagerAlgorithmAttribute("NGBR_VIDEO_TCP_DEFAULT", UintegerValue(0));
    NetDeviceContainer ueDevUrllc = nrHelper->InstallUeDevice(urllcUes, allBwps);

    nrHelper->SetUeBwpManagerAlgorithmAttribute("NGBR_VIDEO_TCP_DEFAULT", UintegerValue(1));
    NetDeviceContainer ueDevEmbb = nrHelper->InstallUeDevice(embbUes, allBwps);

    nrHelper->SetUeBwpManagerAlgorithmAttribute("NGBR_VIDEO_TCP_DEFAULT", UintegerValue(2));
    NetDeviceContainer ueDevMmtc = nrHelper->InstallUeDevice(mmtcUes, allBwps);

    // Update configs
    gnbDev->UpdateConfig();
    auto updateUeConfigs = [](NetDeviceContainer& devs) {
        for (auto it = devs.Begin(); it != devs.End(); ++it)
        {
            if (auto ue = DynamicCast<NrUeNetDevice>(*it))
                ue->UpdateConfig();
        }
    };
    updateUeConfigs(ueDevUrllc);
    updateUeConfigs(ueDevEmbb);
    updateUeConfigs(ueDevMmtc);

    // -----------------------------------------------------------------------
    // 8. Retrieve scheduler pointers for weight updates
    // -----------------------------------------------------------------------
    g_scheduler[SLICE_URLLC] =
        DynamicCast<NrMacSchedulerWeightedPF>(nrHelper->GetScheduler(gnbDevs.Get(0), 0));
    g_scheduler[SLICE_EMBB] =
        DynamicCast<NrMacSchedulerWeightedPF>(nrHelper->GetScheduler(gnbDevs.Get(0), 1));
    g_scheduler[SLICE_MMTC] =
        DynamicCast<NrMacSchedulerWeightedPF>(nrHelper->GetScheduler(gnbDevs.Get(0), 2));

    NS_ABORT_MSG_IF(!g_scheduler[SLICE_URLLC], "Could not retrieve URLLC scheduler from BWP 0");
    NS_ABORT_MSG_IF(!g_scheduler[SLICE_EMBB], "Could not retrieve eMBB scheduler from BWP 1");
    NS_ABORT_MSG_IF(!g_scheduler[SLICE_MMTC], "Could not retrieve mMTC scheduler from BWP 2");

    // -----------------------------------------------------------------------
    // 9. Internet stack + IP addressing on UEs
    // -----------------------------------------------------------------------
    internet.Install(allUes);

    Ipv4AddressHelper ueIpHelper;

    ueIpHelper.SetBase("7.0.1.0", "255.255.255.0");
    Ipv4InterfaceContainer urllcIpIface = ueIpHelper.Assign(ueDevUrllc);

    ueIpHelper.SetBase("7.0.2.0", "255.255.255.0");
    Ipv4InterfaceContainer embbIpIface = ueIpHelper.Assign(ueDevEmbb);

    ueIpHelper.SetBase("7.0.3.0", "255.255.255.0");
    Ipv4InterfaceContainer mmtcIpIface = ueIpHelper.Assign(ueDevMmtc);

    for (uint32_t i = 0; i < allUes.GetN(); ++i)
    {
        Ptr<Ipv4StaticRouting> ueRouting =
            ipv4RoutingHelper.GetStaticRouting(allUes.Get(i)->GetObject<Ipv4>());
        ueRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    // -----------------------------------------------------------------------
    // 10. Attach UEs
    // -----------------------------------------------------------------------
    nrHelper->AttachToClosestEnb(ueDevUrllc, gnbDevs);
    nrHelper->AttachToClosestEnb(ueDevEmbb, gnbDevs);
    nrHelper->AttachToClosestEnb(ueDevMmtc, gnbDevs);

    // -----------------------------------------------------------------------
    // 11. Applications
    // -----------------------------------------------------------------------
    double appStart = 1.0;
    double appStop = simTime; // IMPROVEMENT: Run until sim ends to prevent metric collapse

    // URLLC: tight-loop UDP CBR (robotic-arm control, 512 B / 1 ms)
    InstallUdpCbr(urllcUes, remoteHost, urllcIpIface, 3000, 512, 0.001, appStart, appStop);

    // eMBB: Persistent TCP bulk-send with auto-reconnect (Recommendation for Reliability)
    PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), 4000));
    ApplicationContainer sinkApps = sink.Install(embbUes);
    sinkApps.Start(Seconds(appStart));
    sinkApps.Stop(Seconds(appStop));

    for (uint32_t i = 0; i < embbUes.GetN(); ++i)
    {
        Ptr<PersistentBulkSender> sender = CreateObject<PersistentBulkSender>();
        sender->Setup(InetSocketAddress(embbIpIface.GetAddress(i), 4000), 1400);

        remoteHost->AddApplication(sender);
        sender->SetStartTime(Seconds(appStart + 0.05 * i));
        sender->SetStopTime(Seconds(appStop));
    }

    // mMTC: periodic tiny UDP (IoT sensors, 64 B / 100 ms)
    InstallUdpCbr(mmtcUes, remoteHost, mmtcIpIface, 5000, 64, 0.1, appStart, appStop);

    // -----------------------------------------------------------------------
    // 12. Flow Monitor
    // -----------------------------------------------------------------------
    FlowMonitorHelper fmHelper;
    Ptr<FlowMonitor> flowMonitor = fmHelper.InstallAll();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(fmHelper.GetClassifier());

    // -----------------------------------------------------------------------
    // 13. OpenGym / ns3-gym setup
    // -----------------------------------------------------------------------
    Ptr<SliceGymEnv> gymEnv = CreateObject<SliceGymEnv>();
    gymEnv->SetAttribute("StepInterval", DoubleValue(stepInterval));
    gymEnv->SetAttribute("MaxSteps", UintegerValue(static_cast<uint32_t>(simTime / stepInterval)));

    gymEnv->SetFlowMonitor(flowMonitor, classifier);

    // Register slice subnets for flow classification.
    // EPC assigns addresses from 7.0.0.0/8; inspect flowMonitor output to
    // confirm the exact addresses assigned and update these if needed.
    gymEnv->RegisterSliceSubnet(SLICE_URLLC, Ipv4Address("7.0.1.0"), Ipv4Mask("255.255.255.0"));
    gymEnv->RegisterSliceSubnet(SLICE_EMBB, Ipv4Address("7.0.2.0"), Ipv4Mask("255.255.255.0"));
    gymEnv->RegisterSliceSubnet(SLICE_MMTC, Ipv4Address("7.0.3.0"), Ipv4Mask("255.255.255.0"));

    gymEnv->SetSliceSLA(SLICE_URLLC, 1.0, 1.0, 0.85);
    gymEnv->SetSliceSLA(SLICE_EMBB, 20.0, 50.0, 0.90);
    gymEnv->SetSliceSLA(SLICE_MMTC, 0.1, 200.0, 0.80);

    gymEnv->SetWeightSetterCallback(ApplyWeights);

    // Connect gym env to OpenGym TCP interface
    Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface>(gymPort);
    gymEnv->SetOpenGymInterface(openGymInterface); // registers all callbacks

    gymEnv->ScheduleNextStateRead();

    // -----------------------------------------------------------------------
    // PRB Telemetry Trace Connections (Bug 6 Fix)
    // -----------------------------------------------------------------------
    // We connect SlotDataStats and SlotCtrlStats from the gNB PHY (per BWP)
    // to the GymEnv TraceSink to get accurate resource block utilization.
    auto connectPrbTraces = [&gymEnv](Ptr<NetDevice> gnbDev) {
        Ptr<NrGnbNetDevice> gnb = DynamicCast<NrGnbNetDevice>(gnbDev);
        if (!gnb)
            return;

        for (uint32_t sliceId = 0; sliceId < NUM_SLICES; ++sliceId)
        {
            Ptr<NrGnbPhy> phy = DynamicCast<NrGnbPhy>(gnb->GetPhy(sliceId));
            if (!phy)
                continue;

            phy->TraceConnectWithoutContext(
                "SlotDataStats",
                MakeCallback(&SliceGymEnv::SlotStatsTraceSink, gymEnv).Bind(sliceId));
            phy->TraceConnectWithoutContext(
                "SlotCtrlStats",
                MakeCallback(&SliceGymEnv::SlotStatsTraceSink, gymEnv).Bind(sliceId));
        }
    };
    connectPrbTraces(gnbDevs.Get(0));

    // -----------------------------------------------------------------------
    // 14. Optional PCAP
    // -----------------------------------------------------------------------
    if (pcapEnabled)
        p2ph.EnablePcapAll("5g-slice");

    // -----------------------------------------------------------------------
    // 15. NetAnim Trace & Configuration
    // -----------------------------------------------------------------------
    static AnimationInterface anim("slice-simulation.xml"); // Global-ish lifecycle
    g_anim = &anim;
    g_gnbId = gnbNodes.Get(0)->GetId(); // The single central gNB node for all annotations

    anim.EnablePacketMetadata(true);
    anim.SetMaxPktsPerTraceFile(100000);
    anim.SetMobilityPollInterval(
        Seconds(0.05)); // Increase poll frequency for smoother mobility tracking

    // Register node counters for real-time throughput plots
    g_urllcResId = anim.AddNodeCounter("URLLC Tput (Mbps)", AnimationInterface::DOUBLE_COUNTER);
    g_embbResId = anim.AddNodeCounter("eMBB Tput (Mbps)", AnimationInterface::DOUBLE_COUNTER);
    g_mmtcResId = anim.AddNodeCounter("mMTC Tput (Mbps)", AnimationInterface::DOUBLE_COUNTER);

    // gNB nodes (Blue)
    for (uint32_t i = 0; i < gnbNodes.GetN(); ++i)
    {
        anim.UpdateNodeDescription(gnbNodes.Get(i), "gNB-" + std::to_string(i));
        anim.UpdateNodeColor(gnbNodes.Get(i), 0, 0, 255);
        anim.UpdateNodeSize(gnbNodes.Get(i)->GetId(), 3.0, 3.0);
    }
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
        anim.UpdateNodeColor(urllcUes.Get(i), 255, 0, 0); // Red
        anim.UpdateNodeSize(urllcUes.Get(i)->GetId(), 2.0, 2.0);
    }
    for (uint32_t i = 0; i < embbUes.GetN(); ++i)
    {
        anim.UpdateNodeDescription(embbUes.Get(i), "E" + std::to_string(i));
        anim.UpdateNodeColor(embbUes.Get(i), 0, 200, 0); // Green
        anim.UpdateNodeSize(embbUes.Get(i)->GetId(), 2.0, 2.0);
    }
    for (uint32_t i = 0; i < mmtcUes.GetN(); ++i)
    {
        anim.UpdateNodeDescription(mmtcUes.Get(i), "M" + std::to_string(i));
        anim.UpdateNodeColor(mmtcUes.Get(i), 255, 165, 0); // Orange
        anim.UpdateNodeSize(mmtcUes.Get(i)->GetId(), 2.0, 2.0);
    }

    // Schedule periodic counter updates
    Simulator::Schedule(Seconds(appStart + 0.1),
                        &UpdateAnimCounters,
                        gymEnv,
                        urllcUes,
                        embbUes,
                        mmtcUes);

    // -----------------------------------------------------------------------
    // 16. Run simulation
    // -----------------------------------------------------------------------
    NS_LOG_INFO("Starting simulation for " << simTime << " seconds...");

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    // -----------------------------------------------------------------------
    // 17. Post-simulation statistics
    // -----------------------------------------------------------------------
    NS_LOG_INFO("\n===== Flow Statistics =====");
    flowMonitor->SerializeToXmlFile("5g-slice-flow-stats-1.xml", true, true);

    openGymInterface->NotifySimulationEnd();

    Simulator::Destroy();

    NS_LOG_INFO("Simulation complete.");
    return 0;
}