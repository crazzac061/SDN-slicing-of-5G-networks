/* =============================================================================
 * slice-simulation.cc
 * 5G Network Slicing Simulation — Infrastructure + Control Layer
 *
 * FIXES APPLIED:
 *   Bug 1 — Scheduler retrieval: GetObject<>() → DynamicCast<>()
 *            GetObject traverses ns-3 aggregation chains; DynamicCast does a
 *            proper C++ dynamic_cast.  GetObject returned nullptr every time,
 *            so the g_scheduler[] pointers were never set and weights were
 *            never applied to the actual schedulers.
 *
 *   Bug 2 — OpenGym interface setup: removed the erroneous
 *            SetGetObservationSpaceCb() call on a throwaway SliceGymEnv.
 *            SetOpenGymInterface() on the real gymEnv registers all callbacks
 *            automatically through the OpenGymEnv base class.
 *
 *   Bug 3 — UE position allocators: eMBB UEs now centred on their gNB at
 *            (50, 0) and mMTC UEs centred on their gNB at (100, 0), not both
 *            at origin (0, 0).  Previously almost all mMTC UEs were >900 m
 *            from their gNB, causing very poor SINR / disconnected UEs.
 *
 *   ApplyWeights redesign: changed to call SetWeight() (single weight per
 *            instance) instead of SetSliceWeight(bwpId, …) with the old
 *            3-element array.  Multiply normalised weight by NUM_SLICES so
 *            the neutral value 1/3 maps to weight=1.0 for each scheduler.
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
#include "ns3/nr-module.h"

// ns3-gym
#include "ns3/opengym-module.h"

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
    double stepInterval = 0.1;
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
    gnbNodes.Create(3); // [0]=URLLC, [1]=eMBB, [2]=mMTC

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
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    // -- gNBs: Use separate nodes at the SAME POSITION (Co-located) --
    Ptr<ListPositionAllocator> gnbPos = CreateObject<ListPositionAllocator>();
    gnbPos->Add(Vector(200.0, 200.0, 10.0)); // URLLC site
    gnbPos->Add(Vector(200.0, 200.0, 10.0)); // eMBB  site
    gnbPos->Add(Vector(200.0, 200.0, 10.0)); // mMTC  site
    mobility.SetPositionAllocator(gnbPos);
    mobility.Install(gnbNodes);

    // -- UEs: Random scattering and RandomWalk2d Mobility --
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
    uePos->SetAttribute(
        "Z",
        StringValue(
            "ns3::UniformRandomVariable[Min=1.5|Max=1.5]")); // FIX: 1.5m Height for 3GPP channel
    mobility.SetPositionAllocator(uePos);

    mobility.Install(urllcUes);
    mobility.Install(mmtcUes);

    // -- eMBB: Constrained mobility to the middle of the cell (100-300m range) --
    // This prevents RLF by keeping UEs in good coverage.
    Ptr<RandomRectanglePositionAllocator> embbPos =
        CreateObject<RandomRectanglePositionAllocator>();
    embbPos->SetAttribute("X", StringValue("ns3::UniformRandomVariable[Min=100.0|Max=300.0]"));
    embbPos->SetAttribute("Y", StringValue("ns3::UniformRandomVariable[Min=100.0|Max=300.0]"));

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
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(pgw);
    pgw->GetObject<MobilityModel>()->SetPosition(Vector(200.0, 60.0, 0.0));
    mobility.Install(remoteHost);
    remoteHost->GetObject<MobilityModel>()->SetPosition(Vector(200.0, 10.0, 0.0));

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
    //    BWP 0 — URLLC:  10 MHz @ 28.0 GHz  (mmWave, FR2, UMi)
    //    BWP 1 — eMBB:   40 MHz @  3.5 GHz  (sub-6, FR1, UMa)
    //    BWP 2 — mMTC:   10 MHz @  0.7 GHz  (sub-GHz, FR1, RMa)
    // -----------------------------------------------------------------------
    CcBwpCreator ccBwpCreator;

    CcBwpCreator::SimpleOperationBandConf urllcBandConf(
        6.0e9,
        100e6,
        1,
        BandwidthPartInfo::UMa); // CHANGE: 6 GHz (FR1-High/UMa) for better range than 28 GHz
    OperationBandInfo urllcBand = ccBwpCreator.CreateOperationBandContiguousCc(urllcBandConf);

    CcBwpCreator::SimpleOperationBandConf embbBandConf(
        3.5e9,
        100e6,
        1,
        BandwidthPartInfo::UMa); // IMPROVEMENT: 100 MHz for higher eMBB throughput
    OperationBandInfo embbBand = ccBwpCreator.CreateOperationBandContiguousCc(embbBandConf);

    CcBwpCreator::SimpleOperationBandConf mmtcBandConf(700e6, 10e6, 1, BandwidthPartInfo::RMa);
    OperationBandInfo mmtcBand = ccBwpCreator.CreateOperationBandContiguousCc(mmtcBandConf);

    nrHelper->InitializeOperationBand(&urllcBand);
    nrHelper->InitializeOperationBand(&embbBand);
    nrHelper->InitializeOperationBand(&mmtcBand);

    BandwidthPartInfoPtrVector urllcBwps = CcBwpCreator::GetAllBwps({urllcBand});
    BandwidthPartInfoPtrVector embbBwps = CcBwpCreator::GetAllBwps({embbBand});
    BandwidthPartInfoPtrVector mmtcBwps = CcBwpCreator::GetAllBwps({mmtcBand});

    // -----------------------------------------------------------------------
    // 7. Install gNB + UE devices (weighted PF scheduler per BWP)
    // -----------------------------------------------------------------------

    // --- URLLC BWP ---
    nrHelper->SetSchedulerTypeId(NrMacSchedulerWeightedPF::GetTypeId());
    nrHelper->SetSchedulerAttribute("SliceWeight", DoubleValue(1.0));
    nrHelper->SetGnbPhyAttribute("Numerology",
                                 UintegerValue(3)); // 120 kHz SCS (User requested to keep this)
    nrHelper->SetGnbPhyAttribute("TxPower", DoubleValue(43.0));
    NetDeviceContainer gnbDevUrllc =
        nrHelper->InstallGnbDevice(NodeContainer(gnbNodes.Get(SLICE_URLLC)), urllcBwps);
    NetDeviceContainer ueDevUrllc = nrHelper->InstallUeDevice(urllcUes, urllcBwps);

    // --- eMBB BWP ---
    nrHelper->SetSchedulerTypeId(NrMacSchedulerWeightedPF::GetTypeId());
    nrHelper->SetSchedulerAttribute("SliceWeight", DoubleValue(1.0));
    nrHelper->SetGnbPhyAttribute("Numerology", UintegerValue(1)); // 30 kHz SCS
    nrHelper->SetGnbPhyAttribute("TxPower", DoubleValue(43.0));
    NetDeviceContainer gnbDevEmbb =
        nrHelper->InstallGnbDevice(NodeContainer(gnbNodes.Get(SLICE_EMBB)), embbBwps);
    NetDeviceContainer ueDevEmbb = nrHelper->InstallUeDevice(embbUes, embbBwps);

    // --- mMTC BWP ---
    nrHelper->SetSchedulerTypeId(NrMacSchedulerWeightedPF::GetTypeId());
    nrHelper->SetSchedulerAttribute("SliceWeight", DoubleValue(1.0));
    nrHelper->SetGnbPhyAttribute("Numerology", UintegerValue(0)); // 15 kHz SCS
    nrHelper->SetGnbPhyAttribute("TxPower", DoubleValue(43.0));
    NetDeviceContainer gnbDevMmtc =
        nrHelper->InstallGnbDevice(NodeContainer(gnbNodes.Get(SLICE_MMTC)), mmtcBwps);
    NetDeviceContainer ueDevMmtc = nrHelper->InstallUeDevice(mmtcUes, mmtcBwps);

    // Update device configs
    auto updateConfigs = [](NetDeviceContainer& devs) {
        for (auto it = devs.Begin(); it != devs.End(); ++it)
        {
            if (auto gnb = DynamicCast<NrGnbNetDevice>(*it))
                gnb->UpdateConfig();
            else if (auto ue = DynamicCast<NrUeNetDevice>(*it))
                ue->UpdateConfig();
        }
    };
    updateConfigs(gnbDevUrllc);
    updateConfigs(ueDevUrllc);
    updateConfigs(gnbDevEmbb);
    updateConfigs(ueDevEmbb);
    updateConfigs(gnbDevMmtc);
    updateConfigs(ueDevMmtc);

    // -----------------------------------------------------------------------
    // 8. Retrieve scheduler pointers for weight updates
    // -----------------------------------------------------------------------
    // FIX Bug 1: was sched->GetObject<NrMacSchedulerWeightedPF>()
    //            GetObject() traverses ns-3 aggregation — not a cast.
    //            DynamicCast<T>() is the correct way to downcast a pointer.
    //
    // Note: NrHelper::GetScheduler(Ptr<NetDevice>, uint32_t bwpId) is an
    //       INSTANCE method in ns-3.40 5G-LENA.  If your build reports
    //       "not a member" for the instance call, use the fallback below.
    // -----------------------------------------------------------------------
    auto GetSchedPtr = [&](Ptr<NetDevice> gnbDev) -> Ptr<NrMacSchedulerWeightedPF> {
        // Primary: instance method (5G-LENA v2.x / ns-3.40)
        Ptr<NrMacScheduler> sched = nrHelper->GetScheduler(gnbDev, 0);
        // FIX Bug 1: DynamicCast, NOT GetObject
        return sched ? DynamicCast<NrMacSchedulerWeightedPF>(sched) : nullptr;

        // Fallback if the above fails to compile (some 5G-LENA builds expose
        // the scheduler via the gNB MAC):
        //   auto gnbNetDev = DynamicCast<NrGnbNetDevice>(gnbDev);
        //   auto mac       = gnbNetDev ? gnbNetDev->GetMac(0) : nullptr;
        //   return mac ? DynamicCast<NrMacSchedulerWeightedPF>(mac->GetScheduler()) : nullptr;
    };

    g_scheduler[SLICE_URLLC] = GetSchedPtr(gnbDevUrllc.Get(0));
    g_scheduler[SLICE_EMBB] = GetSchedPtr(gnbDevEmbb.Get(0));
    g_scheduler[SLICE_MMTC] = GetSchedPtr(gnbDevMmtc.Get(0));

    NS_ABORT_MSG_IF(!g_scheduler[SLICE_URLLC], "Could not retrieve URLLC scheduler");
    NS_ABORT_MSG_IF(!g_scheduler[SLICE_EMBB], "Could not retrieve eMBB scheduler");
    NS_ABORT_MSG_IF(!g_scheduler[SLICE_MMTC], "Could not retrieve mMTC scheduler");

    // -----------------------------------------------------------------------
    // 9. Internet stack + IP addressing on UEs
    // -----------------------------------------------------------------------
    internet.Install(allUes);

    // FIX: Manual IP assignment using a local Ipv4AddressHelper.
    // epcHelper->AssignUeIpv4Address() is a convenience wrapper around a private
    // address helper. Since SetUeAddressHelper() is not accessible, we assign
    // the addresses manually here. The EpcHelper will automatically pick them
    // up when the default bearers are activated during attachment.
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
    nrHelper->AttachToClosestEnb(ueDevUrllc, gnbDevUrllc);
    nrHelper->AttachToClosestEnb(ueDevEmbb, gnbDevEmbb);
    nrHelper->AttachToClosestEnb(ueDevMmtc, gnbDevMmtc);

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
    // FIX Bug 2: Create gymEnv FIRST, configure it, THEN connect interface.
    //            The original code called SetGetObservationSpaceCb() on a
    //            throwaway CreateObject<SliceGymEnv>() — that object was
    //            immediately discarded and had nothing to do with gymEnv.
    //            SetOpenGymInterface() registers all required callbacks
    //            automatically through the OpenGymEnv base class.

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
    auto connectPrbTraces = [&gymEnv](NetDeviceContainer& devs, uint32_t sliceId) {
        for (auto it = devs.Begin(); it != devs.End(); ++it)
        {
            Ptr<NrGnbNetDevice> gnb = DynamicCast<NrGnbNetDevice>(*it);
            if (!gnb)
                continue;
            // In this multi-slice setup, each gNB device instance has 1 BWP (index 0)
            Ptr<NrGnbPhy> phy = DynamicCast<NrGnbPhy>(gnb->GetPhy(0));
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
    connectPrbTraces(gnbDevUrllc, SLICE_URLLC);
    connectPrbTraces(gnbDevEmbb, SLICE_EMBB);
    connectPrbTraces(gnbDevMmtc, SLICE_MMTC);

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
    g_gnbId = gnbNodes.Get(SLICE_EMBB)->GetId(); // Use eMBB gNB as the central label node

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
    // 16. Post-simulation statistics
    // -----------------------------------------------------------------------
    NS_LOG_INFO("\n===== Flow Statistics =====");
    flowMonitor->SerializeToXmlFile("5g-slice-flow-stats.xml", true, true);

    openGymInterface->NotifySimulationEnd();

    Simulator::Destroy();

    NS_LOG_INFO("Simulation complete.");
    return 0;
}