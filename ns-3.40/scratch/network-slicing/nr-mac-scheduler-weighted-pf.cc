// =============================================================================
// nr-mac-scheduler-weighted-pf.cc
// Weight-Based Proportional Fair Scheduler — Implementation
// =============================================================================

#include "nr-mac-scheduler-weighted-pf.h"

#include "ns3/log.h"
#include "ns3/double.h"

#include <algorithm>
#include <cmath>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("NrMacSchedulerWeightedPF");
NS_OBJECT_ENSURE_REGISTERED (NrMacSchedulerWeightedPF);

// ---------------------------------------------------------------------------
TypeId
NrMacSchedulerWeightedPF::GetTypeId ()
{
    static TypeId tid =
        TypeId ("ns3::NrMacSchedulerWeightedPF")
            .SetParent<NrMacSchedulerOfdmaPF> ()
            .SetGroupName ("nr")
            .AddConstructor<NrMacSchedulerWeightedPF> ()
            .AddAttribute ("SliceWeight",
                           "Scheduling aggressiveness weight for this slice. "
                           "1.0 = neutral. >1.0 = more aggressive (shorter PF window). "
                           "<1.0 = more conservative (longer PF window).",
                           DoubleValue (1.0),
                           MakeDoubleAccessor (&NrMacSchedulerWeightedPF::SetSliceWeightAttr,
                                               &NrMacSchedulerWeightedPF::GetSliceWeightAttr),
                           MakeDoubleChecker<double> (0.01, 100.0));
    return tid;
}

NrMacSchedulerWeightedPF::NrMacSchedulerWeightedPF ()
    : NrMacSchedulerOfdmaPF ()
{
    NS_LOG_FUNCTION (this);
    // Initial weight = 1.0; no change to default time window needed.
}

// ---------------------------------------------------------------------------
// Primary weight setter — this is what the SDN controller calls
// ---------------------------------------------------------------------------

void
NrMacSchedulerWeightedPF::SetWeight (double weight)
{
    NS_LOG_FUNCTION (this << weight);
    NS_ASSERT_MSG (weight > 0.0, "Slice weight must be strictly positive");

    m_weight = weight;
    ApplyWeightAsTimeWindow ();

    NS_LOG_INFO ("Slice weight set to " << weight
                 << "  → PF TimeWindow = "
                 << std::max (1.0, BASE_WINDOW_MS / weight) << " ms");
}

// ---------------------------------------------------------------------------
// Backward-compat wrappers
// ---------------------------------------------------------------------------

void
NrMacSchedulerWeightedPF::SetSliceWeight (uint8_t bwpId, double weight)
{
    // bwpId accepted but ignored — this instance is for exactly one slice.
    NS_LOG_FUNCTION (this << +bwpId << weight);
    SetWeight (weight);
}

double
NrMacSchedulerWeightedPF::GetSliceWeight (uint8_t /*bwpId*/) const
{
    return m_weight;
}

void
NrMacSchedulerWeightedPF::SetSliceWeightAttr (double weight)
{
    SetWeight (weight);
}

double
NrMacSchedulerWeightedPF::GetSliceWeightAttr () const
{
    return m_weight;
}

// ---------------------------------------------------------------------------
// Private: translate weight to PF TimeWindow attribute
// ---------------------------------------------------------------------------

void
NrMacSchedulerWeightedPF::ApplyWeightAsTimeWindow ()
{
    // PF metric = achievableRate / avgThroughput(timeWindow)
    // Shorter window → avg decays faster → metric is more sensitive to
    // current channel quality → scheduler serves good-channel UEs harder.
    // That means the slice gets "pushed" harder → effectively higher priority.
    //
    // Mapping: timeWindow = BASE_WINDOW / weight
    //   weight=1.0 → timeWindow=99  (default, neutral)
    //   weight=3.0 → timeWindow=33  (3× more reactive)
    //   weight=0.5 → timeWindow=198 (2× more conservative)

    double tw = std::max (1.0, BASE_WINDOW_MS / m_weight);

    // SetAttribute on the base class attribute.
    // "TimeWindow" is registered in NrMacSchedulerOfdmaPF::GetTypeId().
    SetAttribute ("LastAvgTPutWeight", DoubleValue (tw));
}

} // namespace ns3