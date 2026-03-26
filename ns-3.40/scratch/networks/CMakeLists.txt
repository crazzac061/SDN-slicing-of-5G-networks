# =============================================================================
# CMakeLists.txt — ns-3.40 scratch/network-slicing/
# =============================================================================

build_exec(
  EXECNAME
    networks
  SOURCE_FILES
    slice-simulation.cc
    slice-gym-env.cc
    nr-mac-scheduler-weighted-pf.cc
  HEADER_FILES
    slice-gym-env.h
    nr-mac-scheduler-weighted-pf.h
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libmobility}
    ${libapplications}
    ${libflow-monitor}
    ${libpoint-to-point}
    ${libnr}
    ${libopengym}
    ${libnetanim}
  EXECUTABLE_DIRECTORY_PATH
    scratch/networks/
)