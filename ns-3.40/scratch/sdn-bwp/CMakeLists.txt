# =============================================================================
# CMakeLists.txt — ns-3.40 scratch/sdn-bwp/
# 5G NR Network Slicing with Dynamic BWP Management
# =============================================================================

build_exec(
  EXECNAME
    sdn_bwp
  SOURCE_FILES
    slicing-sim.cc
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
    scratch/sdn-bwp/
)
