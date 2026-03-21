# Check and try to use system-vendored XNNPACK
# If failed, fallback to manual compiling

if(USE_SYSTEM_XNNPACK)
  find_path(XNNPACK_INCLUDE_DIR
    NAMES xnnpack.h
    PATHS /usr/include /usr/local/include
  )

  find_library(XNNPACK_LIBRARY
    NAMES XNNPACK
    PATHS
    /usr/lib
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib
    /usr/lib64
    /usr/lib64/aarch64-linux-gnu
    /usr/local/lib64
  )

  if(XNNPACK_INCLUDE_DIR AND XNNPACK_LIBRARY)
    add_library(XNNPACK::SysXNNPACK UNKNOWN IMPORTED)

    set_target_properties(XNNPACK::SysXNNPACK PROPERTIES
      IMPORTED_LOCATION ${XNNPACK_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${XNNPACK_INCLUDE_DIR}
    )

    try_compile(XNNPACK_VERSION_CORRECT
      SOURCES "${PROJECT_SOURCE_DIR}/cmake/xnnpack_try_compile.c"
      LINK_LIBRARIES PRIVATE XNNPACK::SysXNNPACK
    )

    if (NOT XNNPACK_VERSION_CORRECT)
      message(WARNING 
        "System XNNPACK API Mismatch"
        " -- on debian, install version 0.0~git20241108.4ea82e5")
      set(USE_SYSTEM_XNNPACK OFF)
    else()
      add_library(XNNPACK::XNNPACK ALIAS XNNPACK::SysXNNPACK)
      message(STATUS "Using system XNNPACK")
    endif()
  else()
    message(WARNING "System XNNPACK not found, falling back")
    set(USE_SYSTEM_XNNPACK OFF)
  endif()
endif()

if(NOT USE_SYSTEM_XNNPACK)
  message(STATUS "Building XNNPACK from scratch, this will take a while")

  include(FetchContent)

  FetchContent_Declare(
    XNNPACK
    GIT_REPOSITORY https://github.com/google/XNNPACK.git
    GIT_TAG 4ea82e595b36106653175dcb04b2aa532660d0d8
  )

  set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(XNNPACK)
  add_library(XNNPACK::XNNPACK ALIAS XNNPACK)
endif()

