cmake_minimum_required(VERSION 3.12)
project(rpi_rt)

set(CMAKE_CXX_STANDARD 17)
find_package(Threads)

include(FetchContent)

FetchContent_Declare(
  XNNPACK
  GIT_REPOSITORY https://github.com/google/XNNPACK.git
  GIT_TAG 810b9afd9a293f84be15185e0e35dcfab4f596c7 
)

set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(XNNPACK)

find_package(JPEG REQUIRED)

add_executable(demo_siren
  src/demo_siren.cpp
  src/sensor/mock_temperature_sensor.cpp
  src/alarm/stdout_alarm.cpp
)
target_include_directories(demo_siren PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(demo_siren PRIVATE
  Threads::Threads
)

add_executable(demo_shufflenet
  src/demo_shufflenet.cpp
  src/logic/shufflenet.cpp
  src/misc/jpeg_utils.cpp
)
target_include_directories(demo_shufflenet PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(demo_shufflenet PRIVATE
  Threads::Threads
  XNNPACK
  JPEG::JPEG
)

add_subdirectory(tests)

