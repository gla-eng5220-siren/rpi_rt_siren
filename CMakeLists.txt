cmake_minimum_required(VERSION 3.12)
project(rpi_rt)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 17)

option(USE_SYSTEM_XNNPACK "Use system-installed XNNPACK" ON)

include(EnsureXNNPACK) # provides XNNPACK::XNNPACK
find_package(Threads)
find_package(JPEG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LibV4l2 REQUIRED IMPORTED_TARGET GLOBAL libv4l2)

enable_testing()
add_subdirectory(tests)

add_executable(flame_iris
  src/flame_iris.cpp
  src/sensor/v4l2_camera.cpp
  src/logic/shufflenet.cpp
  src/logic/temperature_threshold_logic.cpp
  src/logic/visual_classify_logic.cpp
  src/misc/jpeg_utils.cpp
  src/sensor/mock_temperature_sensor.cpp
  src/alarm/stdout_alarm.cpp
)
target_include_directories(flame_iris PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(flame_iris PRIVATE
  Threads::Threads
  XNNPACK::XNNPACK
  PkgConfig::LibV4l2
  JPEG::JPEG
)

