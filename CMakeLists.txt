cmake_minimum_required(VERSION 3.12)
project(rpi_rt)

set(CMAKE_CXX_STANDARD 17)
find_package(Threads)

include(FetchContent)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LibV4l2 REQUIRED IMPORTED_TARGET GLOBAL libv4l2)

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

FetchContent_Declare(
    CImg
    GIT_REPOSITORY https://github.com/GreycLab/CImg.git
    GIT_TAG v.3.7.0
)

FetchContent_MakeAvailable(CImg)

add_executable(demo_capture
  src/demo_capture.cpp
  src/sensor/v4l2_camera.cpp
)
target_include_directories(demo_capture PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${cimg_SOURCE_DIR}
)
target_link_libraries(demo_capture PRIVATE
  Threads::Threads
  PkgConfig::LibV4l2
)

