cmake_minimum_required(VERSION 3.12)
project(rpi_rt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 17)

option(USE_SYSTEM_XNNPACK "Use system-installed XNNPACK" ON)

include(EnsureXNNPACK) # provides XNNPACK::XNNPACK

find_package(Threads REQUIRED)
find_package(JPEG REQUIRED)
find_package(FFMPEG COMPONENTS avformat avcodec avutil swscale REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenSSL COMPONENTS Crypto SSL REQUIRED)

pkg_check_modules(LibV4l2 REQUIRED IMPORTED_TARGET GLOBAL libv4l2)
pkg_check_modules(LibGpiod REQUIRED IMPORTED_TARGET GLOBAL libgpiod libgpiodcxx)
pkg_check_modules(LibCamera REQUIRED IMPORTED_TARGET GLOBAL libcamera libcamera-base)
pkg_check_modules(LibI2C REQUIRED IMPORTED_TARGET GLOBAL i2c)

enable_testing()
add_subdirectory(tests)

add_executable(flame_iris
  src/flame_iris.cpp
  src/sensor/v4l2_camera.cpp
  src/sensor/libcamera.cpp
  src/sensor/mock_camera.cpp
  src/sensor/mock_temperature_sensor.cpp
  src/sensor/breadpi_temperature_sensor.cpp
  src/logic/shufflenet.cpp
  src/logic/temperature_threshold_logic.cpp
  src/logic/visual_classify_logic.cpp
  src/misc/jpeg_utils.cpp
  src/alarm/stdout_alarm.cpp
  src/alarm/buzzer_alarm.cpp
  src/alarm/brevo_email_alarm.cpp
  src/alarm/buzzer.cpp
  src/http_server/http_server.cpp
)

target_include_directories(flame_iris PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(flame_iris PRIVATE
  Threads::Threads
  XNNPACK::XNNPACK
  PkgConfig::LibV4l2
  PkgConfig::LibGpiod
  PkgConfig::LibCamera
  PkgConfig::LibI2C
  JPEG::JPEG
  FFMPEG::avformat
  FFMPEG::avcodec
  FFMPEG::avutil
  FFMPEG::swscale
  OpenSSL::Crypto
  OpenSSL::SSL
)
