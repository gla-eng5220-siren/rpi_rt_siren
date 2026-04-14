#include "catch2/catch_test_macros.hpp"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <iostream>
#include <thread>
#include <sstream>

#include "detection_result.hpp"
#include "frame.hpp"
#include "alarm.hpp"
#include "sensor.hpp"

#ifndef TESTDATA_PATH
  #define TESTDATA_PATH "testdata"
#endif

TEST_CASE("Frame", "[system][frame]") {
  rpi_rt::Frame<uint8_t> u8_frame{100, 80, 3};
  CHECK(u8_frame.height() == 100);
  CHECK(u8_frame.width() == 80);
  CHECK(u8_frame.channels() == 3);
  CHECK(u8_frame.size() == 100 * 80 * 3);

  u8_frame.resize(300, 200, 1);
  CHECK(u8_frame.height() == 300);
  CHECK(u8_frame.width() == 200);
  CHECK(u8_frame.channels() == 1);
  CHECK(u8_frame.size() == 300 * 200 * 1);
}

class mock_detection_result : public rpi_rt::detection_result_t {
  public:
    ~mock_detection_result() {}

    virtual bool has_fire() override {
      return true;
    }

    virtual std::string explain() override {
      return "!!TEST!!";
    }
};

TEST_CASE("PassDetectionResult", "[system][alarm][detection_result]") {
  auto alarm = rpi_rt::create_stdout_alarm();
  std::thread th{[&alarm](){
    std::this_thread::sleep_for(std::chrono::seconds{1});
    auto res = std::make_unique<mock_detection_result>();
    alarm->report(std::move(res));
    std::this_thread::sleep_for(std::chrono::seconds{1});
    alarm->close();
  }};

  std::ostringstream oss;
  auto* old_buf = std::cout.rdbuf(oss.rdbuf());

  alarm->run();
  th.join();

  std::cout.rdbuf(old_buf);
  CHECK(oss.str().find("!!TEST!!") != std::string::npos);
}

TEST_CASE("MockTempSensor", "[system][sensor]") {
  auto sensor = rpi_rt::create_mock_temperature_sensor();
  size_t got_data = 0;
  sensor->set_celsius_reciever([&got_data](float){
    got_data++;
  });
  std::thread th{[&sensor](){
    std::this_thread::sleep_for(std::chrono::seconds{2});
    sensor->close();
  }};
  sensor->run();
  th.join();
  CHECK(got_data > 0);
}

TEST_CASE("MockCameraSensor", "[system][sensor][ffmpeg]") {
  auto sensor = rpi_rt::create_mock_camera_sensor(TESTDATA_PATH "/vid.mp4");
  size_t got_frame = 0;
  sensor->set_frame_callback([&got_frame](rpi_rt::Frame<uint8_t> frame) {
    CHECK(frame.height() > 0);
    CHECK(frame.width() > 0);
    CHECK(frame.channels() == 3);
    got_frame++;
  });
  std::thread th{[&sensor](){
    std::this_thread::sleep_for(std::chrono::seconds{2});
    sensor->close();
  }};
  sensor->run();
  th.join();
  CHECK(got_frame > 0);
}

