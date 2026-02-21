#include "frame.hpp"
#include "sensor.hpp"

#include <sstream>
#include <thread>
#include <iostream>

#define cimg_display 0
#include "CImg.h"

int main(void) {
  using cimg_library::CImg;

  auto sensor = rpi_rt::create_v4l2_camera_sensor();
  rpi_rt::Frame<uint8_t> last_frame;

  sensor->set_frame_callback([&last_frame](rpi_rt::Frame<uint8_t> frame) {
    std::cout << "." << std::flush;
    last_frame = std::move(frame);
  });

  std::thread sensor_thread([self = sensor](){
    self->run();
  });

  std::this_thread::sleep_for(std::chrono::seconds{5});

  sensor->close();
  sensor_thread.join();

  const auto& frame = last_frame;
  CImg<uint8_t> img(frame.data(), frame.width(), frame.height(), 1, 3);
  for (size_t i = 0; i < frame.height(); i++) {
    for (size_t j = 0; j < frame.width(); j++) {
      const uint8_t* pixel = frame.data() + (i * frame.width() + j) * 3;
      img(j, i, 0, 0) = pixel[0];
      img(j, i, 0, 1) = pixel[1];
      img(j, i, 0, 2) = pixel[2];
    }
  }
  img.save("output.png");
  return 0;
}

