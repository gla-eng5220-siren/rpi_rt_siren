#include "frame.hpp"
#include "sensor.hpp"

#include <sstream>
#include <thread>
#include <iostream>

int main(void) {
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

  rpi_rt::jpeg_utils::write_to_file(last_frame, "output.jpg");
  return 0;
}

