#include "frame.hpp"
#include "sensor.hpp"
#include "logic.hpp"

#include <sstream>
#include <thread>
#include <iostream>

int main(void) {
  auto sensor = rpi_rt::create_v4l2_camera_sensor();

  const char* model_path_ptr = std::getenv("MODEL_PATH");
  std::string model_path{"model"};
  if (model_path_ptr)
    model_path = std::string(model_path_ptr);
  auto model = rpi_rt::create_shufflenet_model();
  model->setup(model_path);

  sensor->set_frame_callback([&model](rpi_rt::Frame<uint8_t> frame) {
    float result = model->process(frame);
    std::cout << "Result : " << result;

    if (result > 0) {
      std::cout << "(NO FIRE)" << std::endl;
    } else {
      std::cout << "(FIRE)" << std::endl;
    }
  });

  std::thread sensor_thread([self = sensor](){
    self->run();
  });

  for (;;) {
    std::this_thread::sleep_for(std::chrono::seconds{10});
  }

  /*
  sensor->close();
  sensor_thread.join();
  */
  return 0;
}

