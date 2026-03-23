#include "frame.hpp"
#include "sensor.hpp"
#include "logic.hpp"
#include "http_server.hpp"

#include <sstream>
#include <thread>
#include <iostream>

int main(void) {
  // auto sensor = rpi_rt::create_v4l2_camera_sensor();
  auto sensor = rpi_rt::create_mock_camera_sensor("vid.mp4");

  const char* model_path_ptr = std::getenv("MODEL_PATH");
  std::string model_path{"model"};
  if (model_path_ptr)
    model_path = std::string(model_path_ptr);

  const char* cockpit_path_ptr = std::getenv("COCKPIT_PATH");
  std::string cockpit_path{"cockpit"};
  if (cockpit_path_ptr)
    cockpit_path = std::string(cockpit_path_ptr);

  auto model = rpi_rt::create_shufflenet_model();
  model->setup(model_path);

  auto server = rpi_rt::create_http_server();
  server->setup(cockpit_path);

  sensor->set_frame_callback([&model, &server](rpi_rt::Frame<uint8_t> frame) {
    server->set_cam_frame(frame);

    float result = model->process(frame);
    std::cout << "Result : " << result;

    server->set_logit(result);

    if (result > 0) {
      std::cout << "(NO FIRE)" << std::endl;
    } else {
      std::cout << "(FIRE)" << std::endl;
    }
  });

  std::thread sensor_thread([self = sensor](){
    self->run();
  });

  std::thread server_thread([self = server](){
    self->run("0.0.0.0", 8080);
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

