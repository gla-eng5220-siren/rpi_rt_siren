#include "alarm.hpp"
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

  rpi_rt::visual_classify_logic_t logic;
  logic.logit_threshold(0.0f);
  logic.model(model);

  auto alarm = rpi_rt::create_stdout_alarm();

  logic.set_detection_result_callback([&alarm](rpi_rt::detection_result_t& result) {
    if (result.has_fire()) {
      alarm->report_fire();
    }
  });

  sensor->set_frame_callback([&logic](rpi_rt::Frame<uint8_t> frame) {
    logic.process(frame);
  });

  std::thread sensor_thread([self = sensor](){
    self->run();
  });
  std::thread alarm_thread([self = alarm](){
    self->run();
  });

  for (;;) {
    std::this_thread::sleep_for(std::chrono::seconds{10});
  }

  sensor->close();
  alarm->close();
  sensor_thread.join();
  alarm_thread.join();
  return 0;
}

