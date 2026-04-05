#include "alarm.hpp"
#include "frame.hpp"
#include "sensor.hpp"
#include "logic.hpp"
#include "thread_actor.hpp"

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

  auto logic = std::make_shared<rpi_rt::visual_classify_logic_t>();
  logic->logit_threshold(0.0f);
  logic->model(model);

  auto alarm = rpi_rt::create_stdout_alarm();

  rpi_rt::SensorLogicThread<rpi_rt::camera_sensor_t, rpi_rt::visual_classify_logic_t>
    sensor_logic_thread;
  sensor_logic_thread.set_sensor(sensor);
  sensor_logic_thread.set_logic(logic);

  sensor_logic_thread.set_detection_result_callback([&alarm](std::unique_ptr<rpi_rt::detection_result_t> result) {
    alarm->report(std::move(result));
  });

  rpi_rt::AlarmThread alarm_thread;
  alarm_thread.set_alarm(alarm);

  sensor_logic_thread.run();
  alarm_thread.run();

  for (;;) {
    std::this_thread::sleep_for(std::chrono::seconds{10});
  }

  sensor_logic_thread.close();
  alarm_thread.close();
  return 0;
}

