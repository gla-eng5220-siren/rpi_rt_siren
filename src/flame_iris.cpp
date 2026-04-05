#include "alarm.hpp"
#include "frame.hpp"
#include "sensor.hpp"
#include "logic.hpp"
#include "thread_actor.hpp"

#include "third_party/argparse.hpp"

#include <sstream>
#include <stdexcept>
#include <thread>
#include <iostream>

auto make_sensor_logic_thread(const argparse::ArgumentParser& program) {
  std::unique_ptr<rpi_rt::sensor_logic_thread_t> thread;
  if (program.present("--model") && program.present("--v4l2")) {
    auto sensor = rpi_rt::create_v4l2_camera_sensor(
        program.get<std::string>("--v4l2"));
    auto model = rpi_rt::create_shufflenet_model();
    model->setup(program.get<std::string>("--model"));
    auto logic = std::make_shared<rpi_rt::visual_classify_logic_t>();
    logic->logit_threshold(program.get<float>("--logit-threshold"));
    logic->model(model);
    auto v_thread = std::make_unique<
      rpi_rt::SensorLogicThread<
        rpi_rt::camera_sensor_t, rpi_rt::visual_classify_logic_t>>();
    v_thread->set_sensor(sensor);
    v_thread->set_logic(logic);
    thread = std::move(v_thread);
  } else if (program.present("--model") && program.present("--mock-cam")) {
    auto sensor = rpi_rt::create_mock_camera_sensor(
        program.get<std::string>("--mock-cam"));
    auto model = rpi_rt::create_shufflenet_model();
    model->setup(program.get<std::string>("--model"));
    auto logic = std::make_shared<rpi_rt::visual_classify_logic_t>();
    logic->logit_threshold(program.get<float>("--logit-threshold"));
    logic->model(model);
    auto v_thread = std::make_unique<
      rpi_rt::SensorLogicThread<
        rpi_rt::camera_sensor_t, rpi_rt::visual_classify_logic_t>>();
    v_thread->set_sensor(sensor);
    v_thread->set_logic(logic);
    thread = std::move(v_thread);
  } else if (program.get<bool>("--mock-temp")) {
    auto sensor = rpi_rt::create_mock_temperature_sensor();
    auto logic = std::make_shared<rpi_rt::temperature_threshold_logic_t>();
    logic->celsius_threshold(program.get<float>("--temp-threshold"));
    auto t_thread = std::make_unique<
      rpi_rt::SensorLogicThread<
        rpi_rt::temperature_sensor_t, rpi_rt::temperature_threshold_logic_t>>();
    t_thread->set_sensor(sensor);
    t_thread->set_logic(logic);
    thread = std::move(t_thread);
  } else {
    throw std::runtime_error("No valid sensor logic specified");
  }
  return thread;
}

auto make_alarm_thread(const argparse::ArgumentParser& program) {
  auto thread = std::make_unique<rpi_rt::alarm_thread_t>();
  if (program.get<bool>("--alarm-stdout")) {
    auto alarm = rpi_rt::create_stdout_alarm();
    thread->set_alarm(alarm);
  } else {
    throw std::runtime_error("No valid alarm specified");
  }
  return thread;
}

int main(int argc, char** argv) {
  argparse::ArgumentParser program("flame_iris");
  program.add_argument("--v4l2")
    .help("Path to v4l2 camera device (e.g. /dev/video0)");
  program.add_argument("--mock-cam")
    .help("Use ffmpeg to loop a video as mock camera sensor");
  program.add_argument("--model")
    .help("Path to shufflenet model dir (e.g. testdata/model)");
  program.add_argument("--mock-temp")
    .flag()
    .help("Use mocked temperature sensor");
  program.add_argument("--alarm-stdout")
    .help("Print alarm message to stdout")
    .flag();
  program.add_argument("--temp-threshold")
    .help("Temperature threshold in celsius degree")
    .default_value(200.0f)
    .scan<'g', float>();
  program.add_argument("--logit-threshold")
    .help("Logit threshold for vistual detection")
    .default_value(0.0f)
    .scan<'g', float>();

  program.parse_args(argc, argv);

  auto sensor_logic_thread = make_sensor_logic_thread(program);
  auto alarm_thread = make_alarm_thread(program);

  sensor_logic_thread->run();
  alarm_thread->run();

  sensor_logic_thread->set_detection_result_callback([&alarm_thread](
        std::unique_ptr<rpi_rt::detection_result_t> result) {
    alarm_thread->report(std::move(result));
  });

  for (;;) {
    std::this_thread::sleep_for(std::chrono::seconds{10});
  }

  sensor_logic_thread->close();
  alarm_thread->close();
  return 0;
}

