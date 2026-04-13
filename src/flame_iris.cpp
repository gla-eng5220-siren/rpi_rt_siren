#include "alarm.hpp"
#include "frame.hpp"
#include "http_server.hpp"
#include "sensor.hpp"
#include "logic.hpp"
#include "thread_actor.hpp"

#include "third_party/argparse.hpp"

#include <cerrno>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <iostream>
#include <unistd.h>
#include <signal.h>
#include <sys/signalfd.h>

static std::shared_ptr<rpi_rt::http_server_t> webui;

auto make_brevo_config(const argparse::ArgumentParser& program) {
  rpi_rt::brevo_config_t cfg;
  cfg.api_host = program.get<std::string>("--brevo-api-host");
  cfg.api_path = program.get<std::string>("--brevo-api-path");
  cfg.api_key = program.get<std::string>("--brevo-api-key");
  cfg.sender_name = program.get<std::string>("--brevo-sender-name");
  cfg.sender_email = program.get<std::string>("--brevo-sender-email");
  cfg.to_name = program.get<std::string>("--brevo-to-name");
  cfg.to_email = program.get<std::string>("--brevo-to-email");
  return cfg;
}

auto make_vision_logic(const argparse::ArgumentParser& program) {
  auto model = rpi_rt::create_shufflenet_model();
  model->setup(program.get<std::string>("--model"));
  auto logic = std::make_shared<rpi_rt::visual_classify_logic_t>();
  logic->logit_threshold(program.get<float>("--logit-threshold"));
  logic->model(model);
  auto v_thread = std::make_unique<
    rpi_rt::SensorLogicThread<
    rpi_rt::camera_sensor_t, rpi_rt::visual_classify_logic_t>>();
  return logic;
}

auto make_vision_thread(
    std::shared_ptr<rpi_rt::camera_sensor_t> sensor,
    std::shared_ptr<rpi_rt::visual_classify_logic_t> logic) {
  auto v_thread = std::make_unique<
    rpi_rt::SensorLogicThread<
    rpi_rt::camera_sensor_t, rpi_rt::visual_classify_logic_t>>();
  v_thread->set_sensor(sensor);
  v_thread->set_logic(logic);
  if (webui) {
    v_thread->set_http_server(webui);
  }
  return v_thread;
}

auto make_sensor_logic_thread(const argparse::ArgumentParser& program) {
  std::unique_ptr<rpi_rt::sensor_logic_thread_t> thread;
  if (program.present("--model") && program.present<int>("--libcamera")) {
    auto sensor = rpi_rt::create_libcamera_sensor(
        program.get<int>("--libcamera"));
    auto logic = make_vision_logic(program);
    thread = make_vision_thread(sensor, logic);
  } else if (program.present("--model") && program.present("--v4l2")) {
    auto sensor = rpi_rt::create_v4l2_camera_sensor(
        program.get<std::string>("--v4l2"));
    auto logic = make_vision_logic(program);
    thread = make_vision_thread(sensor, logic);
  } else if (program.present("--model") && program.present("--mock-cam")) {
    auto sensor = rpi_rt::create_mock_camera_sensor(
        program.get<std::string>("--mock-cam"));
    auto logic = make_vision_logic(program);
    thread = make_vision_thread(sensor, logic);
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
  if (program.present("--brevo-api-key")) {
    auto cfg = make_brevo_config(program);
    auto alarm = rpi_rt::create_brevo_email_alarm(std::move(cfg));
    thread->set_alarm(alarm);
  } else if (program.get<bool>("--alarm-stdout")) {
    auto alarm = rpi_rt::create_stdout_alarm();
    thread->set_alarm(alarm);
  } else if (program.present<int>("--buzzer")) {
    auto alarm = rpi_rt::create_buzzer_alarm(program.get<int>("--buzzer"));
    thread->set_alarm(alarm);
  } else {
    throw std::runtime_error("No valid alarm specified");
  }
  return thread;
}

int main(int argc, char** argv) {
  sigset_t mask;
  sigemptyset(&mask);
  sigaddset(&mask, SIGINT);
  sigaddset(&mask, SIGQUIT);

  if (sigprocmask(SIG_BLOCK, &mask, NULL) < 0)
    throw std::system_error(std::make_error_code(std::errc(errno)));

  argparse::ArgumentParser program("flame_iris");
  program.add_argument("--libcamera")
    .help("libcamera camera index (e.g. 0)")
    .scan<'i', int>();
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
  program.add_argument("--buzzer")
    .help("GPIO buzzer pin")
    .scan<'i', int>();
  program.add_argument("--temp-threshold")
    .help("Temperature threshold in celsius degree")
    .default_value(200.0f)
    .scan<'g', float>();
  program.add_argument("--logit-threshold")
    .help("Logit threshold for vistual detection")
    .default_value(0.0f)
    .scan<'g', float>();
  program.add_argument("--webui-path")
    .help("Path to webui static files (e.g. webui)");
  program.add_argument("--webui-host")
    .default_value("127.0.0.1")
    .help("Webui listening host");
  program.add_argument("--webui-port")
    .scan<'i', int>()
    .default_value(8383)
    .help("Webui listening port");

  // brevo email alarming
  program.add_argument("--brevo-api-host")
    .default_value("api.brevo.com")
    .help("Brevo email alarming API host");
  program.add_argument("--brevo-api-path")
    .default_value("/v3/smtp/email")
    .help("Brevo email alarming API path");
  program.add_argument("--brevo-api-key")
    .help("Brevo email alarming API key");
  program.add_argument("--brevo-sender-name")
    .default_value("FlameIris Webhook")
    .help("Brevo email alarming sender name");
  program.add_argument("--brevo-sender-email")
    .help("Brevo email alarming sender email");
  program.add_argument("--brevo-to-name")
    .default_value("FlameIris User")
    .help("Brevo email alarming receiver name");
  program.add_argument("--brevo-to-email")
    .help("Brevo email alarming receiver email");

  program.parse_args(argc, argv);

  if (program.present("--webui-path")) {
    webui = rpi_rt::create_http_server();
    webui->setup(program.get<std::string>("--webui-path"));
  }

  auto sensor_logic_thread = make_sensor_logic_thread(program);
  auto alarm_thread = make_alarm_thread(program);

  sensor_logic_thread->set_detection_result_callback([&alarm_thread](
        std::unique_ptr<rpi_rt::detection_result_t> result) {
    alarm_thread->report(std::move(result));
  });

  std::thread webui_thread;
  if (webui) {
    auto host = program.get<std::string>("--webui-host");
    auto port = program.get<int>("--webui-port");
    webui_thread = std::thread([self = webui, host, port](){
      std::cout << "[WebUI] Listening on http://" << host << ":" << port << std::endl;
      self->run(host, port);
    });
  }

  sensor_logic_thread->run();
  alarm_thread->run();

  int sfd = signalfd(-1, &mask, 0);
  if (sfd < 0)
    throw std::system_error(std::make_error_code(std::errc(errno)));

  bool running = true;
  while (running) {
    signalfd_siginfo fdsi;
    ssize_t sz = read(sfd, &fdsi, sizeof(fdsi));
    if (sz < 0)
      throw std::system_error(std::make_error_code(std::errc(errno)));

    switch (fdsi.ssi_signo) {
      case SIGINT: // fallthrough
      case SIGQUIT:
        std::cout << "Gracefully exitting .." << std::endl;
        running = false;
        break;
    }
  }

  sensor_logic_thread->close();
  alarm_thread->close();

  if (webui) {
    webui->close();
    webui_thread.join();
  }

  std::cout << "Bye!" << std::endl;
  return 0;
}

