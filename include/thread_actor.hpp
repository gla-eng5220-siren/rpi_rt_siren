#pragma once

#include <memory>
#include <functional>
#include <thread>

#include "detection_result.hpp"
#include "logic.hpp"
#include "sensor.hpp"

namespace rpi_rt {
  namespace detail {
    void sensor_logic_setup_impl(
      std::shared_ptr<camera_sensor_t> s,
      std::shared_ptr<visual_classify_logic_t> l
    ) {
      s->set_frame_callback([l](rpi_rt::Frame<uint8_t> frame) {
        l->process(frame);
      });
    }

    void sensor_logic_setup_impl(
      std::shared_ptr<temperature_sensor_t> s,
      std::shared_ptr<temperature_threshold_logic_t> l
    ) {
      s->set_celsius_reciever([l](float celsius) {
        l->process(celsius);
      });
    }
  }

  template <class Sensor, class Logic>
  class SensorLogicThread {
    public:
      SensorLogicThread() {};
      SensorLogicThread(const SensorLogicThread&) = delete;
      SensorLogicThread(SensorLogicThread&&) = delete;
      SensorLogicThread& operator=(const SensorLogicThread&) = delete;
      SensorLogicThread& operator=(SensorLogicThread&&) = delete;

      void run() {
        detail::sensor_logic_setup_impl(sensor_, logic_);
        thread_ = std::thread([sensor = sensor_](){
          sensor->run();
        });
      }

      void close() {
        sensor_->close();
        thread_.join();
      }

      void set_sensor(std::shared_ptr<Sensor> sensor) {
        sensor_ = sensor;
      }
      void set_logic(std::shared_ptr<Logic> logic) {
        logic_ = logic;
      }

      void set_detection_result_callback(
          std::function<void (std::unique_ptr<detection_result_t>)> callback) {
        logic_->set_detection_result_callback(callback);
      }

    private:
      std::shared_ptr<Sensor> sensor_;
      std::shared_ptr<Logic> logic_;
      std::thread thread_;
  };

  class AlarmThread {
    public:
      AlarmThread() {};
      AlarmThread(const AlarmThread&) = delete;
      AlarmThread(AlarmThread&&) = delete;
      AlarmThread& operator=(const AlarmThread&) = delete;
      AlarmThread& operator=(AlarmThread&&) = delete;

      void run() {
        thread_ = std::thread([alarm = alarm_](){
          alarm->run();
        });
      }

      void set_alarm(std::shared_ptr<alarm_t> alarm) {
        alarm_ = alarm;
      }

      void close() {
        alarm_->close();
        thread_.join();
      }

      void report(std::unique_ptr<detection_result_t> result) {
        alarm_->report(std::move(result));
      }

    private:
      std::shared_ptr<alarm_t> alarm_;
      std::thread thread_;
  };
}


