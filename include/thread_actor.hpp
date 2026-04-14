#pragma once

#include <memory>
#include <functional>
#include <thread>

#include "detection_result.hpp"
#include "logic.hpp"
#include "sensor.hpp"
#include "http_server.hpp"

namespace rpi_rt {

/** \addtogroup Threads
 *  @{
 */

/// @cond PRIVATE_DETAILS

  namespace detail {
    void sensor_logic_setup_impl(
      std::shared_ptr<camera_sensor_t> s,
      std::shared_ptr<visual_classify_logic_t> l,
      std::shared_ptr<http_server_t> webui
    ) {
      s->set_frame_callback([l, webui](uint64_t frame_id, rpi_rt::Frame<uint8_t> frame) {
        if (webui) {
          webui->set_cam_frame(frame);
        }
        l->process(frame_id, frame);
        if (webui) {
          webui->set_logit(l->last_logit());
        }
      });
    }

    void sensor_logic_setup_impl(
      std::shared_ptr<temperature_sensor_t> s,
      std::shared_ptr<temperature_threshold_logic_t> l,
      std::shared_ptr<http_server_t> webui
    ) {
      (void) webui; // TODO report this too maybe?
      s->set_celsius_reciever([l](uint64_t frame_id, float celsius) {
        l->process(frame_id, celsius);
      });
    }
  }

/// @endcond

  /**
   * The base class for all sensor-and-logic threads.
   *
   * Adhere to the Liskov Substitution Principle (LSP) and Interface Segregation
   * Principle (ISP) in SOLID.
   */
  class sensor_logic_thread_t {
    public:
      virtual ~sensor_logic_thread_t() {}

      /**
       * Sets the callback for detecting results.
       *
       * Adhere to the Interface Segregation Principle (ISP) in SOLID.
       */
      virtual void set_detection_result_callback(
          std::function<void (std::unique_ptr<detection_result_t>)> callback) = 0;

      /**
       * Starts the thread.
       */
      virtual void run() = 0;

      /**
       * Stops the thread and wait for its join.
       */
      virtual void close() = 0;
  };

  /**
   * Provide the implementation for each valid Sensor-Logic pairs.
   */
  template <class Sensor, class Logic>
  class SensorLogicThread : public sensor_logic_thread_t {
    public:
      SensorLogicThread() {};
      virtual ~SensorLogicThread() {}
      SensorLogicThread(const SensorLogicThread&) = delete;
      SensorLogicThread(SensorLogicThread&&) = delete;
      SensorLogicThread& operator=(const SensorLogicThread&) = delete;
      SensorLogicThread& operator=(SensorLogicThread&&) = delete;

      /**
       * Starts the thread.
       */
      void run() {
        detail::sensor_logic_setup_impl(sensor_, logic_, http_server_);
        thread_ = std::thread([sensor = sensor_](){
          sensor->run();
        });
      }

      /**
       * Stops the thread and wait for its join.
       */
      void close() {
        sensor_->close();
        thread_.join();
      }

      /**
       * Sets the sensor.
       */
      void set_sensor(std::shared_ptr<Sensor> sensor) {
        sensor_ = sensor;
      }

      /**
       * Sets the logic.
       */
      void set_logic(std::shared_ptr<Logic> logic) {
        logic_ = logic;
      }

      /**
       * Optionally report data to WebUI http server.
       */
      void set_http_server(std::shared_ptr<http_server_t> http_server) {
        http_server_ = http_server;
      }

      /**
       * Sets the callback for detecting results.
       *
       * Adhere to the Interface Segregation Principle (ISP) in SOLID.
       */
      virtual void set_detection_result_callback(
          std::function<void (std::unique_ptr<detection_result_t>)> callback) override {
        logic_->set_detection_result_callback(callback);
      }

    private:
      std::shared_ptr<Sensor> sensor_;
      std::shared_ptr<Logic> logic_;
      std::thread thread_;

      // nullptr if no webui
      std::shared_ptr<http_server_t> http_server_;
  };

  class alarm_thread_t {
    public:
      alarm_thread_t() {};
      alarm_thread_t(const alarm_thread_t&) = delete;
      alarm_thread_t(alarm_thread_t&&) = delete;
      alarm_thread_t& operator=(const alarm_thread_t&) = delete;
      alarm_thread_t& operator=(alarm_thread_t&&) = delete;

      /**
       * Starts the thread.
       */
      void run() {
        thread_ = std::thread([alarm = alarm_](){
          alarm->run();
        });
      }

      /**
       * Sets the alarm.
       */
      void set_alarm(std::shared_ptr<alarm_t> alarm) {
        alarm_ = alarm;
      }

      /**
       * Stops the thread and wait for its join.
       */
      void close() {
        alarm_->close();
        thread_.join();
      }

      /**
       * Reports a detection result.
       *
       * Wakes up the run function to perform the actual alarming. Subclasses
       * might have custom throttle implemented.
       *
       * @param result The outcome reported by sensor and detection logic.
       */
      void report(std::unique_ptr<detection_result_t> result) {
        alarm_->report(std::move(result));
      }

    private:
      std::shared_ptr<alarm_t> alarm_;
      std::thread thread_;
  };

/** @}*/

}

