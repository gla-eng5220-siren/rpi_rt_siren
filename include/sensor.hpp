#pragma once

#include <memory>
#include <functional>
#include <cstdint>

#include "frame.hpp"

namespace rpi_rt {

/** \addtogroup Sensor
 *  @{
 */

  /**
   * The base class for all sensors.
   *
   * Adhere to the Liskov Substitution Principle (LSP) and Interface Segregation
   * Principle (ISP) in SOLID.
   */
  class sensor_t {
    public:
      virtual ~sensor_t() {}

      /**
       * Start the sensor process.
       *
       * Blocks until close, so should run in a thread.
       */
      virtual void run() = 0;

      /**
       * Stop the run function and return.
       */
      virtual void close() = 0;
  };

  struct breadpi_ntc_config_t{
    std::string i2c_bus = "/dev/i2c-1";
    unsigned short i2c_addr = 0x48;
    int adc_channel = 1;

    float vref = 3.3f;
    float r_fixed = 10000.0f;
    float r_25 = 10000.0f;
    float beta = 3435.0f;
    bool ntc_top = false;
  };

  /**
   * The base class for all temperature sensors.
   *
   * Adhere to the Liskov Substitution Principle (LSP) and Interface Segregation
   * Principle (ISP) in SOLID.
   */
  class temperature_sensor_t : public sensor_t {
    public:
      virtual ~temperature_sensor_t() {}

      /**
       * Sets the callback for gathered temperature data.
       *
       * Adhere to the Interface Segregation Principle (ISP) in SOLID.
       */
      virtual void set_celsius_reciever(std::function<void (float)> callback) = 0;
  };

  /**
   * The base class for all camera sensors.
   *
   * Adhere to the Liskov Substitution Principle (LSP) and Interface Segregation
   * Principle (ISP) in SOLID.
   */
  class camera_sensor_t : public sensor_t {
    public:
      virtual ~camera_sensor_t() {}

      /**
       * Sets the callback for gathered camera image frames.
       *
       * Adhere to the Interface Segregation Principle (ISP) in SOLID.
       */
      virtual void set_frame_callback(std::function<void (Frame<uint8_t>)> callback) = 0;
  };

  /**
   * The factory method for creating a temperature_sensor_t reporting mock data.
   *
   * Adhere to the Interface Segregation Principle (ISP) in SOLID.
   */
  std::shared_ptr<temperature_sensor_t> create_mock_temperature_sensor();
  std::shared_ptr<temperature_sensor_t> create_breadpi_temperature_sensor(const breadpi_ntc_config_t& cfg);
  /**
   * The factory method for creating a camera_sensor_t reporting mock data.
   *
   * Decodes the given video into frames and report them.
   *
   * @param filename the video file
   *
   * Adhere to the Interface Segregation Principle (ISP) in SOLID.
   */
  std::shared_ptr<camera_sensor_t> create_mock_camera_sensor(const std::string& filename);

  /**
   * The factory method for creating a camera_sensor_t from V4L2 cameras.
   *
   * @param device a V4L2 camera device like /dev/video0
   *
   * Adhere to the Interface Segregation Principle (ISP) in SOLID.
   */
  std::shared_ptr<camera_sensor_t> create_v4l2_camera_sensor(const std::string& device);

  /**
   * The factory method for creating a camera_sensor_t from libcamera.
   *
   * @param cam_index The index of camera enumerated by libcamera
   *
   * Adhere to the Interface Segregation Principle (ISP) in SOLID.
   */
  std::shared_ptr<camera_sensor_t> create_libcamera_sensor(unsigned cam_index);

/** @}*/

}

