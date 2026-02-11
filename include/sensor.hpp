#pragma once

#include <memory>
#include <functional>

namespace rpi_rt {
  class sensor_t {
    public:
      virtual ~sensor_t() {}
      virtual void run() = 0;
      virtual void close() = 0;
  };

  class temperature_sensor_t : public sensor_t {
    public:
      virtual ~temperature_sensor_t() {}
      virtual void set_celsius_reciever(std::function<void (float)> callback) = 0;
  };

  class camera_sensor_t : public sensor_t {
    public:
      virtual ~camera_sensor_t() {}
      // TODO: virtual void set_frame_callback(std::function<void (frame_t)> callback) = 0;
  };

  std::shared_ptr<temperature_sensor_t> create_mock_temperature_sensor();
  std::shared_ptr<camera_sensor_t> create_mock_camera_sensor();
}

