#include <chrono>
#include <functional>
#include <memory>
#include <thread>
#include <atomic>

#include "sensor.hpp"

namespace rpi_rt {
  class mock_camera_sensor_t : public camera_sensor_t {
    public:
      virtual ~mock_camera_sensor_t() override {}

      virtual void run() override {
        size_t i_mock = 0;
        while (!closing_) {
          std::this_thread::sleep_for(std::chrono::milliseconds{500});
          report_celsius_(mock_data[i_mock++]);
          i_mock %= mock_data.size();
        }
      }

      virtual void close() override {
        closing_ = true;
      }

    private:
      std::atomic<bool> closing_ = ATOMIC_VAR_INIT(false);
  };

  std::shared_ptr<camera_sensor_t> create_mock_camera_sensor() {
    return std::make_shared<mock_camera_sensor_t>();
  }
}


