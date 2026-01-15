#include <chrono>
#include <functional>
#include <memory>
#include <thread>
#include <atomic>

#include "sensor.hpp"

namespace rpi_rt {
  class mock_temperature_sensor_t : public temperature_sensor_t {
    public:
      virtual ~mock_temperature_sensor_t() override {}

      virtual void run() override {
        std::array<float, 5> mock_data = {
          20.0f,
          22.0f,
          316.0f,
          18.0f,
          28.0f,
        };
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

      virtual void set_celsius_reciever(std::function<void (float)> callback) override {
        report_celsius_ = callback;
      }

    private:
      std::function<void (float)> report_celsius_;
      std::atomic<bool> closing_ = ATOMIC_VAR_INIT(false);
  };

  std::shared_ptr<temperature_sensor_t> create_mock_temperature_sensor() {
    return std::make_shared<mock_temperature_sensor_t>();
  }
}


