#include <chrono>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <fstream>
#include <iterator>
#include <utility>

#include "logic.hpp"

namespace rpi_rt {
  class temperature_threshold_result : public detection_result_t {
    public:
      explicit temperature_threshold_result(float celsius, float threshold)
        : celsius_(celsius), threshold_(threshold)
      {}
      virtual ~temperature_threshold_result() {}

      virtual bool has_fire() override {
        return celsius_ > threshold_;
      }

      virtual std::string explain() override {
        std::ostringstream oss;
        oss << "Temperature CELSIUS: " << celsius_
          << " THRESHOLD: " << threshold_
          << (has_fire() ? "[FIRE]" : "[NO FIRE]");
        return oss.str();
      }

    private:
      float celsius_;
      float threshold_;
  };

  void temperature_threshold_logic_t::process(float celsius) {
    auto result = std::make_unique<temperature_threshold_result>(
        celsius, celsius_threshold_);
    callback_(std::move(result));
  }
}

