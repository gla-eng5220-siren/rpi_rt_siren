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
      explicit temperature_threshold_result(float celsius, float threshold, uint64_t frame_id)
        : celsius_(celsius), threshold_(threshold), frame_id_(frame_id)
      {}
      virtual ~temperature_threshold_result() {}

      virtual bool has_fire() override {
        return celsius_ > threshold_;
      }

      virtual std::string explain() override {
        std::ostringstream oss;
        oss << "Temperature CELSIUS: " << celsius_
          << " THRESHOLD: " << threshold_
          << (has_fire() ? " [FIRE]" : " [NO FIRE]");
        return oss.str();
      }

      virtual uint64_t frame_id() const noexcept override {
        return frame_id_;
      }

    private:
      float celsius_;
      float threshold_;
      uint64_t frame_id_ = 0;
  };

  void temperature_threshold_logic_t::process(uint64_t frame_id, float celsius) {
    auto result = std::make_unique<temperature_threshold_result>(
        celsius, celsius_threshold_, frame_id);
    callback_(std::move(result));
  }
}

