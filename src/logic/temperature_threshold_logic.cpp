#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <iterator>

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

    private:
      float celsius_;
      float threshold_;
  };

  void temperature_threshold_logic_t::process(float celsius) {
    temperature_threshold_result result{celsius, celsius_threshold_};
    callback_(result);
  }
}

