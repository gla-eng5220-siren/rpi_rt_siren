#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <iterator>
#include <sstream>

#include "logic.hpp"

namespace rpi_rt {
  class visual_detection_result : public detection_result_t {
    public:
      explicit visual_detection_result(float logit, float logit_threshold)
        : logit_(logit), logit_threshold_(logit_threshold)
      {}
      virtual ~visual_detection_result() {}

      virtual bool has_fire() override {
        return logit_ < logit_threshold_;
      }

      virtual std::string explain() override {
        std::ostringstream oss;
        oss << "Visual LOGIT: " << logit_
          << "THRESHOLD: " << logit_threshold_
          << (has_fire() ? "[FIRE]" : "[NO FIRE]");
        return oss.str();
      }

    private:
      float logit_;
      float logit_threshold_;
  };

  void visual_classify_logic_t::process(const Frame<uint8_t>& frame) {
    float logit = model_->process(frame);
    auto result = std::make_unique<visual_detection_result>(
        logit, logit_threshold_);
    callback_(std::move(result));
  }
}

