#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <iterator>

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

    private:
      float logit_;
      float logit_threshold_;
  };

  void visual_classify_logic_t::process(const Frame<uint8_t>& frame) {
    float logit = model_->process(frame);
    visual_detection_result result{logit, logit_threshold_};
    callback_(result);
  }
}

