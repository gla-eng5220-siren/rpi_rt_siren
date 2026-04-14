#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <iterator>
#include <sstream>
#include <optional>

#include "frame.hpp"
#include "logic.hpp"

namespace rpi_rt {
  class visual_detection_result : public detection_result_t {
    public:
      visual_detection_result(float logit, float logit_threshold)
        : logit_(logit), logit_threshold_(logit_threshold)
      {}
      visual_detection_result(float logit, float logit_threshold, Frame<uint8_t> frame, uint64_t frame_id)
        : logit_(logit), logit_threshold_(logit_threshold), frame_(frame), frame_id_(frame_id)
      {}
      virtual ~visual_detection_result() {}

      virtual bool has_fire() override {
        return logit_ < logit_threshold_;
      }

      virtual std::string explain() override {
        std::ostringstream oss;
        oss << "Visual LOGIT: " << logit_
          << " THRESHOLD: " << logit_threshold_
          << (has_fire() ? " [FIRE]" : " [NO FIRE]");
        return oss.str();
      }

      virtual std::vector<uint8_t> jpg_attachment() override {
        if (frame_) {
          std::vector<uint8_t> jpg_data;
          jpg_data = jpeg_utils::write_to_mem(*frame_);
          return jpg_data;
        }
        return {};
      }

      virtual uint64_t frame_id() const noexcept override {
        return frame_id_;
      }

    private:
      float logit_;
      float logit_threshold_;
      uint64_t frame_id_ = 0;
      std::optional<Frame<uint8_t>> frame_ = std::nullopt;
  };

  void visual_classify_logic_t::process(uint64_t frame_id, const Frame<uint8_t>& frame) {
    float logit = model_->process(frame);
    auto result = std::make_unique<visual_detection_result>(
        logit, logit_threshold_, frame, frame_id);
    last_logit_ = logit;
    callback_(std::move(result));
  }
}

