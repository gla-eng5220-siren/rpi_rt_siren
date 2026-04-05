#pragma once

#include <memory>

#include "frame.hpp"

namespace rpi_rt {
  class detection_result_t {
    public:
      virtual ~detection_result_t() {}
      virtual bool has_fire() = 0;
  };

  class visual_classfying_model_t {
    public:
      virtual ~visual_classfying_model_t() {}
      virtual void setup(const std::string& model_path) = 0;
      virtual float process(const Frame<uint8_t>& frame) = 0;
  };

 std::shared_ptr<visual_classfying_model_t> create_shufflenet_model();

  class visual_classify_logic_t {
    public:
      ~visual_classify_logic_t() {}

      float logit_threshold() const noexcept {
        return logit_threshold_;
      }
      void logit_threshold(float logit) noexcept {
        logit_threshold_ = logit;
      }

      std::shared_ptr<visual_classfying_model_t> model() const noexcept {
        return model_;
      }
      void model(std::shared_ptr<visual_classfying_model_t> m) noexcept {
        model_ = m;
      }

      void set_detection_result_callback(std::function<void (detection_result_t&)> callback) {
        callback_ = callback;
      }

      void process(const Frame<uint8_t>& frame);

    private:
      float logit_threshold_ = 0.0;
      std::shared_ptr<visual_classfying_model_t> model_;
      std::function<void (detection_result_t&)> callback_;
  };

  class temperature_threshold_logic_t {
    public:
      ~temperature_threshold_logic_t() {}

      float celsius_threshold() const noexcept {
        return celsius_threshold_;
      }
      void celsius_threshold(float celsius) noexcept {
        celsius_threshold_ = celsius;
      }

      void set_detection_result_callback(std::function<void (detection_result_t&)> callback) {
        callback_ = callback;
      }

      virtual void process(float);

    private:
      float celsius_threshold_ = 0.0;
      std::function<void (detection_result_t&)> callback_;
  };
}

