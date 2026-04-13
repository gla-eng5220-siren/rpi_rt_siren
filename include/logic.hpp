#pragma once

#include <memory>

#include "frame.hpp"
#include "detection_result.hpp"

namespace rpi_rt {

/** \addtogroup Logic
 *  @{
 */

  /**
   * The base class for Visual Classifying Models
   *
   * Adhere to the Liskov Substitution Principle (LSP) and Interface Segregation
   * Principle (ISP) in SOLID.
   */
  class visual_classfying_model_t {
    public:
      virtual ~visual_classfying_model_t() {}

      /**
       * Load the model from file system.
       */
      virtual void setup(const std::string& model_path) = 0;

      /**
       * Perform classification.
       *
       * @param frame The frame input. Should have 3 channels RGB.
       * @return The logit output. Use sigmoid to get the probability.
       */
      virtual float process(const Frame<uint8_t>& frame) = 0;
  };

  /**
   * The factory method for creating a ShuffleNetV2OnFire visual classification
   * model.
   *
   * The model has its Conv-BatchNorm-ReLU merged for best performance.
   * The model must be setup with data from PROJECT_ROOT/testdata/model
   *
   * Adhere to the Interface Segregation Principle (ISP) in SOLID.
   */
  std::shared_ptr<visual_classfying_model_t> create_shufflenet_model();

  /**
   * Implements the logic for visual classification.
   *
   * Adhere to the Single Responsibility Principle (SRP) in SOLID.
   */
  class visual_classify_logic_t {
    public:
      ~visual_classify_logic_t() {}

      /**
       * Returns the logic threshold.
       */
      float logit_threshold() const noexcept {
        return logit_threshold_;
      }

      /**
       * Sets the logic threshold.
       *
       * @param logit If logits are below this, it is considered a fire.
       */
      void logit_threshold(float logit) noexcept {
        logit_threshold_ = logit;
      }

      /**
       * The model to use.
       */
      std::shared_ptr<visual_classfying_model_t> model() const noexcept {
        return model_;
      }

      /**
       * Sets the model to use.
       *
       * @param m The vision classification model.
       */
      void model(std::shared_ptr<visual_classfying_model_t> m) noexcept {
        model_ = m;
      }

      /**
       * Sets the callback for detecting results.
       *
       * Adhere to the Interface Segregation Principle (ISP) in SOLID.
       */
      void set_detection_result_callback(std::function<void (std::unique_ptr<detection_result_t>)> callback) {
        callback_ = callback;
      }

      /**
       * Perform the actual detection.
       *
       * @param frame The frame input. Should have 3 channels RGB.
       */
      void process(const Frame<uint8_t>& frame);

      /**
       * Get the logit for last detection.
       */
      float last_logit() const noexcept {
        return last_logit_;
      }

    private:
      float logit_threshold_ = 0.0;
      float last_logit_ = 0.0;
      std::shared_ptr<visual_classfying_model_t> model_;
      std::function<void (std::unique_ptr<detection_result_t>)> callback_;
  };

  /**
   * Implements the logic for temperature-based fire detection.
   *
   * Adhere to the Single Responsibility Principle (SRP) in SOLID.
   */
  class temperature_threshold_logic_t {
    public:
      ~temperature_threshold_logic_t() {}

      /**
       * Returns the threshold, in celsius degree.
       */
      float celsius_threshold() const noexcept {
        return celsius_threshold_;
      }
      /**
       * Sets the threshold, in celsius degree.
       *
       * @param celsius If the temperature is higher than this, it is considered a fire.
       */
      void celsius_threshold(float celsius) noexcept {
        celsius_threshold_ = celsius;
      }

      /**
       * Sets the callback for detectin results.
       *
       * Adhere to the Interface Segregation Principle (ISP) in SOLID.
       */
      void set_detection_result_callback(std::function<void (std::unique_ptr<detection_result_t>)> callback) {
        callback_ = callback;
      }

      /**
       * Perform the actual detection.
       *
       * @param celsius The temperature.
       */
      void process(float celsius);

    private:
      float celsius_threshold_ = 0.0;
      std::function<void (std::unique_ptr<detection_result_t>)> callback_;
  };

/** @}*/

}

