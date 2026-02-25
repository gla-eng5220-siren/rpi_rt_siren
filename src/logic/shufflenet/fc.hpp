#pragma once

#include <stdexcept>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cassert>
#include <optional>

#include "xnnpack.h"
#include "xnn_common.hpp"
#include "frame.hpp"

namespace rpi_rt::logic::shufflenet {

/*
 Fully Connected layer (also known as a Dense or Linear layer). It is one of the most
 fundamental building blocks in neural networks.

 Assumes NHWC compact layout.
 */
template <class Elem>
class Fc {
public:
  using elem_t = Elem;
  static_assert(std::is_same_v<Elem, float>, "Only F32 is implemented for Fc");

  // assumes (output_feature, input_feature) shape
  class Params {
  public:
    Params() {}

    Params(size_t output_feature, size_t input_feature) {
      resize(output_feature, input_feature);
    }

    void resize(size_t output_feature, size_t input_feature) {
      output_feature_ = output_feature;
      input_feature_ = input_feature;
      buffer_.resize(size());
      if (bias_.has_value()) {
        bias_->resize(output_feature);
      }
    }

    size_t output_feature() const noexcept {
      return output_feature_;
    }

    size_t input_feature() const noexcept {
      return input_feature_;
    }

    size_t size() const noexcept {
      return output_feature_ * input_feature_;
    }

    const Elem* data() const noexcept {
      return buffer_.data();
    }

    Elem* data() noexcept {
      return buffer_.data();
    }

    void add_bias() {
      if (! bias_.has_value()) {
        bias_.emplace(output_feature_);
      }
    }

    void remove_bias() {
      if (bias_.has_value()) {
        bias_ = std::nullopt;
      }
    }

    bool has_bias() const noexcept {
      return bias_.has_value();
    }

    Bias<elem_t>& bias() noexcept {
      return *bias_;
    }

    const Bias<elem_t>& bias() const noexcept {
      return *bias_;
    }

  private:
    size_t output_feature_ = 0;
    size_t input_feature_ = 0;
    std::vector<elem_t> buffer_;

    std::optional<Bias<elem_t>> bias_ = std::nullopt;
  };

  Fc() {}
  ~Fc() {
    xnn_delete_operator(fc_op_);
  }

  Fc(const Fc&) = delete;
  Fc(Fc&&) = delete;
  Fc& operator=(const Fc&) = delete;
  Fc& operator=(Fc&&) = delete;

  void setup(const Frame<float>& input, Frame<float>& output, const Params& params) {
    (void)XNNPackGuard::instance();

    assert(input.channels() == params.input_feature());
    assert(output.channels() == params.output_feature());
    assert(input.height() == 1);
    assert(input.width() == 1);
    assert(output.height() == 1);
    assert(output.width() == 1);

    xnn_status status;

    status = xnn_create_fully_connected_nc_f32(
        params.input_feature(),
        params.output_feature(),
        params.input_feature(),
        params.output_feature(),
        params.data(),
        params.has_bias() ? params.bias().data() : nullptr,
        -INFINITY, INFINITY,
        0,
        nullptr,
        &fc_op_);
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_create_fully_connected_nc_f32");
    }

    status = xnn_reshape_fully_connected_nc_f32(
        fc_op_,
        1,
        nullptr);
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_reshape_fully_connected_nc_f32");
    }

    status = xnn_setup_fully_connected_nc_f32(
        fc_op_,
        input.data(),
        output.data());
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_setup_fully_connected_nc_f32");
    }
  }

  void forward() {
    (void)XNNPackGuard::instance();

    xnn_status status;
    status = xnn_run_operator(fc_op_, nullptr);
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_run_operator(convolution2d_nhwc)");
    }
  }

private:
  xnn_operator_t fc_op_ = nullptr;
};

}

