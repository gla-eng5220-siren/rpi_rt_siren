#pragma once

#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cassert>

#include "xnnpack.h"
#include "xnn_common.hpp"
#include "frame.hpp"

namespace rpi_rt::logic::shufflenet {

template <class Elem>
class Conv2D {
public:
  using elem_t = Elem;
  static_assert(std::is_same_v<Elem, float>, "Only F32 is implemented for Conv2D");

  // assumes (output_feature, kernel_height, kernel_width, input_feature) shape
  class Params {
  public:
    Params() {}

    Params(size_t output_feature, size_t kernel_height, size_t kernel_width, size_t input_feature) {
      resize(output_feature, kernel_height, kernel_width, input_feature);
    }

    void resize(size_t output_feature, size_t kernel_height, size_t kernel_width, size_t input_feature) {
      output_feature_ = output_feature;
      kernel_width_ = kernel_width;
      kernel_height_ = kernel_height;
      input_feature_ = input_feature;
      buffer_.resize(size());
      if (bias_.has_value()) {
        bias_->resize(output_feature_);
      }
    }

    size_t output_feature() const noexcept {
      return output_feature_;
    }

    size_t input_feature() const noexcept {
      return input_feature_;
    }

    size_t kernel_width() const noexcept {
      return kernel_width_;
    }

    size_t kernel_height() const noexcept {
      return kernel_height_;
    }

    size_t size() const noexcept {
      return output_feature() * kernel_width() * kernel_height() * input_feature();
    }

    const Elem* data() const noexcept {
      return buffer_.data();
    }

    Elem* data() noexcept {
      return buffer_.data();
    }

    void stride_width(size_t stride_width) noexcept {
      stride_width_ = stride_width;
    }

    size_t stride_width() const noexcept {
      return stride_width_;
    }

    void stride_height(size_t stride_height) noexcept {
      stride_height_ = stride_height;
    }

    size_t stride_height() const noexcept {
      return stride_height_;
    }
 
    void padding_width(size_t padding_width) noexcept {
      padding_width_ = padding_width;
    }

    size_t padding_width() const noexcept {
      return padding_width_;
    }

    void padding_height(size_t padding_height) noexcept {
      padding_height_ = padding_height;
    }

    size_t padding_height() const noexcept {
      return padding_height_;
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

    bool relu() const noexcept {
      return relu_;
    }

    void relu(bool fuse) noexcept {
      relu_ = fuse;
    }

  private:
    size_t output_feature_ = 0;
    size_t kernel_width_ = 0;
    size_t kernel_height_ = 0;
    size_t input_feature_ = 0;
    std::vector<elem_t> buffer_;

    size_t stride_width_ = 1;
    size_t stride_height_ = 1;
    size_t padding_width_ = 0;
    size_t padding_height_ = 0;

    std::optional<Bias<elem_t>> bias_ = std::nullopt;

    bool relu_ = false;
  };

  Conv2D() {}
  ~Conv2D() {
    xnn_delete_operator(conv_op_);
  }

  Conv2D(const Conv2D&) = delete;
  Conv2D(Conv2D&&) = delete;
  Conv2D& operator=(const Conv2D&) = delete;
  Conv2D& operator=(Conv2D&&) = delete;

  void setup(const Frame<float>& input, Frame<float>& output, const Params& params) {
    (void)XNNPackGuard::instance();

    assert(input.channels() == params.input_feature());
    assert(output.channels() == params.output_feature());

    xnn_status status;

    status = xnn_create_convolution2d_nhwc_f32(
        params.padding_height(), params.padding_width(),
        params.padding_height(), params.padding_width(),
        params.kernel_height(), params.kernel_width(),
        params.stride_height(), params.stride_width(),
        1, 1,
        1,
        params.input_feature(),
        params.output_feature(),
        params.input_feature(),
        params.output_feature(),
        params.data(),
        params.has_bias() ? params.bias().data() : nullptr,
        params.relu() ? 0 : -INFINITY, INFINITY,
        0,
        nullptr,
        &conv_op_);
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_create_convolution2d_nhwc_f32");
    }

    size_t workspace_size, output_height, output_width;
    status = xnn_reshape_convolution2d_nhwc_f32(
        conv_op_,
        1,
        input.height(),
        input.width(),
        &workspace_size,
        &output_height,
        &output_width,
        nullptr);
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_reshape_convolution2d_nhwc_f32");
    }

    assert(output_height == output.height());
    assert(output_width == output.width());

    status = xnn_setup_convolution2d_nhwc_f32(
        conv_op_,
        nullptr,
        input.data(),
        output.data());
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_setup_convolution2d_nhwc_f32");
    }
  }

  void forward() {
    (void)XNNPackGuard::instance();

    xnn_status status;
    status = xnn_run_operator(conv_op_, nullptr);
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_run_operator(convolution2d_nhwc)");
    }
  }

private:
  xnn_operator_t conv_op_ = nullptr;
};

}

