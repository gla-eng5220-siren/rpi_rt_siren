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

template <class Elem>
class DepthwiseConv2D {
public:
  using elem_t = Elem;
  static_assert(std::is_same_v<Elem, float>, "Only F32 is implemented for DepthwiseConv2D");

  // assumes (channels, kernel_height, kernel_width) shape
  class Params {
  public:
    Params() {}

    Params(size_t channels, size_t kernel_height, size_t kernel_width) {
      resize(channels, kernel_height, kernel_width);
    }

    void resize(size_t channels, size_t kernel_height, size_t kernel_width) {
      channels_ = channels;
      kernel_width_ = kernel_width;
      kernel_height_ = kernel_height;
      buffer_.resize(size());
      if (bias_.has_value()) {
        bias_->resize(channels_);
      }
    }

    size_t channels() const noexcept {
      return channels_;
    }

    size_t kernel_width() const noexcept {
      return kernel_width_;
    }

    size_t kernel_height() const noexcept {
      return kernel_height_;
    }

    size_t size() const noexcept {
      return channels() * kernel_width() * kernel_height();
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
        bias_.emplace(channels_);
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
    size_t channels_ = 0;
    size_t kernel_height_ = 0;
    size_t kernel_width_ = 0;
    std::vector<elem_t> buffer_;

    size_t stride_height_ = 1;
    size_t stride_width_ = 1;
    size_t padding_height_ = 0;
    size_t padding_width_ = 0;

    std::optional<Bias<elem_t>> bias_ = std::nullopt;
  };

  DepthwiseConv2D() {}
  ~DepthwiseConv2D() {
    xnn_delete_operator(conv_op_);
  }

  DepthwiseConv2D(const DepthwiseConv2D&) = delete;
  DepthwiseConv2D(DepthwiseConv2D&&) = delete;
  DepthwiseConv2D& operator=(const DepthwiseConv2D&) = delete;
  DepthwiseConv2D& operator=(DepthwiseConv2D&&) = delete;

  void setup(const Frame<float>& input, Frame<float>& output, const Params& params) {
    (void)XNNPackGuard::instance();

    assert(input.channels() == output.channels());
    assert(input.channels() == params.channels());

    xnn_status status;

    status = xnn_create_convolution2d_nhwc_f32(
        params.padding_height(), params.padding_width(),
        params.padding_height(), params.padding_width(),
        params.kernel_height(), params.kernel_width(),
        params.stride_height(), params.stride_width(),
        1, 1,
        params.channels(),
        1,
        1,
        params.channels(),
        params.channels(),
        params.data(),
        params.has_bias() ? params.bias().data() : nullptr,
        -INFINITY, INFINITY,
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

