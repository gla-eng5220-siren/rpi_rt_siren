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
class Maxpool2D {
public:
  using elem_t = Elem;
  static_assert(std::is_same_v<Elem, float>, "Only F32 is implemented for Maxpool2D");

  // assumes (channels, kernel_height, kernel_width) shape
  class Params {
  public:
    Params() {}
 
    void width(size_t width) noexcept {
      width_ = width;
    }

    size_t width() const noexcept {
      return width_;
    }

    void height(size_t height) noexcept {
      height_ = height;
    }

    size_t height() const noexcept {
      return height_;
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
 
    void dilation_width(size_t dilation_width) noexcept {
      dilation_width_ = dilation_width;
    }

    size_t dilation_width() const noexcept {
      return dilation_width_;
    }

    void dilation_height(size_t dilation_height) noexcept {
      dilation_height_ = dilation_height;
    }

    size_t dilation_height() const noexcept {
      return dilation_height_;
    }

  private:
    size_t height_ = 1;
    size_t width_ = 1;
    size_t stride_height_ = 1;
    size_t stride_width_ = 1;
    size_t padding_height_ = 0;
    size_t padding_width_ = 0;
    size_t dilation_height_ = 1;
    size_t dilation_width_ = 1;
  };

  Maxpool2D() {}
  ~Maxpool2D() {
    xnn_delete_operator(maxpool_op_);
  }

  Maxpool2D(const Maxpool2D&) = delete;
  Maxpool2D(Maxpool2D&&) = delete;
  Maxpool2D& operator=(const Maxpool2D&) = delete;
  Maxpool2D& operator=(Maxpool2D&&) = delete;

  void setup(const Frame<float>& input, Frame<float>& output, const Params& params) {
    (void)XNNPackGuard::instance();

    assert(input.channels() == output.channels());

    xnn_status status;

    status = xnn_create_max_pooling2d_nhwc_f32(
        params.padding_height(), params.padding_width(),
        params.padding_height(), params.padding_width(),
        params.height(), params.width(),
        params.stride_height(), params.stride_width(),
        params.dilation_height(), params.dilation_width(),
        -INFINITY, INFINITY,
        0,
        &maxpool_op_);
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_create_max_pooling2d_nhwc_f32");
    }

    size_t output_height, output_width;
    status = xnn_reshape_max_pooling2d_nhwc_f32(
        maxpool_op_,
        1,
        input.height(),
        input.width(),
        input.channels(),
        input.channels(),
        output.channels(),
        &output_height,
        &output_width,
        nullptr);
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_reshape_max_pooling2d_nhwc_f32");
    }

    assert(output_height == output.height());
    assert(output_width == output.width());

    status = xnn_setup_max_pooling2d_nhwc_f32(
        maxpool_op_,
        input.data(),
        output.data());
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_setup_max_pooling2d_nhwc_f32");
    }
  }

  void forward() {
    (void)XNNPackGuard::instance();

    xnn_status status;
    status = xnn_run_operator(maxpool_op_, nullptr);
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_run_operator(max_pooling2d_nhwc)");
    }
  }

private:
  xnn_operator_t maxpool_op_ = nullptr;
};

}

