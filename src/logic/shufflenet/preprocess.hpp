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

class Preprocess {
public:
  Preprocess() {}
  ~Preprocess() {
    xnn_delete_operator(resize_op_);
  }

  Preprocess(const Preprocess&) = delete;
  Preprocess(Preprocess&&) = delete;
  Preprocess& operator=(const Preprocess&) = delete;
  Preprocess& operator=(Preprocess&&) = delete;

  void setup(Frame<float>& output) {
    (void)XNNPackGuard::instance();

    assert(output.channels() == channels);
    assert(output.width() == small_size);
    assert(output.height() == small_size);
    resize_output_.resize(small_size, small_size, channels);

    xnn_status status;

    status = xnn_create_resize_bilinear2d_nhwc(
        xnn_datatype_quint8,
        small_size,
        small_size,
        0,
        &resize_op_);
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_create_resize_bilinear2d_nhwc");
    }

    output_ptr_ = output.data();
  }

  void process(const Frame<uint8_t>& input) {
    (void)XNNPackGuard::instance();
    assert(input.channels() == channels);

    xnn_status status;

    size_t workspace_size;
    status = xnn_reshape_resize_bilinear2d_nhwc(
        resize_op_,
        1,
        input.height(),
        input.width(),
        channels,
        channels,
        channels,
        &workspace_size,
        nullptr);
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_reshape_resize_bilinear2d_nhwc");
    }

    status = xnn_setup_resize_bilinear2d_nhwc(
        resize_op_,
        nullptr,
        input.data(),
        resize_output_.data());
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_setup_resize_bilinear2d_nhwc");
    }

    status = xnn_run_operator(resize_op_, nullptr);
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_run_operator(resize_bilinear2d_nhwc)");
    }

    // okay this is not pretty, but I have yet to found a fused operator in XNNPACK
    // for u8 -> f32 conversion AND ImageNet normalization ..
    // weighed versus more internal buffers, this code is reasonably good
    for (size_t y = 0; y < small_size; y++) {
      for (size_t x = 0; x < small_size; x++) {
        uint8_t* in_pixel = resize_output_.data() + ((y * small_size) + x) * channels;
        float* out_pixel = output_ptr_ + ((y * small_size) + x) * channels;

        // transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        out_pixel[0] = (float(in_pixel[0]) / 255.0 - 0.485) / 0.229; // R
        out_pixel[1] = (float(in_pixel[1]) / 255.0 - 0.456) / 0.224; // G
        out_pixel[2] = (float(in_pixel[2]) / 255.0 - 0.406) / 0.225; // B
      }
    }
  }

  Frame<uint8_t> resize_output_;
private:
  xnn_operator_t resize_op_ = nullptr;
  float* output_ptr_ = nullptr;

  constexpr static size_t small_size = 224;
  constexpr static size_t channels = 3;
};

}

