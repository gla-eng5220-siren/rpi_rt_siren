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
 Global Average Pooling (GAP) is a layer used in convolutional neural
 networks (CNNs) that reduces each feature map to a single number by
 taking the average of all its values.

 Assumes NHWC compact memory layout.
 */
template <class Elem>
class GlobalAveragePool2D {
public:
  using elem_t = Elem;
  static_assert(std::is_same_v<Elem, float>, "Only F32 is implemented for GlobalAveragePool2D");

  GlobalAveragePool2D() {}
  ~GlobalAveragePool2D() {
    xnn_delete_operator(pool_op_);
  }

  GlobalAveragePool2D(const GlobalAveragePool2D&) = delete;
  GlobalAveragePool2D(GlobalAveragePool2D&&) = delete;
  GlobalAveragePool2D& operator=(const GlobalAveragePool2D&) = delete;
  GlobalAveragePool2D& operator=(GlobalAveragePool2D&&) = delete;

  void setup(const Frame<float>& input, Frame<float>& output) {
    (void)XNNPackGuard::instance();

    assert(input.channels() == output.channels());
    assert(output.width() == 1);
    assert(output.height() == 1);

    xnn_status status;

    xnn_quantization_params no_quantization;
    no_quantization.scale = 1.0f;
    no_quantization.zero_point = 0;
    status = xnn_create_reduce_nd(
        xnn_reduce_mean,
        xnn_datatype_fp32,
        &no_quantization,
        &no_quantization,
        0,
        &pool_op_);
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_create_reduce_nd");
    }

    size_t shape[] = {input.height(), input.width(), input.channels()};
    int64_t axes[] = {0, 1};
    size_t workspace_size;
    status = xnn_reshape_reduce_nd(
        pool_op_,
        2,
        axes,
        3,
        shape,
        &workspace_size,
        nullptr);
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_reshape_reduce_nd");
    }

    status = xnn_setup_reduce_nd(
        pool_op_,
        nullptr,
        input.data(),
        output.data());
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_setup_reduce_nd");
    }
  }

  void forward() {
    (void)XNNPackGuard::instance();

    xnn_status status;
    status = xnn_run_operator(pool_op_, nullptr);
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_run_operator(reduce_nd:mean)");
    }
  }

private:
  xnn_operator_t pool_op_ = nullptr;
};

}

