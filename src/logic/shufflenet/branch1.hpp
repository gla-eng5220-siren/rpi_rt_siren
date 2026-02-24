#pragma once

#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cassert>

#include "conv2d.hpp"
#include "depthwise_conv2d.hpp"
#include "xnnpack.h"
#include "xnn_common.hpp"
#include "frame.hpp"

namespace rpi_rt::logic::shufflenet {

template <class Elem>
class Branch1 {
public:
  using elem_t = Elem;

  // assumes stride > 1 (branch1 is non-existent otherwise)
  class Params {
  public:
    Params() {}

    Params(size_t output_feature, size_t input_feature, size_t stride) {
      resize(output_feature, input_feature, stride);
    }

    void resize(size_t output_feature, size_t input_feature, size_t stride) {
      size_t branch_features = output_feature / 2;

      first_conv_params_.resize(input_feature, 3, 3);
      first_conv_params_.stride_width(stride);
      first_conv_params_.stride_height(stride);
      first_conv_params_.padding_width(1);
      first_conv_params_.padding_height(1);
      first_conv_params_.add_bias();

      second_conv_params_.resize(branch_features, 1, 1, input_feature);
      second_conv_params_.stride_width(1);
      second_conv_params_.stride_height(1);
      second_conv_params_.add_bias();
      second_conv_params_.relu(true);
    }

    elem_t* data(char type, char index) noexcept {
      switch (type) {
        case 'w':
          switch (index) {
            case '0':
              return first_conv_params_.data();
            case '1':
              return second_conv_params_.data();
            default:
              return nullptr;
          }
        case 'b':
          switch (index) {
            case '0':
              return first_conv_params_.bias().data();
            case '1':
              return second_conv_params_.bias().data();
            default:
              return nullptr;
          }
        default:
          return nullptr;
      }
    }

    size_t size(char type, char index) const noexcept {
      switch (type) {
        case 'w':
          switch (index) {
            case '0':
              return first_conv_params_.size();
            case '1':
              return second_conv_params_.size();
            default:
              return 0;
          }
        case 'b':
          switch (index) {
            case '0':
              return first_conv_params_.bias().size();
            case '1':
              return second_conv_params_.bias().size();
            default:
              return 0;
          }
        default:
          return 0;
      }
    }

    const auto& first_conv_params() const noexcept {
      return first_conv_params_;
    }

    const auto& second_conv_params() const noexcept {
      return second_conv_params_;
    }

  private:
    typename DepthwiseConv2D<elem_t>::Params first_conv_params_;
    typename Conv2D<elem_t>::Params second_conv_params_;
  };

  Branch1() {}

  Branch1(const Branch1&) = delete;
  Branch1(Branch1&&) = delete;
  Branch1& operator=(const Branch1&) = delete;
  Branch1& operator=(Branch1&&) = delete;

  void setup(const Frame<float>& input, Frame<float>& output, const Params& params) {
    buffer_.resize(output.height(), output.width(), params.first_conv_params().channels());

    first_conv_.setup(input, buffer_, params.first_conv_params());
    second_conv_.setup(buffer_, output, params.second_conv_params());
  }

  void forward() {
    first_conv_.forward();
    second_conv_.forward();
  }

private:
  DepthwiseConv2D<elem_t> first_conv_;
  Conv2D<elem_t> second_conv_;
  Frame<elem_t> buffer_;
};

}

