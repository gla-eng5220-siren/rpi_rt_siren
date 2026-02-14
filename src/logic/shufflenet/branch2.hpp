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
class Branch2 {
public:
  using elem_t = Elem;

  class Params {
  public:
    Params() {}

    Params(size_t output_feature, size_t input_feature, size_t stride) {
      resize(output_feature, input_feature, stride);
    }

    void resize(size_t output_feature, size_t input_feature, size_t stride) {
      size_t branch_features = output_feature / 2;

      first_conv_params_.resize(branch_features, 1, 1, stride > 1 ? input_feature : branch_features);
      first_conv_params_.stride_width(1);
      first_conv_params_.stride_height(1);
      first_conv_params_.add_bias();
      first_conv_params_.relu(true);

      second_conv_params_.resize(branch_features, 3, 3);
      second_conv_params_.stride_width(stride);
      second_conv_params_.stride_height(stride);
      second_conv_params_.padding_width(1);
      second_conv_params_.padding_height(1);
      second_conv_params_.add_bias();

      third_conv_params_.resize(branch_features, 1, 1, branch_features);
      third_conv_params_.stride_width(1);
      third_conv_params_.stride_height(1);
      third_conv_params_.add_bias();
      third_conv_params_.relu(true);
    }

    elem_t* data(char type, char index) noexcept {
      switch (type) {
        case 'w':
          switch (index) {
            case '0':
              return first_conv_params_.data();
            case '1':
              return second_conv_params_.data();
            case '2':
              return third_conv_params_.data();
            default:
              return nullptr;
          }
        case 'b':
          switch (index) {
            case '0':
              return first_conv_params_.bias().data();
            case '1':
              return second_conv_params_.bias().data();
            case '2':
              return third_conv_params_.bias().data();
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
            case '2':
              return third_conv_params_.size();
            default:
              return 0;
          }
        case 'b':
          switch (index) {
            case '0':
              return first_conv_params_.bias().size();
            case '1':
              return second_conv_params_.bias().size();
            case '2':
              return third_conv_params_.bias().size();
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

    const auto& third_conv_params() const noexcept {
      return third_conv_params_;
    }

  private:
    typename Conv2D<elem_t>::Params first_conv_params_;
    typename DepthwiseConv2D<elem_t>::Params second_conv_params_;
    typename Conv2D<elem_t>::Params third_conv_params_;
  };

  Branch2() {}

  Branch2(const Branch2&) = delete;
  Branch2(Branch2&&) = delete;
  Branch2& operator=(const Branch2&) = delete;
  Branch2& operator=(Branch2&&) = delete;

  void setup(const Frame<float>& input, Frame<float>& output, const Params& params) {
    buffer_one_.resize(input.height(), input.width(), params.first_conv_params().output_feature());
    buffer_two_.resize(output.height(), output.width(), params.second_conv_params().channels());

    first_conv_.setup(input, buffer_one_, params.first_conv_params());
    second_conv_.setup(buffer_one_, buffer_two_, params.second_conv_params());
    third_conv_.setup(buffer_two_, output, params.third_conv_params());
  }

  void forward() {
    first_conv_.forward();
    second_conv_.forward();
    third_conv_.forward();
  }

private:
  Conv2D<elem_t> first_conv_;
  DepthwiseConv2D<elem_t> second_conv_;
  Conv2D<elem_t> third_conv_;
  Frame<elem_t> buffer_one_;
  Frame<elem_t> buffer_two_;
};

}

