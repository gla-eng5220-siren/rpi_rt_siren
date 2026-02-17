#pragma once

#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cassert>

#include "branch1.hpp"
#include "branch2.hpp"
#include "chunk.hpp"
#include "shuffle.hpp"
#include "frame.hpp"

namespace rpi_rt::logic::shufflenet {

template <class Elem>
class InvertedResidual {
public:
  using elem_t = Elem;

  class Params {
  public:
    Params() {}

    Params(size_t output_feature, size_t input_feature, size_t stride) {
      resize(output_feature, input_feature, stride);
    }

    void resize(size_t output_feature, size_t input_feature, size_t stride) {
      output_feature_ = output_feature;
      input_feature_ = input_feature;
      stride_ = stride;

      if (stride > 1) {
        branch1_params_.emplace(output_feature, input_feature, stride);
      } else {
        branch1_params_ = std::nullopt;
      }
      branch2_params_.resize(output_feature, input_feature, stride);
    }

    elem_t* data(char branch, char type, char index) noexcept {
      switch (branch) {
        case '1':
          return branch1_params_->data(type, index);
        case '2':
          return branch2_params_.data(type, index);
        default:
          return nullptr;
      }
    }

    size_t size(char branch, char type, char index) const noexcept {
      switch (branch) {
        case '1':
          return branch1_params_->size(type, index);
        case '2':
          return branch2_params_.size(type, index);
        default:
          return 0;
      }
    }

    bool has_branch1() const noexcept {
      return branch1_params_.has_value();
    }

    const auto& branch1_params() const noexcept {
      return *branch1_params_;
    }

    const auto& branch2_params() const noexcept {
      return branch2_params_;
    }

    const auto& output_feature() const noexcept {
      return output_feature_;
    }

    const auto& input_feature() const noexcept {
      return input_feature_;
    }

    const auto& stride() const noexcept {
      return stride_;
    }

  private:
    size_t input_feature_ = 0;
    size_t output_feature_ = 0;
    size_t stride_ = 0;
    std::optional<typename Branch1<elem_t>::Params> branch1_params_ = std::nullopt;
    typename Branch2<elem_t>::Params branch2_params_;
  };

  InvertedResidual() {}

  InvertedResidual(const InvertedResidual&) = delete;
  InvertedResidual(InvertedResidual&&) = delete;
  InvertedResidual& operator=(const InvertedResidual&) = delete;
  InvertedResidual& operator=(InvertedResidual&&) = delete;

  void setup(const Frame<float>& input, Frame<float>& output, const Params& params) {
    out1_.resize(output.height(), output.width(), output.channels() / 2);
    out2_.resize(output.height(), output.width(), output.channels() / 2);

    if (params.has_branch1()) {
      branch1_.emplace();
      branch1_->setup(input, out1_, params.branch1_params());
      branch2_.setup(input, out2_, params.branch2_params());
    } else {
      branch1_ = std::nullopt;
      in2_.resize(input.height(), input.width(), input.channels() / 2);
      chunk_.setup(input, out1_, in2_);
      branch2_.setup(in2_, out2_, params.branch2_params());
    }
    shuffle_.setup(out1_, out2_, output);
  }

  void forward() {
    if (branch1_.has_value()) {
      branch1_->forward();
    } else {
      chunk_.forward();
    }
    branch2_.forward();
    shuffle_.forward();
  }

private:
  std::optional<Branch1<elem_t>> branch1_ = std::nullopt;
  Branch2<elem_t> branch2_;
  Frame<elem_t> out1_;
  Frame<elem_t> out2_;
  Frame<elem_t> in2_;
  Chunk<elem_t> chunk_;
  Shuffle<elem_t> shuffle_;
};

}

