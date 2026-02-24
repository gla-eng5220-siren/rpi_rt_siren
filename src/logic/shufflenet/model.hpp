#pragma once

#include <functional>
#include <list>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cassert>
#include <sstream>

#include "conv2d.hpp"
#include "maxpool2d.hpp"
#include "global_average_pool2d.hpp"
#include "inverted_residual.hpp"
#include "frame.hpp"
#include "fc.hpp"

namespace rpi_rt::logic::shufflenet {

template <class Elem>
class Model {
public:
  using elem_t = Elem;

  class Params {
  public:
    Params() {}

    Params(const std::vector<size_t>& stage_repeats, const std::vector<size_t>& output_channels) {
      resize(stage_repeats, output_channels);
    }

    void load(std::function<void (const std::string&, elem_t*, size_t)> load_fn) {
      load_fn("c1w", conv_pre_.data(), conv_pre_.size());
      load_fn("c1b", conv_pre_.bias().data(), conv_pre_.bias().size());
      load_fn("c5w", conv_post_.data(), conv_post_.size());
      load_fn("c5b", conv_post_.bias().data(), conv_post_.bias().size());
      load_fn("fcw", fc_.data(), fc_.size());
      load_fn("fcb", fc_.bias().data(), fc_.bias().size());

      int stage_nr = 2;
      for (auto& stage : stages_) {
        int repeat_nr = 0;
        for (auto& repeat : stage) {
          std::ostringstream oss;
          oss << "s" << stage_nr << "r" << repeat_nr;
          std::string prefix = oss.str();

          auto load_repeat_fn = [&prefix, &load_fn, &repeat](const std::string& suffix){
            load_fn(prefix + "x" + suffix, repeat.data(suffix[0], suffix[1], suffix[2]), repeat.size(suffix[0], suffix[1], suffix[2]));
          };

          // branch1
          if (repeat.has_branch1()) {
            load_repeat_fn("1w0");
            load_repeat_fn("1b0");
            load_repeat_fn("1w1");
            load_repeat_fn("1b1");
          }

          // branch2
          load_repeat_fn("2w0");
          load_repeat_fn("2b0");
          load_repeat_fn("2w1");
          load_repeat_fn("2b1");
          load_repeat_fn("2w2");
          load_repeat_fn("2b2");

          repeat_nr++;
        }
        stage_nr++;
      }
    }

    void resize(const std::vector<size_t>& stage_repeats, const std::vector<size_t>& output_channels) {
      conv_pre_.resize(output_channels[0], 3, 3, 3);
      conv_pre_.stride_width(2);
      conv_pre_.stride_height(2);
      conv_pre_.padding_width(1);
      conv_pre_.padding_height(1);
      conv_pre_.add_bias();
      conv_pre_.relu(true);

      maxpool_.width(3);
      maxpool_.height(3);
      maxpool_.stride_width(2);
      maxpool_.stride_height(2);
      maxpool_.padding_width(1);
      maxpool_.padding_height(1);
      maxpool_.dilation_width(1);
      maxpool_.dilation_height(1);

      stages_.clear();
      for (size_t i = 0; i < stage_repeats.size(); i++) {
        const auto& repeats = stage_repeats[i];
        stages_.emplace_back();
        auto& stage = stages_.back();
        for (size_t j = 0; j < repeats; j++) {
          if (j == 0) {
            stage.emplace_back(output_channels[i + 1], output_channels[i], 2);
          } else {
            stage.emplace_back(output_channels[i + 1], output_channels[i + 1], 1);
          }
        }
      }

      conv_post_.resize(output_channels[output_channels.size() - 1], 1, 1, output_channels[output_channels.size() - 2]);
      conv_post_.stride_width(1);
      conv_post_.stride_height(1);
      conv_post_.add_bias();
      conv_post_.relu(true);

      fc_.resize(1, output_channels[output_channels.size() - 1]);
      fc_.add_bias();
    }

    const auto& conv_pre_params() const noexcept {
      return conv_pre_;
    }

    const auto& maxpool_params() const noexcept {
      return maxpool_;
    }

    const auto& stages_params() const noexcept {
      return stages_;
    }

    const auto& conv_post_params() const noexcept {
      return conv_post_;
    }

    const auto& fc_params() const noexcept {
      return fc_;
    }

  private:
    typename Conv2D<elem_t>::Params conv_pre_;
    typename Maxpool2D<elem_t>::Params maxpool_;
    typename std::list<std::list<typename InvertedResidual<elem_t>::Params>> stages_;
    typename Conv2D<elem_t>::Params conv_post_;
    typename Fc<elem_t>::Params fc_;
  };

  Model() {}

  Model(const Model&) = delete;
  Model(Model&&) = delete;
  Model& operator=(const Model&) = delete;
  Model& operator=(Model&&) = delete;

  void setup(const Frame<float>& input, Frame<float>& output, const Params& params) {
    assert(input.channels() == 3);
    assert(output.width() == 1);
    assert(output.height() == 1);
    assert(output.channels() == 1);

    size_t h = input.height();
    size_t w = input.width();

    h /= 2;
    w /= 2;
    auto& after_conv_pre = create_intermediate(h, w, params.conv_pre_params().output_feature());
    conv_pre_.setup(input, after_conv_pre, params.conv_pre_params());

    h /= 2;
    w /= 2;
    auto& after_maxpool = create_intermediate(h, w, after_conv_pre.channels());
    maxpool_.setup(after_conv_pre, after_maxpool, params.maxpool_params());

    const auto* last_frame = &after_maxpool;
    for (const auto& stage_param : params.stages_params()) {
      h /= 2;
      w /= 2;
      for (const auto& repeat_param : stage_param) {
        auto& after_repeat = create_intermediate(h, w, repeat_param.output_feature());
        stage_repeats_.emplace_back();
        stage_repeats_.back().setup(*last_frame, after_repeat, repeat_param);
        last_frame = &after_repeat;
      }
    }

    auto& after_conv_post = create_intermediate(h, w, params.conv_post_params().output_feature());
    conv_post_.setup(*last_frame, after_conv_post, params.conv_post_params());

    auto& after_mean = create_intermediate(1, 1, after_conv_post.channels());
    mean_.setup(after_conv_post, after_mean);

    fc_.setup(after_mean, output, params.fc_params());
  }

  void forward() {
    conv_pre_.forward();
    maxpool_.forward();
    for (auto& repeat : stage_repeats_) {
      repeat.forward();
    }
    conv_post_.forward();
    mean_.forward();
    fc_.forward();
  }

private:
  Frame<elem_t>& create_intermediate(size_t height, size_t width, size_t channels) {
    intermediate_.emplace_back(height, width, channels);
    return intermediate_.back();
  }

  Conv2D<elem_t> conv_pre_;
  Maxpool2D<elem_t> maxpool_;
  std::list<InvertedResidual<elem_t>> stage_repeats_;
  Conv2D<elem_t> conv_post_;
  GlobalAveragePool2D<elem_t> mean_;
  Fc<elem_t> fc_;
  std::list<Frame<elem_t>> intermediate_;
};

}


