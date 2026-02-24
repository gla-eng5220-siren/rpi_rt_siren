#pragma once

#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cassert>

#include "frame.hpp"

namespace rpi_rt::logic::shufflenet {

template <class Elem>
class Chunk {
public:
  using elem_t = Elem;

  Chunk() {}

  Chunk(const Chunk&) = delete;
  Chunk(Chunk&&) = delete;
  Chunk& operator=(const Chunk&) = delete;
  Chunk& operator=(Chunk&&) = delete;

  void setup(const Frame<elem_t>& input, Frame<elem_t>& output_a, Frame<elem_t>& output_b) {
    assert(output_a.channels() == output_b.channels());
    assert(output_a.channels() + output_b.channels() == input.channels());
    assert(output_a.width() == output_b.width());
    assert(output_a.height() == output_b.height());
    assert(output_a.width() == input.width());
    assert(output_a.height() == input.height());

    repeats_ = output_a.width() * output_a.height();
    step_ = output_a.channels();
    output_a_ = output_a.data();
    output_b_ = output_b.data();
    input_ = input.data();
  }

  void forward() {
    for (size_t i = 0; i < repeats_; i++) {
      for (size_t j = 0; j < step_; j++) {
        output_a_[i * step_ + j] = input_[(2 * i + 0) * step_ + j];
      }
      for (size_t j = 0; j < step_; j++) {
        output_b_[i * step_ + j] = input_[(2 * i + 1) * step_ + j];
      }
    }
  }

private:
  size_t repeats_ = 0;
  size_t step_ = 0;
  const elem_t *input_ = nullptr;
  elem_t *output_a_ = nullptr;
  elem_t *output_b_ = nullptr;
};

}


