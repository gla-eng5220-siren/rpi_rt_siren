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
class Shuffle {
public:
  using elem_t = Elem;

  Shuffle() {}

  Shuffle(const Shuffle&) = delete;
  Shuffle(Shuffle&&) = delete;
  Shuffle& operator=(const Shuffle&) = delete;
  Shuffle& operator=(Shuffle&&) = delete;

  void setup(const Frame<elem_t>& input_a, const Frame<elem_t>& input_b, Frame<elem_t>& output) {
    assert(input_a.channels() == input_b.channels());
    assert(input_a.channels() + input_b.channels() == output.channels());
    assert(input_a.width() == input_b.width());
    assert(input_a.height() == input_b.height());
    assert(input_a.width() == output.width());
    assert(input_a.height() == output.height());

    repeats_ = input_a.width() * input_a.height() * input_a.channels();
    input_a_ = input_a.data();
    input_b_ = input_b.data();
    output_ = output.data();
  }

  void forward() {
    for (size_t i = 0; i < repeats_; i++) {
      output_[i * 2 + 0] = input_a_[i];
      output_[i * 2 + 1] = input_b_[i];
    }
  }

private:
  size_t repeats_ = 0;
  const elem_t *input_a_ = nullptr;
  const elem_t *input_b_ = nullptr;
  elem_t *output_ = nullptr;
};

}


