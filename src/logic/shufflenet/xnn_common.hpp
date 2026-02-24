#pragma once

#include <stdexcept>

#include "xnnpack.h"

namespace rpi_rt::logic::shufflenet {

class XNNPackGuard {
public:
  // meyer's singleton
  static XNNPackGuard& instance() {
    static XNNPackGuard guard;
    return guard;
  }

private:
  XNNPackGuard() {
    xnn_status status = xnn_initialize(nullptr);
    if (status != xnn_status_success) {
      throw std::runtime_error("xnn_initialize");
    }
  }

  ~XNNPackGuard() {
    (void)xnn_deinitialize();
  }
};

template <class Elem>
class Bias {
public:
    Bias() {}

    Bias(size_t channels) {
      resize(channels);
    }

    void resize(size_t channels) {
      buffer_.resize(channels);
    }

    size_t channels() const noexcept {
      return buffer_.size();
    }

    size_t size() const noexcept {
      return channels();
    }

    const Elem* data() const noexcept {
      return buffer_.data();
    }

    Elem* data() noexcept {
      return buffer_.data();
    }

private:
  std::vector<Elem> buffer_;
};

}

