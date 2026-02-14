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

}

