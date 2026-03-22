#pragma once

#include <memory>
#include <cstdint>

#include "frame.hpp"

namespace rpi_rt {
  class http_server_t {
    public:
      virtual ~http_server_t() {}

      virtual void setup() = 0;
      virtual void run(const std::string& host, int port) = 0;
      virtual void close() = 0;
      virtual void set_cam_frame(Frame<uint8_t>) = 0;
      virtual void set_logit(float) = 0;
  };

  std::shared_ptr<http_server_t> create_http_server();
}

