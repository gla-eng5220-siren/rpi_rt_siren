#pragma once

#include <memory>
#include <functional>

#include "detection_result.hpp"

namespace rpi_rt {
  class alarm_t {
    public:
      virtual ~alarm_t() {}
      virtual void run() = 0;
      virtual void report(std::unique_ptr<detection_result_t>) = 0;
      virtual void close() = 0;
  };

  std::shared_ptr<alarm_t> create_stdout_alarm();
}

