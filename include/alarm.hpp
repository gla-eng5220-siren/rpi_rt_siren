#pragma once

#include <memory>
#include <functional>

namespace rpi_rt {
  class alarm_t {
    public:
      virtual ~alarm_t() {}
      virtual void run() = 0;
      virtual void report_fire() = 0;
      virtual void close() = 0;
  };

  std::shared_ptr<alarm_t> create_stdout_alarm();
}

