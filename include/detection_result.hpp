#pragma once

namespace rpi_rt {
  class detection_result_t {
    public:
      virtual ~detection_result_t() {}
      virtual bool has_fire() = 0;
      virtual std::string explain() = 0;
  };
}

