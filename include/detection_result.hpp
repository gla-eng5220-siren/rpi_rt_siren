#pragma once

#include <vector>
#include <string>

namespace rpi_rt {
  class detection_result_t {
    public:
      virtual ~detection_result_t() {}
      virtual bool has_fire() = 0;
      virtual std::string explain() = 0;
      virtual std::vector<uint8_t> jpg_attachment() {
        return {};
      };
  };
}

