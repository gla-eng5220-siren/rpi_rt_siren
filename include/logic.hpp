#pragma once

#include <memory>

#include "frame.hpp"

namespace rpi_rt {
  // TODO logics should be on their own threads

  class visual_classfying_model_t {
    public:
      virtual ~visual_classfying_model_t() {}
      virtual void setup(const std::string& model_path) = 0;
      virtual float process(const Frame<uint8_t>& frame) = 0;
  };

  std::shared_ptr<visual_classfying_model_t> create_shufflenet_model();
}

