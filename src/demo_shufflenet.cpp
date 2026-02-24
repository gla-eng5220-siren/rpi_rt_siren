#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <iostream>

#include "jpeglib.h"

#include "logic.hpp"
#include "frame.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: MODEL_PATH=./model demo_shufflenet example.jpg" << std::endl;
    return 1;
  }

  const char* model_path_ptr = std::getenv("MODEL_PATH");
  std::string model_path{"model"};
  if (model_path_ptr)
    model_path = std::string(model_path_ptr);

  auto model = rpi_rt::create_shufflenet_model();
  model->setup(model_path);

  auto input = rpi_rt::jpeg_utils::read_from_file(std::string(argv[1]));
  float result = model->process(input);
  std::cout << "Result : " << result << std::endl;

  if (result > 0) {
    std::cout << "NO FIRE" << std::endl;
  } else {
    std::cout << "FIRE" << std::endl;
  }
  return 0;
}

