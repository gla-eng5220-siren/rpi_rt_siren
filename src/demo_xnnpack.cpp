#include "xnnpack.h"
#include <algorithm>
#include <vector>
#include <cmath>

#include "logic/shufflenet/frame.hpp"
#include "logic/shufflenet/depthwise_conv2d.hpp"

#define cimg_display 0
#include "CImg.h"

int main() {
  using cimg_library::CImg;
  using rpi_rt::logic::shufflenet::Frame;
  using rpi_rt::logic::shufflenet::DepthwiseConv2D;

  CImg<float> img("Ogdenhousefire10125.png");
  Frame<float> input_frame(img.height(), img.width(), 3);
  Frame<float> output_frame(img.height(), img.width(), 3);
  DepthwiseConv2D<float>::Params params(3, 3, 3);
  params.padding_width(1);
  params.padding_height(1);
  params.stride_width(1);
  params.stride_height(1);

  std::copy(img.data(), img.data() + img.size(), input_frame.data());
  std::fill(params.data(), params.data() + params.size(), 1.0f / 9);

  DepthwiseConv2D<float> conv2d;
  conv2d.setup(input_frame, output_frame, params);
  conv2d.forward();

  CImg<float> out_img(output_frame.data(), output_frame.width(), output_frame.height(), 1, 3);
  out_img.save("output.png");

  return 0;
}

