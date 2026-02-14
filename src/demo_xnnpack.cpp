#include "xnnpack.h"
#include <algorithm>
#include <vector>
#include <cmath>

#define cimg_display 0
#include "CImg.h"

int main() {
  using cimg_library::CImg;

  CImg<float> img("Ogdenhousefire10125.png");

  const size_t batch_size = 1;
  const size_t height = img.height();
  const size_t width = img.width();
  const size_t channels = 3;
  const size_t kernel_height = 3;
  const size_t kernel_width = 3;
  const size_t padding = 1;
  const size_t stride = 1;

  xnn_status status = xnn_initialize(nullptr);
  if (status != xnn_status_success) {
    return 1;
  }

  std::vector<float> input(batch_size * height * width * channels);
  auto it = input.begin();
  it = std::copy(img.data(), img.data() + img.size(), it);

  std::vector<float> output(batch_size * height * width * channels);

  std::vector<float> kernel(channels * kernel_height * kernel_width * channels);
  std::fill(kernel.begin(), kernel.end(), 1.0f / 9);

  std::vector<float> bias(channels);
  std::fill(bias.begin(), bias.end(), 0);

  xnn_operator_t conv_op = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
      padding, padding, padding, padding,
      kernel_height, kernel_width,
      stride, stride,
      1, 1,
      channels,
      1,
      1,
      channels,
      channels,
      kernel.data(),
      bias.data(),
      -INFINITY, INFINITY,
      0,
      nullptr,
      &conv_op);

  if (status != xnn_status_success) {
    return 1;
  }

  size_t ws_size, oh, ow;

  xnn_reshape_convolution2d_nhwc_f32(
    conv_op,
    1,
    height,
    width,
    &ws_size,
    &oh,
    &ow,
    nullptr);

  status = xnn_setup_convolution2d_nhwc_f32(
      conv_op,
      nullptr,
      input.data(),
      output.data());

  if (status != xnn_status_success) {
    return 1;
  }

  status = xnn_run_operator(conv_op, nullptr);

  if (status != xnn_status_success) {
    return 1;
  }

  CImg<float> out_img(output.data(), width, height, 1, 3);

  out_img.save("output.png");

  xnn_delete_operator(conv_op);
  xnn_deinitialize();

  return 0;
}

