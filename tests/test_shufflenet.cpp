#include "catch2/catch_test_macros.hpp"
#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <iostream>

#include "src/logic/shufflenet/frame.hpp"
#include "src/logic/shufflenet/depthwise_conv2d.hpp"
#include "src/logic/shufflenet/conv2d.hpp"
#include "src/logic/shufflenet/branch2.hpp"

#ifndef TESTDATA_PATH
  #define TESTDATA_PATH "testdata"
#endif

std::vector<float> load_testdata(std::string path) {
  std::ifstream ifs{std::string(TESTDATA_PATH) + "/" + path, std::ios::binary};
  ifs.unsetf(std::ios::skipws);
  ifs.seekg(0, std::ios::end);
  auto size = ifs.tellg() / sizeof(float);
  ifs.seekg(0, std::ios::beg);

  std::vector<float> vec;
  vec.resize(size);

  std::copy(std::istream_iterator<char>(ifs), std::istream_iterator<char>(), (char *)vec.data());
  return vec;
}

bool compare_result(const float* a, const float* b, size_t size, float eps = 0.001) {
  float max_error = 0;
  for (size_t i = 0; i < size; i++) {
    if (a[i] - b[i] > max_error) {
      max_error = a[i] - b[i];
    }
  }
  // std::cout << "error: " << max_error << std::endl;
  return max_error < eps;
}

TEST_CASE("Conv2d", "[shufflenet][kernels]") {
  using rpi_rt::logic::shufflenet::Frame;
  using rpi_rt::logic::shufflenet::Conv2D;

  Frame<float> input_frame(56, 56, 24);
  Frame<float> output_frame(56, 56, 24);

  Conv2D<float>::Params params(24, 1, 1, 24);
  params.stride_width(1);
  params.stride_height(1);

  auto input_data = load_testdata("conv2d_input");
  auto output_data = load_testdata("conv2d_output");
  auto weight_data = load_testdata("conv2d_weight");
  std::copy(input_data.begin(), input_data.end(), input_frame.data());
  std::copy(weight_data.begin(), weight_data.end(), params.data());

  Conv2D<float> conv2d;
  conv2d.setup(input_frame, output_frame, params);
  conv2d.forward();

  CHECK(compare_result(output_frame.data(), output_data.data(), output_data.size()));
}

TEST_CASE("FusedConv2dBatchNorm", "[shufflenet][kernels]") {
  using rpi_rt::logic::shufflenet::Frame;
  using rpi_rt::logic::shufflenet::Conv2D;

  Frame<float> input_frame(56, 56, 24);
  Frame<float> output_frame(56, 56, 24);

  Conv2D<float>::Params params(24, 1, 1, 24);
  params.stride_width(1);
  params.stride_height(1);
  params.add_bias();

  auto input_data = load_testdata("fused_batchnorm_input");
  auto output_data = load_testdata("fused_batchnorm_output");
  auto weight_data = load_testdata("fused_batchnorm_weight");
  auto bias_data = load_testdata("fused_batchnorm_bias");
  std::copy(input_data.begin(), input_data.end(), input_frame.data());
  std::copy(weight_data.begin(), weight_data.end(), params.data());
  std::copy(bias_data.begin(), bias_data.end(), params.bias().data());

  Conv2D<float> conv2d;
  conv2d.setup(input_frame, output_frame, params);
  conv2d.forward();

  CHECK(compare_result(output_frame.data(), output_data.data(), output_data.size()));
}

TEST_CASE("FusedConv2dBatchNormRelu", "[shufflenet][kernels]") {
  using rpi_rt::logic::shufflenet::Frame;
  using rpi_rt::logic::shufflenet::Conv2D;

  Frame<float> input_frame(56, 56, 24);
  Frame<float> output_frame(56, 56, 24);

  Conv2D<float>::Params params(24, 1, 1, 24);
  params.stride_width(1);
  params.stride_height(1);
  params.add_bias();
  params.relu(true);

  auto input_data = load_testdata("fused_batchnorm_input");
  auto output_data = load_testdata("fused_batchnorm_output_after_relu");
  auto weight_data = load_testdata("fused_batchnorm_weight");
  auto bias_data = load_testdata("fused_batchnorm_bias");
  std::copy(input_data.begin(), input_data.end(), input_frame.data());
  std::copy(weight_data.begin(), weight_data.end(), params.data());
  std::copy(bias_data.begin(), bias_data.end(), params.bias().data());

  Conv2D<float> conv2d;
  conv2d.setup(input_frame, output_frame, params);
  conv2d.forward();

  CHECK(compare_result(output_frame.data(), output_data.data(), output_data.size()));
}

TEST_CASE("DepthwiseConv2d", "[shufflenet][kernels]") {
  using rpi_rt::logic::shufflenet::Frame;
  using rpi_rt::logic::shufflenet::DepthwiseConv2D;

  Frame<float> input_frame(56, 56, 24);
  Frame<float> output_frame(28, 28, 24);

  DepthwiseConv2D<float>::Params params(24, 3, 3);
  params.padding_width(1);
  params.padding_height(1);
  params.stride_width(2);
  params.stride_height(2);

  auto input_data = load_testdata("depthwise_conv2d_input");
  auto output_data = load_testdata("depthwise_conv2d_output");
  auto weight_data = load_testdata("depthwise_conv2d_weight");
  std::copy(input_data.begin(), input_data.end(), input_frame.data());
  std::copy(weight_data.begin(), weight_data.end(), params.data());

  DepthwiseConv2D<float> conv2d;
  conv2d.setup(input_frame, output_frame, params);
  conv2d.forward();

  CHECK(compare_result(output_frame.data(), output_data.data(), output_data.size()));
}

TEST_CASE("Branch2Demo", "[shufflenet][kernels]") {
  using rpi_rt::logic::shufflenet::Frame;
  using rpi_rt::logic::shufflenet::Branch2;

  Frame<float> input_frame(56, 56, 24);
  Frame<float> output_frame(28, 28, 24);

  Branch2<float>::Params params(48, 24, 2);

  auto w0 = load_testdata("demo_x2w0");
  auto w1 = load_testdata("demo_x2w1");
  auto w2 = load_testdata("demo_x2w2");
  auto b0 = load_testdata("demo_x2b0");
  auto b1 = load_testdata("demo_x2b1");
  auto b2 = load_testdata("demo_x2b2");
  auto input_data = load_testdata("demo_x2_input");
  auto output_data = load_testdata("demo_x2_output");
  std::copy(input_data.begin(), input_data.end(), input_frame.data());
  std::copy(w0.begin(), w0.end(), params.data('w', '0'));
  std::copy(w1.begin(), w1.end(), params.data('w', '1'));
  std::copy(w2.begin(), w2.end(), params.data('w', '2'));
  std::copy(b0.begin(), b0.end(), params.data('b', '0'));
  std::copy(b1.begin(), b1.end(), params.data('b', '1'));
  std::copy(b2.begin(), b2.end(), params.data('b', '2'));

  Branch2<float> x2;
  x2.setup(input_frame, output_frame, params);
  x2.forward();

  CHECK(compare_result(output_frame.data(), output_data.data(), output_data.size()));
}

