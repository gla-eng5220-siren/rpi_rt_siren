#include "catch2/catch_test_macros.hpp"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <iostream>

#include "src/logic/shufflenet/frame.hpp"
#include "src/logic/shufflenet/depthwise_conv2d.hpp"
#include "src/logic/shufflenet/maxpool2d.hpp"
#include "src/logic/shufflenet/fc.hpp"
#include "src/logic/shufflenet/global_average_pool2d.hpp"
#include "src/logic/shufflenet/conv2d.hpp"
#include "src/logic/shufflenet/branch1.hpp"
#include "src/logic/shufflenet/branch2.hpp"
#include "src/logic/shufflenet/inverted_residual.hpp"
#include "src/logic/shufflenet/model.hpp"
#include "src/logic/shufflenet/preprocess.hpp"

#ifndef TESTDATA_PATH
  #define TESTDATA_PATH "testdata"
#endif

template<class Elem = float>
std::vector<Elem> load_testdata(std::string path) {
  std::ifstream ifs{std::string(TESTDATA_PATH) + "/" + path, std::ios::binary};
  ifs.unsetf(std::ios::skipws);
  ifs.seekg(0, std::ios::end);
  auto size = ifs.tellg() / sizeof(Elem);
  ifs.seekg(0, std::ios::beg);

  std::vector<Elem> vec;
  vec.resize(size);

  std::copy(std::istream_iterator<char>(ifs), std::istream_iterator<char>(), (char *)vec.data());
  return vec;
}

bool compare_result(const float* a, const float* b, size_t size, float eps = 0.001) {
  float max_error = 0;
  for (size_t i = 0; i < size; i++) {
    if (std::abs(a[i] - b[i]) > max_error) {
      max_error = std::abs(a[i] - b[i]);
    }
  }
  return max_error < eps;
}

TEST_CASE("Maxpool2D", "[shufflenet][kernels]") {
  using rpi_rt::logic::shufflenet::Frame;
  using rpi_rt::logic::shufflenet::Maxpool2D;

  Frame<float> input_frame(112, 112, 24);
  Frame<float> output_frame(56, 56, 24);

  Maxpool2D<float>::Params params;
  params.width(3);
  params.height(3);
  params.stride_width(2);
  params.stride_height(2);
  params.padding_width(1);
  params.padding_height(1);
  params.dilation_width(1);
  params.dilation_height(1);

  auto input_data = load_testdata("maxpool_input");
  auto output_data = load_testdata("maxpool_output");
  std::copy(input_data.begin(), input_data.end(), input_frame.data());

  Maxpool2D<float> maxpool2d;
  maxpool2d.setup(input_frame, output_frame, params);
  maxpool2d.forward();

  CHECK(compare_result(output_frame.data(), output_data.data(), output_data.size()));
}

TEST_CASE("GlobalAveragePool2D", "[shufflenet][kernels]") {
  using rpi_rt::logic::shufflenet::Frame;
  using rpi_rt::logic::shufflenet::GlobalAveragePool2D;

  Frame<float> input_frame(7, 7, 64);
  Frame<float> output_frame(1, 1, 64);

  auto input_data = load_testdata("mean_input");
  auto output_data = load_testdata("mean_output");
  std::copy(input_data.begin(), input_data.end(), input_frame.data());

  GlobalAveragePool2D<float> pool;
  pool.setup(input_frame, output_frame);
  pool.forward();

  CHECK(compare_result(output_frame.data(), output_data.data(), output_data.size()));
}

TEST_CASE("Fc", "[shufflenet][kernels]") {
  using rpi_rt::logic::shufflenet::Frame;
  using rpi_rt::logic::shufflenet::Fc;

  Frame<float> input_frame(1, 1, 64);
  Frame<float> output_frame(1, 1, 1);

  Fc<float>::Params params(1, 64);
  params.add_bias();

  auto input_data = load_testdata("fc_input");
  auto output_data = load_testdata("fc_output");
  auto weight_data = load_testdata("fc_weight");
  auto bias_data = load_testdata("fc_bias");
  std::copy(input_data.begin(), input_data.end(), input_frame.data());
  std::copy(weight_data.begin(), weight_data.end(), params.data());
  std::copy(bias_data.begin(), bias_data.end(), params.bias().data());

  Fc<float> fc;
  fc.setup(input_frame, output_frame, params);
  fc.forward();

  CHECK(compare_result(output_frame.data(), output_data.data(), output_data.size()));
}

TEST_CASE("Conv2D", "[shufflenet][kernels]") {
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

TEST_CASE("FusedConv2DBatchNorm", "[shufflenet][kernels]") {
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

TEST_CASE("FusedConv2DBatchNormRelu", "[shufflenet][kernels]") {
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

TEST_CASE("DepthwiseConv2D", "[shufflenet][kernels]") {
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

TEST_CASE("Branch1Demo", "[shufflenet][branchs]") {
  using rpi_rt::logic::shufflenet::Frame;
  using rpi_rt::logic::shufflenet::Branch1;

  Frame<float> input_frame(56, 56, 24);
  Frame<float> output_frame(28, 28, 24);

  Branch1<float>::Params params(48, 24, 2);

  auto w0 = load_testdata("demo_x1w0");
  auto w1 = load_testdata("demo_x1w1");
  auto b0 = load_testdata("demo_x1b0");
  auto b1 = load_testdata("demo_x1b1");
  auto input_data = load_testdata("demo_x1_input");
  auto output_data = load_testdata("demo_x1_output");
  std::copy(input_data.begin(), input_data.end(), input_frame.data());
  std::copy(w0.begin(), w0.end(), params.data('w', '0'));
  std::copy(w1.begin(), w1.end(), params.data('w', '1'));
  std::copy(b0.begin(), b0.end(), params.data('b', '0'));
  std::copy(b1.begin(), b1.end(), params.data('b', '1'));

  Branch1<float> x1;
  x1.setup(input_frame, output_frame, params);
  x1.forward();

  CHECK(compare_result(output_frame.data(), output_data.data(), output_data.size()));
}

TEST_CASE("Branch2Demo", "[shufflenet][branchs]") {
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

TEST_CASE("InvertedResidualDemo", "[shufflenet][repeats]") {
  using rpi_rt::logic::shufflenet::Frame;
  using rpi_rt::logic::shufflenet::InvertedResidual;

  Frame<float> input_frame(56, 56, 24);
  Frame<float> output_frame(28, 28, 48);

  InvertedResidual<float>::Params params(48, 24, 2);

  auto x1w0 = load_testdata("demo_x1w0");
  auto x1w1 = load_testdata("demo_x1w1");
  auto x1b0 = load_testdata("demo_x1b0");
  auto x1b1 = load_testdata("demo_x1b1");
  auto x2w0 = load_testdata("demo_x2w0");
  auto x2w1 = load_testdata("demo_x2w1");
  auto x2w2 = load_testdata("demo_x2w2");
  auto x2b0 = load_testdata("demo_x2b0");
  auto x2b1 = load_testdata("demo_x2b1");
  auto x2b2 = load_testdata("demo_x2b2");
  auto input_data = load_testdata("demo_input");
  auto output_data = load_testdata("demo_output");
  std::copy(input_data.begin(), input_data.end(), input_frame.data());
  std::copy(x1w0.begin(), x1w0.end(), params.data('1', 'w', '0'));
  std::copy(x1w1.begin(), x1w1.end(), params.data('1', 'w', '1'));
  std::copy(x1b0.begin(), x1b0.end(), params.data('1', 'b', '0'));
  std::copy(x1b1.begin(), x1b1.end(), params.data('1', 'b', '1'));
  std::copy(x2w0.begin(), x2w0.end(), params.data('2', 'w', '0'));
  std::copy(x2w1.begin(), x2w1.end(), params.data('2', 'w', '1'));
  std::copy(x2w2.begin(), x2w2.end(), params.data('2', 'w', '2'));
  std::copy(x2b0.begin(), x2b0.end(), params.data('2', 'b', '0'));
  std::copy(x2b1.begin(), x2b1.end(), params.data('2', 'b', '1'));
  std::copy(x2b2.begin(), x2b2.end(), params.data('2', 'b', '2'));

  InvertedResidual<float> r;
  r.setup(input_frame, output_frame, params);
  r.forward();

  CHECK(compare_result(output_frame.data(), output_data.data(), output_data.size()));
}

TEST_CASE("InvertedResidualChunkingDemo", "[shufflenet][repeats]") {
  using rpi_rt::logic::shufflenet::Frame;
  using rpi_rt::logic::shufflenet::InvertedResidual;

  Frame<float> input_frame(28, 28, 48);
  Frame<float> output_frame(28, 28, 48);

  InvertedResidual<float>::Params params(48, 48, 1);
  CHECK(!params.has_branch1());

  auto x2w0 = load_testdata("chunking_x2w0");
  auto x2w1 = load_testdata("chunking_x2w1");
  auto x2w2 = load_testdata("chunking_x2w2");
  auto x2b0 = load_testdata("chunking_x2b0");
  auto x2b1 = load_testdata("chunking_x2b1");
  auto x2b2 = load_testdata("chunking_x2b2");
  auto input_data = load_testdata("chunking_input");
  auto output_data = load_testdata("chunking_output");
  std::copy(input_data.begin(), input_data.end(), input_frame.data());
  std::copy(x2w0.begin(), x2w0.end(), params.data('2', 'w', '0'));
  std::copy(x2w1.begin(), x2w1.end(), params.data('2', 'w', '1'));
  std::copy(x2w2.begin(), x2w2.end(), params.data('2', 'w', '2'));
  std::copy(x2b0.begin(), x2b0.end(), params.data('2', 'b', '0'));
  std::copy(x2b1.begin(), x2b1.end(), params.data('2', 'b', '1'));
  std::copy(x2b2.begin(), x2b2.end(), params.data('2', 'b', '2'));

  InvertedResidual<float> r;
  r.setup(input_frame, output_frame, params);
  r.forward();

  CHECK(compare_result(output_frame.data(), output_data.data(), output_data.size()));
}

TEST_CASE("Model", "[shufflenet][model]") {
  using rpi_rt::logic::shufflenet::Frame;
  using rpi_rt::logic::shufflenet::Model;

  Frame<float> input_frame(224, 224, 3);
  Frame<float> output_frame(1, 1, 1);

  Model<float>::Params params({4, 8, 4}, {24, 48, 96, 192, 64});
  params.load([](const std::string& name, float* data, size_t size){
    std::cout << "Loading " << name << std::endl;
    auto loaded = load_testdata("model/" + name);
    assert(size == loaded.size());
    std::copy(loaded.begin(), loaded.end(), data);
  });

  auto input = load_testdata("model_input");
  auto output = load_testdata("model_output");
  std::copy(input.begin(), input.end(), input_frame.data());

  Model<float> m;
  m.setup(input_frame, output_frame, params);
  m.forward();

  float result = output_frame.data()[0];
  std::cout << "Result: " << result << std::endl;

  CHECK(std::abs(result - output[0]) < 0.01);
}

TEST_CASE("Preprocess", "[shufflenet][preprocess]") {
  using rpi_rt::logic::shufflenet::Frame;
  using rpi_rt::logic::shufflenet::Preprocess;

  Frame<uint8_t> input_frame(1275, 1920, 3);
  Frame<float> output_frame(224, 224, 3);

  auto input = load_testdata<uint8_t>("preprocess_input");
  auto output = load_testdata("preprocess_output");
  std::copy(input.begin(), input.end(), input_frame.data());

  Preprocess prep;
  prep.setup(input_frame, output_frame);
  prep.forward();

  CHECK(compare_result(output_frame.data(), output.data(), output.size(), 0.03));
}

