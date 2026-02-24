#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <iterator>

#include "logic.hpp"

#include "shufflenet/preprocess.hpp"
#include "shufflenet/model.hpp"

namespace rpi_rt {
  class shufflenet_model_t : public visual_classfying_model_t {
    public:
      virtual ~shufflenet_model_t() override {}

      virtual void setup(const std::string& model_path) override {
        prep_buffer_.resize(224, 224, 3);
        output_buffer_.resize(1, 1, 1);
        model_params_.resize({4, 8, 4}, {24, 48, 96, 192, 64});
        model_params_.load([&model_path](const std::string& name, float* data, size_t size){
            read_param_file(model_path + "/" + name, data, size);
        });
        model_.setup(prep_buffer_, output_buffer_, model_params_);
      }

      virtual float process(const Frame<uint8_t>& frame) override {
        prep_.setup(frame, prep_buffer_);
        prep_.forward();
        model_.forward();
        return output_buffer_.data()[0];
      }

    private:
      static void read_param_file(const std::string& path, float* data, size_t size) {
        std::ifstream ifs{path, std::ios::binary};
        ifs.unsetf(std::ios::skipws);
        ifs.seekg(0, std::ios::end);
        auto fsize = ifs.tellg() / sizeof(float);
        if (fsize != size)
          throw std::runtime_error(std::string("shufflenet: ") + path + " : param file size mismatch");
        ifs.seekg(0, std::ios::beg);
        ifs.read((char *)data, size * sizeof(float));
      }

      logic::shufflenet::Preprocess prep_;
      Frame<float> prep_buffer_;
      logic::shufflenet::Model<float>::Params model_params_;
      logic::shufflenet::Model<float> model_;
      Frame<float> output_buffer_;
  };

  std::shared_ptr<visual_classfying_model_t> create_shufflenet_model() {
    return std::make_shared<shufflenet_model_t>();
  }
}


