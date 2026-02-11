#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace rpi_rt {

// assumes NHWC and N == 1
// alwasys use a compact mem layout

template <class Elem>
class Frame {
public:
  using elem_t = Elem;

  Frame() {}

  Frame(size_t height, size_t width, size_t channels) {
    resize(height, width, channels);
  }

  void resize(size_t height, size_t width, size_t channels) {
    width_ = width;
    height_ = height;
    channels_ = channels;
    buffer_.resize(size());
  }

  size_t width() const noexcept {
    return width_;
  }

  size_t height() const noexcept {
    return height_;
  }

  size_t channels() const noexcept {
    return channels_;
  }

  size_t size() const noexcept {
    return height() * width() * channels();
  }

  const Elem* data() const noexcept {
    return buffer_.data();
  }

  Elem* data() noexcept {
    return buffer_.data();
  }

private:
  size_t height_ = 0;
  size_t width_ = 0;
  size_t channels_ = 0;
  std::vector<elem_t> buffer_;
};

namespace jpeg_utils {
  Frame<uint8_t> read_from_file(const std::string& filename);
  void write_to_file(const Frame<uint8_t>& frame, const std::string& file);
}

}

