#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace rpi_rt {

/**
 * The tensor container passed around by the vision sensor and detection logic.
 *
 * Assumes NHWC and N == 1, always use a compact mem layout.
 *
 * Adhere to the Single Responsibility Principle (SRP) in SOLID.
 */
template <class Elem>
class Frame {
public:
  using elem_t = Elem;

  /**
   * Default constructor.
   */
  Frame() {}

  /**
   * Construct and allocate a frame.
   *
   * @param height The height of frame
   * @param width The width of frame.
   * @param channels The channel count.
   */
  Frame(size_t height, size_t width, size_t channels) {
    resize(height, width, channels);
  }

  /**
   * Resize a frame.
   *
   * The underlying memory will also be resized. The content stored is
   * not guaranteed between resizes.
   *
   * @param height The height of frame
   * @param width The width of frame.
   * @param channels The channel count.
   */
  void resize(size_t height, size_t width, size_t channels) {
    width_ = width;
    height_ = height;
    channels_ = channels;
    buffer_.resize(size());
  }

  /**
   * The width of frame.
   */
  size_t width() const noexcept {
    return width_;
  }

  /**
   * The height of frame.
   */
  size_t height() const noexcept {
    return height_;
  }

  /**
   * The channel count.
   */
  size_t channels() const noexcept {
    return channels_;
  }

  /**
   * The total count of elements in this frame.
   */
  size_t size() const noexcept {
    return height() * width() * channels();
  }

  /**
   * Returns the immutable pointer.
   */
  const Elem* data() const noexcept {
    return buffer_.data();
  }

  /**
   * Returns the mutable pointer.
   */
  Elem* data() noexcept {
    return buffer_.data();
  }

private:
  size_t height_ = 0;
  size_t width_ = 0;
  size_t channels_ = 0;
  std::vector<elem_t> buffer_;
};

/// @cond PRIVATE_DETAILS

namespace jpeg_utils {
  Frame<uint8_t> read_from_file(const std::string& filename);
  void write_to_file(const Frame<uint8_t>& frame, const std::string& file);
  std::vector<uint8_t> write_to_mem(const Frame<uint8_t>& frame);
}

/// @endcond

}

