#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <cassert>

#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include "libv4l2.h"

#include "sensor.hpp"
#include "frame.hpp"

namespace rpi_rt {
  class v4l2_camera_sensor_t : public camera_sensor_t {
    public:
      virtual ~v4l2_camera_sensor_t() override {}

      virtual void run() override {
        open("/dev/video0");
        negotiate_format();
        request_buffers(buffer_count_);
        for (size_t i = 0; i < buffer_count_; i++) {
          auto buffer = perform_mmap(i);
          buffers_[buffer.index] = buffer;
        }
        for (const auto& [index, buffer] : buffers_) {
          qbuf(index);
        }

        stream_on();

        while (!closing_) {
          wait_for_frame();
          unsigned index = dqbuf();
          invoke_callback(buffers_[index]);
          qbuf(index);
        }

        stream_off();
      }

      virtual void close() override {
        closing_ = true;
      }

      virtual void set_frame_callback(std::function<void (Frame<uint8_t>)> callback) override {
        callback_ = callback;
      }

    private:
      struct mmap_buffer_t {
        unsigned index = 0;
        uint8_t *data = nullptr;
        size_t size = 0;
      };

      void open(const std::string& dev_name) {
        fd_ = v4l2_open(dev_name.c_str(), O_RDWR | O_NONBLOCK, 0);
        if (fd_ < 0) {
          throw std::runtime_error("v4l2_open failed");
        }
      }

      void xioctl(int request, void *arg)
      {
        int r;

        do {
          r = v4l2_ioctl(fd_, request, arg);
        } while (r == -1 && ((errno == EINTR) || (errno == EAGAIN)));

        if (r == -1) {
          throw std::system_error(std::make_error_code(std::errc(errno)));
        }
      }

      void negotiate_format() {
        v4l2_format fmt;
        std::memset(&fmt, 0, sizeof(fmt));

        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width       = 640;
        fmt.fmt.pix.height      = 480;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
        fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;
        xioctl(VIDIOC_S_FMT, &fmt);
        if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_RGB24) {
          throw std::runtime_error("v4l2: RGB24 not supported by webcam");
        }
        height_ = fmt.fmt.pix.height;
        width_ = fmt.fmt.pix.width;
      }

      void request_buffers(size_t count) {
        v4l2_requestbuffers req;
        std::memset(&req, 0, sizeof(req));

        req.count = count;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        xioctl(VIDIOC_REQBUFS, &req);
      }

      void do_buf_xioctl(int request, unsigned index, v4l2_buffer& buf) {
        std::memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = index;
        xioctl(request, &buf);
      }

      mmap_buffer_t perform_mmap(unsigned index) {
        v4l2_buffer buf;
        do_buf_xioctl(VIDIOC_QUERYBUF, index, buf);

        mmap_buffer_t mmap_buffer;
        mmap_buffer.index = index;
        mmap_buffer.size = buf.length;

        mmap_buffer.data = (uint8_t *)v4l2_mmap(NULL, buf.length,
            PROT_READ | PROT_WRITE, MAP_SHARED,
            fd_, buf.m.offset);
        if (MAP_FAILED == mmap_buffer.data) {
          throw std::runtime_error("v4l2_mmap failed");
        }

        return mmap_buffer;
      }

      void qbuf(unsigned index) {
        v4l2_buffer buf;
        do_buf_xioctl(VIDIOC_QBUF, index, buf);
      }

      unsigned dqbuf() {
        v4l2_buffer buf;
        do_buf_xioctl(VIDIOC_DQBUF, 0, buf);
        return buf.index;
      }

      void stream_on() {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(VIDIOC_STREAMON, &type);
      }

      void stream_off() {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(VIDIOC_STREAMOFF, &type);
      }

      void wait_for_frame() {
        fd_set fds;
        timeval tv;
        int r;
        do {
          FD_ZERO(&fds);
          FD_SET(fd_, &fds);

          /* Timeout. */
          tv.tv_sec = 2;
          tv.tv_usec = 0;

          r = select(fd_ + 1, &fds, NULL, NULL, &tv);
        } while ((r == -1 && (errno = EINTR)));
        if (r == -1) {
          throw std::runtime_error("select() failed");
        }
      }

      void invoke_callback(const mmap_buffer_t& buffer) {
        Frame<uint8_t> frame{height_, width_, 3};

        assert(buffer.size >= frame.size());
        std::memcpy(frame.data(), buffer.data, frame.size());
        callback_(std::move(frame));
      }

      std::function<void (Frame<uint8_t>)> callback_;
      std::atomic<bool> closing_ = ATOMIC_VAR_INIT(false);
      size_t height_ = 0;
      size_t width_ = 0;
      std::map<unsigned, mmap_buffer_t> buffers_;
      int fd_ = -1;

      static constexpr size_t buffer_count_ = 10;
  };

  std::shared_ptr<camera_sensor_t> create_v4l2_camera_sensor() {
    return std::make_shared<v4l2_camera_sensor_t>();
  }

}


