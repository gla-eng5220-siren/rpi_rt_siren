#include <chrono>
#include <functional>
#include <libcamera/camera.h>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <cassert>
#include <mutex>
#include <condition_variable>

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>

#include "libcamera/libcamera.h"
#include "avwrap.hpp"

#include "sensor.hpp"
#include "frame.hpp"

namespace rpi_rt {
  class libcamera_sensor_t : public camera_sensor_t {
    public:
      explicit libcamera_sensor_t(unsigned cam_index)
        : cam_index_(cam_index) {}

      virtual ~libcamera_sensor_t() override {}

      virtual void run() override {
        std::unique_lock lg{mut_camera_};
        start_libcamera();
        scaler_init();
        while(!closing_) {
          cond_camera_.wait_for(lg, std::chrono::milliseconds{500});
        }
        std::cerr << "???" << std::endl;
      }

      virtual void close() override {
        {
          std::unique_lock lg{mut_camera_};
          closing_ = true;
          stop_libcamera();
        }
        cond_camera_.notify_one();
      }

      virtual void set_frame_callback(std::function<void (Frame<uint8_t>)> callback) override {
        callback_ = callback;
      }

    private:
      void request_complete(libcamera::Request *request) {
        if (request->status() == libcamera::Request::RequestCancelled)
          return;

        const auto& buffers = request->buffers();
        for (auto buffer_pair : request->buffers()) {
          auto buffer = buffer_pair.second;

          // YUYV should have only 1 plane
          auto planes = buffer->planes();
          if (planes.size() == 0)
            continue;
          if (planes.size() > 1) {
            std::cerr << "WARNING: got multi planes for YUYV" << std::endl;
          }
          const auto& plane = *planes.cbegin();
          invoke_callback(plane);
        }

        request->reuse(libcamera::Request::ReuseBuffers);
        camera_->queueRequest(request);
      }

      void invoke_callback(const libcamera::FrameBuffer::Plane& plane) {
        Frame<uint8_t> frame{height_, width_, 3};

        const uint8_t* data = fd_ptrs_[plane.fd.get()] + plane.offset;
        const size_t size = plane.length;

        assert(size == stride_ * height_);
        const uint8_t *src_data[4] = { data, NULL, NULL, NULL };
        int src_linesize[4] = { (int)stride_, 0, 0, 0 };
        uint8_t *dst_data[4] = { frame.data(), NULL, NULL, NULL };
        int dst_linesize[4] = { (int)(frame.width() * frame.channels()), 0, 0, 0 };
        avwrap::sws_scale_chk(sws_ctx_.get(), src_data, src_linesize, 0, height_, dst_data, dst_linesize); 
        callback_(std::move(frame));
      }

      void start_libcamera() {
        cm_  = std::make_unique<libcamera::CameraManager>();
        cm_->start();
	
        std::cerr << "Cams:" << std::endl;
        unsigned index = 0;
        for (auto const &camera : cm_->cameras()) {
          auto maybe_model = camera->properties().get(libcamera::properties::Model);
          std::string_view model = maybe_model.has_value() ? *maybe_model : "??";
          std::cerr << " - " << index++ << ": " << model << " " << camera.get()->id() << std::endl;
        }
	
        if (cm_->cameras().empty())
          throw std::runtime_error("libcamera_sensor_t: no cameras identified");

        if (cam_index_ >= cm_->cameras().size())
          throw std::runtime_error("libcamera_sensor_t: camera index out of range");

        std::string camera_id = cm_->cameras()[cam_index_]->id();
        std::cout << "Using camera : " << camera_id << std::endl;
        camera_ = cm_->get(camera_id);
        camera_->acquire();

        config_ = camera_->generateConfiguration( { libcamera::StreamRole::Viewfinder } );
        libcamera::StreamConfiguration &stream_config = config_->at(0);
        stream_config.pixelFormat = libcamera::formats::YUYV;
        auto status = config_->validate();
        if (status == libcamera::CameraConfiguration::Invalid)
          throw std::runtime_error("libcamera_sensor_t: camera config invalid");
        if (stream_config.pixelFormat != libcamera::formats::YUYV)
          throw std::runtime_error("libcamera_sensor_t: format not supported (requires YUYV)");
        height_ = stream_config.size.height;
        width_ = stream_config.size.width;
        stride_ = stream_config.stride;
        std::cerr << "Stride: " << stride_ << std::endl;

        int ret = camera_->configure(config_.get());
        if (ret < 0)
          throw std::runtime_error("libcamera_sensor_t: stream config failed");

        allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
        for (auto &cfg : *config_) {
          int ret = allocator_->allocate(cfg.stream());
          if (ret < 0)
            throw std::runtime_error("libcamera_sensor_t: buffer allocate failed");
        }

        stream_ = stream_config.stream();
        const auto& buffers = allocator_->buffers(stream_);

        for(const auto& buffer : buffers) {
          for (const auto& plane : buffer->planes()) {
            perform_mmap(plane.fd.get());
          }
        }

        for (unsigned int i = 0; i < buffers.size(); ++i) {
          auto request = camera_->createRequest();
          const auto& buffer = buffers[i];
          int ret = request->addBuffer(stream_, buffer.get());
          if (ret < 0)
            throw std::runtime_error("libcamera_sensor_t: add buffer to request failed");

          requests_.push_back(std::move(request));
        }

        controls_.set(libcamera::controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({ 100000, 100000 }));
        controls_.set(libcamera::controls::Brightness, 0.0f);
        controls_.set(libcamera::controls::Contrast, 1.0f);

        camera_->requestCompleted.connect(this, &libcamera_sensor_t::request_complete);

        camera_->start(&controls_);
        for (auto &request : requests_)
          camera_->queueRequest(request.get());
      }

      void stop_libcamera() {
        camera_->stop();
        allocator_->free(stream_);
        allocator_ = nullptr;
        camera_->release();
        camera_ = nullptr;
        cm_->stop();
      }

      void perform_mmap(int fd) {
        if (fd_ptrs_.count(fd))
          return;
        const size_t length = lseek(fd, 0, SEEK_END);
        uint8_t *data = (uint8_t *)mmap(NULL, length,
            PROT_READ | PROT_WRITE, MAP_SHARED,
            fd, 0);
        if (!data) {
          throw std::runtime_error("mmap failed");
        }
        fd_ptrs_[fd] = data;
      }

      void scaler_init() {
        sws_ctx_.reset(avwrap::sws_getContext_chk(
              width_, height_, AV_PIX_FMT_YUYV422,
              width_, height_,
              AV_PIX_FMT_RGB24, SWS_FAST_BILINEAR,
              nullptr, nullptr, nullptr));
      }

      const unsigned cam_index_ = 0;
      std::function<void (Frame<uint8_t>)> callback_;
      std::atomic<bool> closing_ = ATOMIC_VAR_INIT(false);
      size_t height_ = 0;
      size_t width_ = 0;
      size_t stride_ = 0;

      std::shared_ptr<libcamera::Camera> camera_;
      std::unique_ptr<libcamera::CameraConfiguration> config_;
      std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
      std::unique_ptr<libcamera::CameraManager> cm_;
      std::vector<std::unique_ptr<libcamera::Request>> requests_;
      libcamera::Stream *stream_ = nullptr;
      libcamera::ControlList controls_;

      std::map<int, uint8_t *> fd_ptrs_;

      std::mutex mut_camera_;
      std::condition_variable cond_camera_;

      avwrap::sws_context_ptr sws_ctx_; // for yuyv422 -> rgb24
  };

  std::shared_ptr<camera_sensor_t> create_libcamera_sensor(unsigned cam_index) {
    return std::make_shared<libcamera_sensor_t>(cam_index);
  }

}


