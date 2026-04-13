#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <atomic>

#include "http_server.hpp"
#include "frame.hpp"
#include "third_party/httplib.h"

namespace rpi_rt {
  class http_server_impl: public http_server_t {
    public:
      virtual ~http_server_impl() override {}

      virtual void setup(const std::string& webui_path) override {
        svr_.Get("/cam", [this](
              const httplib::Request& req, httplib::Response& res) {
          this->handle_cam_request(req, res);
        });
        svr_.Get("/logit", [this](
              const httplib::Request& req, httplib::Response& res) {
          this->handle_logit_request(req, res);
        });
        svr_.set_default_headers({
            {"Access-Control-Allow-Origin", "*"},
            {"Access-Control-Allow-Methods", "GET, HEAD, OPTIONS"},
            {"Access-Control-Allow-Headers", "*"}
        });
        auto ret = svr_.set_mount_point("/", webui_path);
        if (!ret) {
          throw std::runtime_error("Specified webui path cannot be mounted");
        }
      }

      virtual void run(const std::string& host, int port) override {
        svr_.listen(host, port);
      }

      virtual void close() override {
        running_ = false;

        // first wake up pool threads blocking on next frame
        cam_frame_cond_.notify_all();
        logit_cond_.notify_all();

        svr_.stop();
      }

      virtual void set_cam_frame(Frame<uint8_t> frame) override {
        {
          std::unique_lock lg{cam_frame_mut_};
          cam_frame_ = std::move(frame);
        }
        cam_frame_cond_.notify_all();
      }

      virtual void set_logit(float logit) override {
        {
          std::unique_lock lg{logit_mut_};
          logit_ = logit;
        }
        logit_cond_.notify_all();
      }

    private:
      void handle_cam_request(const httplib::Request& req, httplib::Response& res) {
        res.set_header("Connection", "close");
        res.set_header("Cache-Control", "no-cache");

        res.set_content_provider(
            "multipart/x-mixed-replace; boundary=MJF",
            [this](size_t sz, httplib::DataSink& sink) {
              return this->provide_cam_content(sz, sink);
            });
      }

      bool provide_cam_content(size_t, httplib::DataSink& sink) {
        while (running_ && sink.is_writable()) {
          std::unique_lock lg{cam_frame_mut_};
          std::vector<uint8_t> data = jpeg_utils::write_to_mem(cam_frame_);

          sink.os << "--MJF\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: " << data.size() << "\r\n\r\n";
          sink.os.flush();
          sink.write(reinterpret_cast<const char*>(data.data()), data.size());

          cam_frame_cond_.wait_for(lg, std::chrono::milliseconds{500});
        }

        sink.done();
        return true;
      }

      void handle_logit_request(const httplib::Request& req, httplib::Response& res) {
        res.set_header("Connection", "keep-alive");
        res.set_header("Cache-Control", "no-cache");

        res.set_content_provider(
            "text/event-stream",
            [this](size_t sz, httplib::DataSink& sink) {
              return this->provide_logit_content(sz, sink);
            });
      }

      bool provide_logit_content(size_t, httplib::DataSink& sink) {
        while (running_ && sink.is_writable()) {
          std::unique_lock lg{logit_mut_};
          if (logit_ != INFINITY) { // only start to steam after first detection
            sink.os << "data: " << logit_ << "\r\n\r\n";
            sink.os.flush();
          }

          logit_cond_.wait_for(lg, std::chrono::milliseconds{500});
        }

        sink.done();
        return true;
      }

      Frame<uint8_t> cam_frame_;
      std::mutex cam_frame_mut_;
      std::condition_variable cam_frame_cond_;

      float logit_ = INFINITY;
      std::mutex logit_mut_;
      std::condition_variable logit_cond_;

      httplib::Server svr_;

      std::atomic<bool> running_ = ATOMIC_VAR_INIT(true);
  };

  std::shared_ptr<http_server_t> create_http_server() {
    return std::make_shared<http_server_impl>();
  }
}



