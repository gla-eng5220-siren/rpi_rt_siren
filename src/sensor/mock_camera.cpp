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

#include "sensor.hpp"
#include "frame.hpp"

#include "avwrap.hpp"

namespace rpi_rt {
  class mock_camera_t : public camera_sensor_t {
    public:
      mock_camera_t(std::string filename)
        : filename_(std::move(filename))
      {}

      virtual ~mock_camera_t() override {}

      virtual void run() override {
        format_init();
        codec_init();
        scaler_init();

        while (!closing_) {
          int ret = ::av_read_frame(fmt_ctx_.get(), pkt_.get());
          if (ret < 0) {
            if (ret == AVERROR_EOF) {
              seek_begin(); // loop over
              continue;
            }
            throw avwrap::av_exception{ret};
          }
          avwrap::av_packet_data_guard packet_dg{pkt_.get()};
          if (pkt_->stream_index == video_stream_index_) {
            avwrap::avcodec_send_packet_chk(video_dec_ctx_.get(), pkt_.get());
            int ret = ::avcodec_receive_frame(
              video_dec_ctx_.get(), frame_.get());
            if (ret < 0) {
              if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN)) // need more data to produce a frame
                continue;
              throw avwrap::av_exception{ret};
            }

            avwrap::av_frame_data_guard frame_dg{frame_.get()};
            auto dst_frame_dg =
              avwrap::sws_scale_frame_wrap(
                sws_ctx_.get(), dst_frame_.get(), frame_.get());

            invoke_callback();
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
          }
        }
      }

      virtual void close() override {
        closing_ = true;
      }

      virtual void set_frame_callback(std::function<void (Frame<uint8_t>)> callback) override {
        callback_ = callback;
      }

    private:
      void format_init() {
        fmt_ctx_ = avwrap::avformat_open_input_wrap(filename_.c_str(), nullptr, nullptr);
        avwrap::avformat_find_stream_info_chk(fmt_ctx_.get(), nullptr);
      }

      void codec_init() {
        video_stream_index_ = avwrap::av_find_best_stream_chk(
          fmt_ctx_.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        AVStream *st = fmt_ctx_->streams[video_stream_index_];
        const AVCodec *dec = avwrap::avcodec_find_decoder_chk(st->codecpar->codec_id);
        video_dec_ctx_.reset(avwrap::avcodec_alloc_context3_chk(dec));

        avwrap::avcodec_parameters_to_context_chk(video_dec_ctx_.get(), st->codecpar);
        avwrap::avcodec_open2_chk(video_dec_ctx_.get(), dec, nullptr);

        frame_.reset(avwrap::av_frame_alloc_chk());
        pkt_.reset(avwrap::av_packet_alloc_chk());

        width_ = video_dec_ctx_->width;
        height_ = video_dec_ctx_->height;
        src_pix_fmt_ = video_dec_ctx_->pix_fmt;
      }

      void scaler_init() {
        sws_ctx_.reset(avwrap::sws_getContext_chk(
              width_, height_, src_pix_fmt_,
              width_, height_,
              AV_PIX_FMT_RGB24, SWS_FAST_BILINEAR,
              nullptr, nullptr, nullptr));
        dst_frame_.reset(avwrap::av_frame_alloc_chk());
      }

      void seek_begin() {
        avwrap::avformat_seek_file_chk(fmt_ctx_.get(), video_stream_index_,
            INT64_MIN, 0, INT64_MAX, 0);
        ::avcodec_flush_buffers(video_dec_ctx_.get());
      }

      void invoke_callback() {
        Frame<uint8_t> frame{height_, width_, 3};

        const uint8_t* data = dst_frame_->buf[0]->data;
        size_t sz = static_cast<size_t>(
            dst_frame_->linesize[0] * dst_frame_->height);

        assert(sz == frame.size()); // TODO does this always hold?
        std::memcpy(frame.data(), data, frame.size());
        callback_(std::move(frame));
      }

      std::string filename_;

      int video_stream_index_ = -1;
      AVPixelFormat src_pix_fmt_ = AV_PIX_FMT_NONE;

      avwrap::av_packet_ptr pkt_;
      avwrap::av_frame_ptr frame_;
      avwrap::av_frame_ptr dst_frame_;

      avwrap::av_input_format_context_ptr fmt_ctx_;
      avwrap::av_codec_context_ptr video_dec_ctx_;
      avwrap::sws_context_ptr sws_ctx_; // for potential color space coversion

      std::function<void (Frame<uint8_t>)> callback_;
      std::atomic<bool> closing_ = ATOMIC_VAR_INIT(false);
      size_t height_ = 0;
      size_t width_ = 0;
  };

  std::shared_ptr<camera_sensor_t> create_mock_camera_sensor(const std::string& filename) {
    return std::make_shared<mock_camera_t>(filename);
  }

}


