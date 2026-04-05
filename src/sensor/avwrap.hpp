#pragma once

/*
 * This file provides a RAII wrapper for libav-related APIs.
 * Errors are handled via custom exceptions.
 */


extern "C" {

#include "libavutil/imgutils.h"
#include "libavutil/samplefmt.h"
#include "libavutil/timestamp.h"
#include "libavutil/error.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"

}

#include <iterator>
#include <optional>
#include <string>
#include <unordered_map>
#include <set>
#include <fstream>
#include <stdexcept>
#include <memory>
#include <chrono>
#include <filesystem>
#include <type_traits>
#include <string>
#include <chrono>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <vector>
#include <variant>
#include <ranges>

namespace rpi_rt::avwrap {

struct av_exception : std::exception {
  av_exception() {}

  explicit av_exception(int code)
  : error_code_(code)
  {
    (void)av_strerror(error_code_, buffer, sizeof(buffer));
  }

  const char* what() const noexcept override {
    return buffer;
  }

  char buffer[512] = "unknown error in ffmpeg";
  int error_code_ = 0;
};

template <class FunctionType> struct av_call_chk_impl {
  explicit av_call_chk_impl(FunctionType *fn) : fn(fn) {}

  template <class... Args> auto operator()(Args &&...args) {
    auto ret = std::invoke(fn, std::forward<Args>(args)...);
    if(ret < 0) {
      throw av_exception{ret};
    } else {
      return ret;
    }
  }

  FunctionType *fn;
};

template <class FunctionType> auto av_call_chk(FunctionType *fn) { return av_call_chk_impl(fn); }

template <class FunctionType> struct av_call_chkptr_impl {
  explicit av_call_chkptr_impl(FunctionType *fn) : fn(fn) {}

  template <class... Args> auto operator()(Args &&...args) {
    auto* ret = std::invoke(fn, std::forward<Args>(args)...);
    if(ret == nullptr) {
      throw av_exception{};
    } else {
      return ret;
    }
  }

  FunctionType *fn;
};

template <class FunctionType> auto av_call_chkptr(FunctionType *fn) { return av_call_chkptr_impl(fn); }

#define AV_FNDEF_CHK(__name) static auto __name##_chk = av_call_chk(__name)
#define AV_FNDEF_CHKPTR(__name) static auto __name##_chk = av_call_chkptr(__name)

AV_FNDEF_CHKPTR(sws_getContext);
AV_FNDEF_CHKPTR(av_frame_alloc);
AV_FNDEF_CHKPTR(av_packet_alloc);
AV_FNDEF_CHKPTR(avcodec_find_decoder);
AV_FNDEF_CHKPTR(avcodec_alloc_context3);

AV_FNDEF_CHK(av_image_alloc);
AV_FNDEF_CHK(avcodec_send_packet);
AV_FNDEF_CHK(av_find_best_stream);
AV_FNDEF_CHK(avcodec_parameters_to_context);
AV_FNDEF_CHK(avcodec_open2);
AV_FNDEF_CHK(avformat_open_input);
AV_FNDEF_CHK(avformat_find_stream_info);
AV_FNDEF_CHK(avformat_seek_file);
AV_FNDEF_CHK(sws_scale_frame);

template <class T, void (*F)(T*)>
struct av_deleter_wrapper {
  void operator()(T* p) {
    F(p);
  }
};

template <class T, void (*F)(T**)>
struct av_pp_deleter_wrapper {
  void operator()(T* p) {
    F(&p);
  }
};

template <class T, void (*PDeleter)(T*)>
using av_unique_ptr_with_pdel = std::unique_ptr<
  T, av_deleter_wrapper<T, PDeleter>>;
template <class T, void (*PpDeleter)(T**)>
using av_unique_ptr_with_ppdel = std::unique_ptr<
  T, av_pp_deleter_wrapper<T, PpDeleter>>;

using av_codec_context_ptr = av_unique_ptr_with_ppdel<
  AVCodecContext, avcodec_free_context>;
using av_input_format_context_ptr = av_unique_ptr_with_ppdel<
  AVFormatContext, avformat_close_input>;
using sws_context_ptr = av_unique_ptr_with_pdel<
  SwsContext, sws_freeContext>;
using av_frame_ptr = av_unique_ptr_with_ppdel<
  AVFrame, av_frame_free>;
using av_packet_ptr = av_unique_ptr_with_ppdel<
  AVPacket, av_packet_free>;

struct av_frame_data_guard {
  av_frame_data_guard() {}

  explicit av_frame_data_guard(AVFrame* ptr)
  : ptr_(ptr) {}

  explicit av_frame_data_guard(av_frame_data_guard&& o)
  : ptr_(o.ptr_)
  {
    o.ptr_ = nullptr;
  }

  ~av_frame_data_guard() {
    release();
  }

  av_frame_data_guard& operator=(av_frame_data_guard&& o){
    release();
    ptr_ = o.ptr_;
    o.ptr_ = nullptr;
    return *this;
  }

  void release() {
    if (ptr_ == nullptr) return;
    av_frame_unref(ptr_);
    ptr_ = nullptr;
  }

  AVFrame* ptr_ = nullptr;
};

struct av_packet_data_guard {
  av_packet_data_guard() {}

  explicit av_packet_data_guard(AVPacket* ptr)
  : ptr_(ptr) {}

  explicit av_packet_data_guard(av_packet_data_guard&& o)
  : ptr_(o.ptr_)
  {
    o.ptr_ = nullptr;
  }

  ~av_packet_data_guard() {
    release();
  }

  av_packet_data_guard& operator=(av_packet_data_guard&& o){
    release();
    ptr_ = o.ptr_;
    o.ptr_ = nullptr;
    return *this;
  }

  void release() {
    if (ptr_ == nullptr) return;
    av_packet_unref(ptr_);
    ptr_ = nullptr;
  }

  AVPacket* ptr_ = nullptr;
};

inline av_input_format_context_ptr avformat_open_input_wrap(
  const char *url,
  const AVInputFormat *fmt,
  AVDictionary **options) {
  AVFormatContext *p = nullptr;
  avformat_open_input_chk(&p, url, fmt, options);
  return av_input_format_context_ptr{p};
}

inline av_frame_data_guard sws_scale_frame_wrap(
  struct SwsContext *c, AVFrame *dst, const AVFrame *src) {
  sws_scale_frame_chk(c, dst, src);
  return av_frame_data_guard{dst};
}

}

