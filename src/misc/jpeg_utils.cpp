#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <iostream>

#include "jpeglib.h"

#include "frame.hpp"

namespace rpi_rt::jpeg_utils {

Frame<uint8_t> read_from_file(const std::string& filename) {
  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    throw std::runtime_error("cannot open file");
  }

  jpeg_decompress_struct cinfo;
  jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, fp);
  jpeg_read_header(&cinfo, true);
  cinfo.out_color_space = JCS_RGB;

  jpeg_start_decompress(&cinfo);

  rpi_rt::Frame<uint8_t> frame{cinfo.output_height, cinfo.output_width, 3};

  while (cinfo.output_scanline < cinfo.output_height) {
    uint8_t *row = frame.data() + cinfo.output_scanline * (cinfo.output_width) * 3;
    jpeg_read_scanlines(&cinfo, (JSAMPARRAY)&row, 1);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(fp);

  return frame;
}

void write_to_file(const Frame<uint8_t>& frame, const std::string& filename) {
  FILE *fp = fopen(filename.c_str(), "wb");
  if (!fp) {
    throw std::runtime_error("cannot open file");
  }

  jpeg_compress_struct cinfo;
  jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, fp);

  cinfo.image_width = frame.width();
  cinfo.image_height = frame.height();
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 80, true);
  jpeg_start_compress(&cinfo, true);

  while (cinfo.next_scanline < cinfo.image_height) {
    const uint8_t *row = frame.data() + cinfo.next_scanline * frame.width() * 3;
    jpeg_write_scanlines(&cinfo, (JSAMPARRAY)&row, 1);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  fclose(fp);
}

}

