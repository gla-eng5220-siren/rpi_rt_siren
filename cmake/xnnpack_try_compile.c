// Used in CMake configuration for detecting the system XNNPACK have the correct API

#include <xnnpack.h>

int main() {
  xnn_create_resize_bilinear2d_nhwc_u8(0, 0, 0, NULL);

  xnn_create_reduce_nd(0, 0, NULL, NULL, 0, NULL);

  xnn_create_convolution2d_nhwc_f32(
    0, 0, 0, 0,
    0, 0,
    0, 0,
    0, 0,
    0,
    0, 0,
    0, 0,
    NULL, NULL,
    0.0f, 0.0f,
    0,
    NULL, NULL,
    NULL);

  xnn_create_max_pooling2d_nhwc_f32(
    0, 0, 0, 0,
    0, 0,
    0, 0,
    0, 0,
    0.0f, 0.0f,
    0,
    NULL);

  xnn_create_fully_connected_nc_f32(
    0, 0,
    0, 0,
    NULL, NULL,
    0.0f, 0.0f,
    0,
    NULL, NULL,
    NULL);

  xnn_setup_resize_bilinear2d_nhwc_u8(NULL, NULL, NULL, NULL);

  xnn_setup_reduce_nd(NULL, NULL, NULL, NULL);

  xnn_setup_convolution2d_nhwc_f32(NULL, NULL, NULL, NULL);

  xnn_setup_max_pooling2d_nhwc_f32(NULL, NULL, NULL);

  xnn_setup_fully_connected_nc_f32(NULL, NULL, NULL);

  xnn_reshape_resize_bilinear2d_nhwc_u8(
    NULL,
    0, 0, 0, 0,
    0, 0,
    NULL, NULL,
    NULL);

  xnn_reshape_reduce_nd(
    NULL,
    0,
    NULL,
    0,
    NULL,
    NULL,
    NULL,
    NULL);

  xnn_reshape_convolution2d_nhwc_f32(
    NULL,
    0, 0, 0,
    NULL, NULL,
    NULL, NULL,
    NULL);

  xnn_reshape_max_pooling2d_nhwc_f32(
    NULL,
    0, 0, 0, 0,
    0, 0,
    NULL, NULL,
    NULL);

  xnn_reshape_fully_connected_nc_f32(NULL, 0, NULL);

  xnn_run_operator(NULL, NULL);

  return 0;
}

