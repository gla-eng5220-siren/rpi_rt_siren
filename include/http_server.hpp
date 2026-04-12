#pragma once

#include <memory>
#include <cstdint>

#include "frame.hpp"

namespace rpi_rt {

/** \addtogroup WebUI
 *  @{
 */

  /**
   * The base class for WebUI http server.
   *
   * Adhere to the Liskov Substitution Principle (LSP) and Interface Segregation
   * Principle (ISP) in SOLID.
   */
  class http_server_t {
    public:
      virtual ~http_server_t() {}

      /**
       * Setup the server and allocate the internal resources.
       *
       * @param webui_path Path to static webui files. Usually PROJECT_ROOT/webui
       */
      virtual void setup(const std::string& webui_path) = 0;

      /**
       * Start the server and listen for web requests.
       *
       * Blocks until stopped. Should run on a thread.
       *
       * @param host The host to listen.
       * @param port The TCP port to listen.
       */
      virtual void run(const std::string& host, int port) = 0;

      /**
       * Stop the run function and return.
       */
      virtual void close() = 0;

      /**
       * Report the camera frame.
       *
       * This data will be served through MJPEG via HTTP.
       */
      virtual void set_cam_frame(Frame<uint8_t>) = 0;

      /**
       * Report the current logits of detection.
       *
       * This data will be served throught server-sent events.
       */
      virtual void set_logit(float) = 0;
  };

  /**
   * The factory method for creating a very simple httplib-based
   * WebUI HTTP server.
   *
   * Adhere to the Interface Segregation Principle (ISP) in SOLID.
   */
  std::shared_ptr<http_server_t> create_http_server();

/** @}*/

}

