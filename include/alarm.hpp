#pragma once

#include <memory>
#include <functional>
#include <string>

#include "detection_result.hpp"

namespace rpi_rt {

  struct brevo_config_t {
    std::string api_host = "api.brevo.com";
    std::string api_path = "/v3/smtp/email";

    std::string api_key;
    std::string sender_name = "FlameIris Webhook";
    std::string sender_email;
    std::string to_email;
    std::string to_name = "FlameIris User";
  };

  class alarm_t {
    public:
      virtual ~alarm_t() {}
      virtual void run() = 0;
      virtual void report(std::unique_ptr<detection_result_t>) = 0;
      virtual void close() = 0;
  };

  std::shared_ptr<alarm_t> create_stdout_alarm();
  std::shared_ptr<alarm_t> create_brevo_email_alarm(brevo_config_t cfg);
}

