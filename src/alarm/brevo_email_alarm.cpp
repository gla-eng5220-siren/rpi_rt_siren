#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <iostream>
#include <atomic>

#include "alarm.hpp"
#include "detection_result.hpp"
#include "frame.hpp"

#define CPPHTTPLIB_OPENSSL_SUPPORT
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "third_party/httplib.h"
#pragma GCC diagnostic pop
#include "third_party/json.hpp"
#include "third_party/base64.hpp"

namespace rpi_rt {
  bool brevo_send_email(const brevo_config_t& cfg, rpi_rt::detection_result_t& result) {
    if (!result.has_fire())
      return false;

    httplib::SSLClient cli(cfg.api_host);
    httplib::Headers headers = {
        {"accept", "application/json"},
        {"api-key", cfg.api_key},
        {"content-type", "application/json"}
    };

    std::ostringstream title_oss; 
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    title_oss << "FlameIris Detection Warning Triggered At " << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");

    std::ostringstream html_oss;
    html_oss << "<html><head></head><body><h1>" << title_oss.str()
      << "</h1><h3>Details:</h3><p>" << result.explain() << "</p></body></html>";

    nlohmann::json j;
    j["sender"] = {
      {"name", cfg.sender_name },
      {"email", cfg.sender_email }
    };
    j["to"] = nlohmann::json::array();
    j["to"].push_back({
      {"name", cfg.to_name },
      {"email", cfg.to_email }
    });
    j["subject"] = title_oss.str();
    j["htmlContent"] = html_oss.str();

    auto jpg_attachment = result.jpg_attachment();
    if (jpg_attachment.size()) {
      j["attachment"] = nlohmann::json::array();
      j["attachment"].push_back({
        {"name", "attach.jpg"},
        {"content", base64::encode_into<std::string>(
            jpg_attachment.begin(),
            jpg_attachment.end())}
      });
    }

    auto res = cli.Post("/v3/smtp/email", headers, j.dump(), "application/json");
    if (!res) {
      // It might be a design flaw of httplib it doesn't use error_category or polymorphism here
      // We'll have to play by it however.
      std::cerr << "Brevo API invoke failed: HTTP: " << httplib::to_string(res.error())
        << " SSL: " << res.ssl_error()
        << " SSL Backend: " << res.ssl_backend_error() << std::endl;
      return false;
    }

    std::cout << "Brevo API Invoke: Status: " << res->status
      << " Body: " << res->body << std::endl;
    return res->status == httplib::OK_200;
  }

  class brevo_email_alarm_t : public alarm_t {
    public:
      brevo_email_alarm_t(brevo_config_t cfg)
        : cfg_(std::move(cfg)) {}
      virtual ~brevo_email_alarm_t() override {}

      virtual void run() override {
        while (!closing_) {
          std::unique_lock lg{mut_result_};
          while(!closing_) {
            cond_result_.wait_for(lg, std::chrono::milliseconds{500});
            if (result_) {
              latency_assessment::report_timepoint(result_->frame_id(), true);
              if (result_->has_fire()) {
                if (last_send_) {
                  auto now = std::chrono::steady_clock::now();
                  auto dur = now - *last_send_;
                  if (dur > std::chrono::hours{2}) {
                    (void)brevo_send_email(cfg_, *result_);
                    last_send_ = std::chrono::steady_clock::now();
                  }
                } else {
                  (void)brevo_send_email(cfg_, *result_);
                  last_send_ = std::chrono::steady_clock::now();
                }
              }
              result_ = nullptr;
            }
          }
        }
      }

      virtual void close() override {
        closing_ = true;
      }

      virtual void report(std::unique_ptr<detection_result_t> result) override {
        {
          std::unique_lock lg{mut_result_};
          result_ = std::move(result);
        }
        cond_result_.notify_one();
      }

    private:
      std::atomic<bool> closing_ = ATOMIC_VAR_INIT(false);
      std::unique_ptr<detection_result_t> result_;
      std::mutex mut_result_;
      std::condition_variable cond_result_;
      std::optional<std::chrono::steady_clock::time_point> last_send_ = std::nullopt;
      brevo_config_t cfg_;
  };

  std::shared_ptr<alarm_t> create_brevo_email_alarm(brevo_config_t cfg) {
    return std::make_shared<brevo_email_alarm_t>(std::move(cfg));
  }
}



