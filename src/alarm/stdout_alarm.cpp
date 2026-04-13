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

namespace rpi_rt {
  class stdout_alarm_t : public alarm_t {
    public:
      virtual ~stdout_alarm_t() override {}

      virtual void run() override {
        while (!closing_) {
          std::unique_lock lg{mut_result_};
          while(!closing_) {
            cond_result_.wait_for(lg, std::chrono::milliseconds{500});
            if (result_) {
              latency_assessment::report_timepoint(result_->frame_id(), true);
              std::cout << result_->explain() << std::endl;
              if (result_->has_fire()) {
                std::cout << "FIRE DETECTED" << std::endl;
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
  };

  std::shared_ptr<alarm_t> create_stdout_alarm() {
    return std::make_shared<stdout_alarm_t>();
  }
}



