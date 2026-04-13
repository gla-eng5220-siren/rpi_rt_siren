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
#include "buzzer.h"
#include "frame.hpp"

namespace rpi_rt {
  class buzzer_alarm_t : public alarm_t {
    public:
      virtual ~buzzer_alarm_t() override {}
      explicit buzzer_alarm_t(int pin) : buzzer_(pin){}

      virtual void run() override {
        while (!closing_) {
          std::unique_lock lg{mut_result_};
          while(!closing_) {
            cond_result_.wait_for(lg, std::chrono::milliseconds{500});
            if (result_) {
              latency_assessment::report_timepoint(result_->frame_id(), true);
              if (result_->has_fire()) {
                std::cout << "FIRE DETECTED" << std::endl;

                lg.unlock();
                buzzer_.turnOn();
                std::this_thread::sleep_for(std::chrono::seconds(2));
                buzzer_.turnOff();
                lg.lock();
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

      Buzzer buzzer_;
  };

  std::shared_ptr<alarm_t> create_buzzer_alarm(int pin) {
    return std::make_shared<buzzer_alarm_t>(pin);
  }
}



