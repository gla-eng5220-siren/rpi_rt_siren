#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <iostream>
#include <atomic>

#include "alarm.hpp"

namespace rpi_rt {
  class stdout_alarm_t : public alarm_t {
    public:
      virtual ~stdout_alarm_t() override {}

      virtual void run() override {
        while (!closing_) {
          std::unique_lock lg{mut_on_fire_};
          while(!closing_) {
            cond_on_fire_.wait_for(lg, std::chrono::milliseconds{500}, [this](){ return this->on_fire_; });
            if (on_fire_) {
              std::cout << "Oh no, the house is on fire!! Do something NOW!!" << std::endl;
              on_fire_ = false;
            }
          }
        }
      }

      virtual void close() override {
        closing_ = true;
      }

      virtual void report_fire() override {
        {
          std::unique_lock lg{mut_on_fire_};
          on_fire_ = true;
        }
        cond_on_fire_.notify_one();
      }

    private:
      std::atomic<bool> closing_ = ATOMIC_VAR_INIT(false);
      bool on_fire_ = false;
      std::mutex mut_on_fire_;
      std::condition_variable cond_on_fire_;
  };

  std::shared_ptr<alarm_t> create_stdout_alarm() {
    return std::make_shared<stdout_alarm_t>();
  }
}



