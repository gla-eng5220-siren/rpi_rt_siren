#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

#include "third_party/i2c.h"

#include "sensor.hpp"
#include "frame.hpp"

namespace rpi_rt {

class breadpi_temperature_sensor_t : public temperature_sensor_t {
public:
    explicit breadpi_temperature_sensor_t(const breadpi_ntc_config_t& cfg)
      : cfg_(cfg)
    {}

    virtual ~breadpi_temperature_sensor_t() override {
        if (bus_ >= 0) {
            i2c_close(bus_);
            bus_ = -1;
        }
    }

    virtual void run() override {
        setup();
        while (!closing_) {
            uint64_t frame_id = latency_assessment::make_frame_id();
            latency_assessment::report_timepoint(frame_id);

            try {
                uint8_t raw = read_adc_raw();
                float celsius = raw_to_celsius(raw);

                if (report_celsius_) {
                    report_celsius_(frame_id, celsius);
                }

                std::cout << "[breadpi-temp] raw=" << static_cast<int>(raw)
                          << " temp=" << celsius << " C" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "[breadpi-temp] " << e.what() << std::endl;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    virtual void close() override {
        closing_ = true;
    }

    virtual void set_celsius_reciever(std::function<void(uint64_t frame_id, float)> callback) override {
        report_celsius_ = callback;
    }

private:
    uint8_t read_adc_raw() {
        uint8_t ctrl = static_cast<uint8_t>(0x40 + cfg_.adc_channel);
        if (i2c_ioctl_write(&device_, 0, &ctrl, 1) != 1) {
            throw std::runtime_error("i2c_ioctl_write failed");
        }
        // perform a dummy read to be safe
        uint8_t dummy = 0;
        if (i2c_ioctl_read(&device_, 0, &dummy, 1) != 1) {
            throw std::runtime_error("dummy read failed");
        }

        uint8_t raw = 0;
        if (i2c_ioctl_read(&device_, 0, &raw, 1) != 1) {
            throw std::runtime_error("i2c read failed");
        }

        return raw;
    }

    float raw_to_celsius(uint8_t raw) {
        float v = (static_cast<float>(raw) / 255.0f) * cfg_.vref;
        v = std::clamp(v, 0.001f, cfg_.vref - 0.001f);

        float r_ntc = 0.0f;
        if (cfg_.ntc_top) {
            // 3V3 -> NTC -> ADC -> Rfixed -> GND
            r_ntc = cfg_.r_fixed * (cfg_.vref / v - 1.0f);
        } else {
            // 3V3 -> Rfixed -> ADC -> NTC -> GND
            r_ntc = cfg_.r_fixed * v / (cfg_.vref - v);
        }
        const float t0 = 25.0f + 273.15f;
        const float temp_k =
            1.0f / ((1.0f / t0) + (1.0f / cfg_.beta) * std::log(r_ntc / cfg_.r_25));

        return temp_k - 273.15f;
    }

    void setup() {
      bus_ = i2c_open(cfg_.i2c_bus.c_str());
      if (bus_ == -1) {
        throw std::runtime_error("failed to open i2c bus");
      }

      i2c_init_device(&device_);
      device_.bus = bus_;
      device_.addr = cfg_.i2c_addr;
      device_.tenbit = 0;
      device_.delay = 1;
      device_.flags = 0;
      device_.page_bytes = 1;
      device_.iaddr_bytes = 0;
    }

private:
    int bus_ = -1;
    breadpi_ntc_config_t cfg_;
    I2CDevice device_{};
    std::function<void(uint64_t frame_id, float)> report_celsius_;
    std::atomic<bool> closing_{false};
};

std::shared_ptr<temperature_sensor_t> create_breadpi_temperature_sensor(const breadpi_ntc_config_t& cfg) {
    return std::make_shared<breadpi_temperature_sensor_t>(cfg);
}

} // namespace rpi_rt
