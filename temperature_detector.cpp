#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

struct detection_result_t {
    bool alarm_triggered;
    float temperature_celsius;
    std::string message;
};

class temperature_sensor_t {
public:
    virtual ~temperature_sensor_t() = default;
    virtual float read_celsius() = 0;
};

class mock_temperature_sensor_t : public temperature_sensor_t {
public:
    explicit mock_temperature_sensor_t(const std::vector<float>& samples)
        : samples_(samples), index_(0) {}

    float read_celsius() override {
        if (samples_.empty()) {
            return 25.0f;
        }
        float value = samples_[index_];
        index_ = (index_ + 1) % samples_.size();
        return value;
    }

private:
    std::vector<float> samples_;
    std::size_t index_;
};

class temperature_threshold_logic_t {
public:
    explicit temperature_threshold_logic_t(float threshold = 60.0f)
        : threshold_celsius_(threshold) {}

    detection_result_t evaluate(float current_temp) const {
        detection_result_t result;
        result.temperature_celsius = current_temp;

        if (current_temp >= threshold_celsius_) {
            result.alarm_triggered = true;
            result.message = "Fire risk detected";
        } else {
            result.alarm_triggered = false;
            result.message = "Safe";
        }

        return result;
    }

private:
    float threshold_celsius_;
};

class stdout_alarm_t {
public:
    void report(const detection_result_t& result) const {
        if (result.alarm_triggered) {
            std::cout << "[ALARM] ";
        } else {
            std::cout << "[INFO] ";
        }

        std::cout << result.message
                  << " | Temperature: "
                  << result.temperature_celsius
                  << " °C"
                  << std::endl;
    }
};

int main() {
    auto sensor = std::make_shared<mock_temperature_sensor_t>(
        std::vector<float>{32.0f, 45.0f, 58.0f, 63.0f, 70.0f, 55.0f}
    );

    temperature_threshold_logic_t logic(60.0f);
    stdout_alarm_t alarm;

    for (int i = 0; i < 6; ++i) {
        float temp = sensor->read_celsius();
        detection_result_t result = logic.evaluate(temp);
        alarm.report(result);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
