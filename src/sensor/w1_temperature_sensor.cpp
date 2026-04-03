#include "sensor.hpp"

#include <atomic>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace rpi_rt {

namespace {

std::string find_w1_sensor_directory(const std::string& base_dir) {
    DIR* dir = opendir(base_dir.c_str());
    if (!dir) {
        throw std::runtime_error("Failed to open W1 base directory: " + base_dir);
    }

    dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;

        if (name == "." || name == "..") {
            continue;
        }

        // DS18B20 folders usually start with "28-"
        if (name.rfind("28-", 0) == 0) {
            closedir(dir);
            return base_dir + "/" + name;
        }
    }

    closedir(dir);
    throw std::runtime_error("No DS18B20 sensor found under " + base_dir);
}

float read_celsius_from_temperature_file(const std::string& sensor_dir) {
    const std::string path = sensor_dir + "/temperature";
    std::ifstream fin(path);

    if (!fin.is_open()) {
        throw std::runtime_error("Failed to open temperature file: " + path);
    }

    long long milli_celsius = 0;
    fin >> milli_celsius;

    if (!fin) {
        throw std::runtime_error("Failed to parse temperature value from: " + path);
    }

    return static_cast<float>(milli_celsius) / 1000.0f;
}

float read_celsius_from_w1_slave(const std::string& sensor_dir) {
    const std::string path = sensor_dir + "/w1_slave";
    std::ifstream fin(path);

    if (!fin.is_open()) {
        throw std::runtime_error("Failed to open w1_slave file: " + path);
    }

    std::string line1;
    std::string line2;
    std::getline(fin, line1);
    std::getline(fin, line2);

    if (line1.find("YES") == std::string::npos) {
        throw std::runtime_error("CRC check failed in: " + path);
    }

    const std::size_t pos = line2.find("t=");
    if (pos == std::string::npos) {
        throw std::runtime_error("Temperature token 't=' not found in: " + path);
    }

    const long long milli_celsius = std::stoll(line2.substr(pos + 2));
    return static_cast<float>(milli_celsius) / 1000.0f;
}

float read_celsius(const std::string& sensor_dir) {
    try {
        return read_celsius_from_temperature_file(sensor_dir);
    } catch (...) {
        return read_celsius_from_w1_slave(sensor_dir);
    }
}

} // namespace

class w1_temperature_sensor_t : public temperature_sensor_t {
public:
    w1_temperature_sensor_t() = default;
    ~w1_temperature_sensor_t() override = default;

    void run() override {
        while (!closing_) {
            try {
                if (sensor_dir_.empty()) {
                    sensor_dir_ = find_w1_sensor_directory(base_dir_);
                }

                const float celsius = read_celsius(sensor_dir_);

                if (report_celsius_) {
                    report_celsius_(celsius);
                }
            } catch (const std::exception& e) {
                std::cerr << "[w1_temperature_sensor] " << e.what() << std::endl;

                // Clear cached directory so the next loop can try discovering again.
                sensor_dir_.clear();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }

    void close() override {
        closing_ = true;
    }

    void set_celsius_reciever(std::function<void(float)> callback) override {
        report_celsius_ = std::move(callback);
    }

private:
    std::function<void(float)> report_celsius_;
    std::atomic<bool> closing_{false};

    std::string base_dir_ = "/sys/bus/w1/devices";
    std::string sensor_dir_;
};

std::shared_ptr<temperature_sensor_t> create_w1_temperature_sensor() {
    return std::make_shared<w1_temperature_sensor_t>();
}

} // namespace rpi_rt
