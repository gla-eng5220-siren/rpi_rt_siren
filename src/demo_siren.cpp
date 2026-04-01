#include "alarm.hpp"
#include "sensor.hpp"

#include <chrono>
#include <iostream>
#include <thread>

int main(void) {
    auto alarm = rpi_rt::create_stdout_alarm();
    auto sensor = rpi_rt::create_w1_temperature_sensor();

    sensor->set_celsius_reciever([&alarm](float degrees) {
        std::cout << "We got a celsius degree: " << degrees << std::endl;

        if (degrees > 60.0f) {
            alarm->report_fire();
        }
    });

    std::thread alarm_thread([self = alarm]() {
        self->run();
    });

    std::thread sensor_thread([self = sensor]() {
        self->run();
    });

    std::this_thread::sleep_for(std::chrono::seconds{30});

    alarm->close();
    sensor->close();

    alarm_thread.join();
    sensor_thread.join();

    return 0;
}
