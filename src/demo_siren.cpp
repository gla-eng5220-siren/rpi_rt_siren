#include "alarm.hpp"
#include "sensor.hpp"
#include <thread>
#include <iostream>

int main(void) {
  auto alarm = rpi_rt::create_stdout_alarm();
  auto sensor = rpi_rt::create_mock_temperature_sensor();

  sensor->set_celsius_reciever([&alarm](float degrees) {
    std::cout << "Temperature in degree celcius:  " << degrees << std::endl;
    if (degrees > 200.0f) {
      alarm->report_fire();
    }
  });

  std::thread alarm_thread([self = alarm](){
    self->run();
  });
  std::thread sensor_thread([self = sensor](){
    self->run();
  });

  std::this_thread::sleep_for(std::chrono::seconds{5});

  alarm->close();
  sensor->close();
  alarm_thread.join();
  sensor_thread.join();
  return 0;
}

