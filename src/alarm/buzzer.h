#include <iostream>
#include <optional>
#include <gpiod.hpp>

class Buzzer {
public:
    // Constructor: Sets up the GPIO pin
    Buzzer(int pin);

    // Destructor: Cleans up the pin when the object is destroyed
    ~Buzzer();

    void turnOn();
    void turnOff();

private:
    int m_pin; //Pin number
    std::optional<gpiod::line_request> m_request;
};

