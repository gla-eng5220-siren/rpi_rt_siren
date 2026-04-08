#include "buzzer.h"
#include <gpiod.hpp>

Buzzer::Buzzer(int pin) : m_pin(pin) {
    // 1. Get the GPIO chip (the Pi's brain) 
    auto chip = gpiod::chip("/dev/gpiochip0"); 
    // 2. Request the specific line (the pin) as an OUTPUT 
    m_request = chip.prepare_request()
        .set_consumer("Buzzer") 
        .add_line_settings(
            m_pin, 
            gpiod::line_settings()
                .set_direction(gpiod::line::direction::OUTPUT)
                .set_output_value(gpiod::line::value::INACTIVE)
        )
        .do_request();
    }
    
void Buzzer::turnOn() {
    // send 3.3V to the buzzer
    m_request->set_value(m_pin, gpiod::line::value::ACTIVE);
}

void Buzzer::turnOff() {
    // drop pin to 0V (silent)
    m_request->set_value(m_pin, gpiod::line::value::INACTIVE);
}

Buzzer::~Buzzer() {
    // Safety check: Make sure the buzzer is quiet before we exit
    turnOff();
}
