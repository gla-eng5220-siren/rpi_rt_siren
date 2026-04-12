#pragma once

#include <memory>
#include <functional>
#include <string>

#include "detection_result.hpp"

namespace rpi_rt {

/** \addtogroup Alarming
 *  @{
 */

  /**
   * Configuration struct for alarming via Brevo Email API.
   */
  struct brevo_config_t {
    //! Host for webhook invocation
    std::string api_host = "api.brevo.com";
    //! Path for webhook invocation
    std::string api_path = "/v3/smtp/email";

    //! The API key to be set in headers
    std::string api_key;
    //! Name of the email sender
    std::string sender_name = "FlameIris Webhook";
    //! The email sender; must be registered in brevo
    std::string sender_email;
    //! The destination email address
    std::string to_email;
    //! Name of the email receiver
    std::string to_name = "FlameIris User";
  };

  /**
   * The base class for all alarm types. Trigger actions if flames detected.
   *
   * Adhere to the Liskov Substitution Principle (LSP) and Interface Segregation
   * Principle (ISP) in SOLID.
   */
  class alarm_t {
    public:
      virtual ~alarm_t() {}

      /**
       * Start the alarm process.
       *
       * Blocks until close, so should run in a thread.
       */
      virtual void run() = 0;

      /**
       * Reports a detection result.
       *
       * Wakes up the run function to perform the actual alarming. Subclasses
       * might have custom throttle implemented.
       *
       * @param result The outcome reported by sensor and detection logic.
       */
      virtual void report(std::unique_ptr<detection_result_t> result) = 0;

      /**
       * Stop the run function and return.
       */
      virtual void close() = 0;
  };

  /**
   * The factory method for creating a alarm_t reporting to stdout.
   *
   * Adhere to the Interface Segregation Principle (ISP) in SOLID.
   */
  std::shared_ptr<alarm_t> create_stdout_alarm();

  /**
   * The factory method for creating a alarm_t reporting via Brevo
   * transactional email service.
   *
   * Adhere to the Interface Segregation Principle (ISP) in SOLID.
   *
   * @param cfg The configuration for brevo API.
   */
  std::shared_ptr<alarm_t> create_brevo_email_alarm(brevo_config_t cfg);

/** @}*/

}

