// @allow-raw-throw
#include <c10/xpu/XPUException.h>

#include <c10/util/Exception.h>

#include <string>

namespace c10::xpu {

void c10_xpu_check_implementation(
    const char* error_msg,
    const int32_t error_code,
    const char* filename,
    const char* function_name,
    const uint32_t line_number) {
  std::string check_message;
#ifndef STRIP_ERROR_MESSAGES
  check_message.append("XPU error: ");
  if (error_msg && error_msg[0] != '\0') {
    check_message.append(error_msg);
  } else {
    check_message.append("unknown error (error code: ");
    check_message.append(std::to_string(error_code));
    check_message.append(")");
  }
  check_message.append("\n");
  check_message.append(
      "XPU kernel errors might be asynchronously reported at some other "
      "API call, so the stacktrace below might be incorrect.\n"
      "For debugging consider passing "
      "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 and ZE_DEBUG=1 to "
      "get more information.");
#endif
  throw c10::AcceleratorError(
      {function_name, filename, line_number}, error_code, check_message);
}

} // namespace c10::xpu
