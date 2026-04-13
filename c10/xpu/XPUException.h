// @allow-raw-throw
#pragma once

#include <c10/util/Exception.h>
#include <c10/xpu/XPUMacros.h>
#include <sycl/sycl.hpp>

namespace c10::xpu {

static inline sycl::async_handler asyncHandler =
    [](const sycl::exception_list& el) {
      if (el.size() == 0) {
        return;
      }
      for (const auto& e : el) {
        try {
          std::rethrow_exception(e);
        } catch (sycl::exception& e) {
          TORCH_WARN("SYCL Exception: ", e.what());
        }
      }
      throw;
    };

C10_XPU_API void c10_xpu_check_implementation(
    const char* error_msg,
    int32_t error_code,
    const char* filename,
    const char* function_name,
    uint32_t line_number);

#define C10_XPU_CHECK(EXPR)                       \
  do {                                            \
    try {                                         \
      EXPR;                                       \
    } catch (const sycl::exception& e) {          \
      c10::xpu::c10_xpu_check_implementation(     \
          e.what(),                               \
          static_cast<int32_t>(e.code().value()), \
          __FILE__,                               \
          __func__,                               \
          static_cast<uint32_t>(__LINE__));       \
    }                                             \
  } while (0)

#define C10_XPU_CHECK_WARN(EXPR)             \
  do {                                       \
    try {                                    \
      EXPR;                                  \
    } catch (const sycl::exception& e) {     \
      TORCH_WARN("XPU warning: ", e.what()); \
    }                                        \
  } while (0)

} // namespace c10::xpu
