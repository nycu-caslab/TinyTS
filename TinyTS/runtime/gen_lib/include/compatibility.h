#ifndef GEN_INCLUDE_COMPATIBILITY_H_
#define GEN_INCLUDE_COMPATIBILITY_H_

#include <cstdint>

#include "gen_lib/tensorflow/op_macro.h"

#ifndef TFLITE_DCHECK
#define TFLITE_DCHECK(condition) (condition) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_EQ
#define TFLITE_DCHECK_EQ(x, y) ((x) == (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_NE
#define TFLITE_DCHECK_NE(x, y) ((x) != (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_GE
#define TFLITE_DCHECK_GE(x, y) ((x) >= (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_GT
#define TFLITE_DCHECK_GT(x, y) ((x) > (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_LE
#define TFLITE_DCHECK_LE(x, y) ((x) <= (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_LT
#define TFLITE_DCHECK_LT(x, y) ((x) < (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

// TODO(ahentz): Clean up: We should stick to the DCHECK versions.
#ifndef TFLITE_CHECK
#define TFLITE_CHECK(condition) (condition) ? (void)0 : TFLITE_ABORT
#endif

#ifndef TFLITE_CHECK_EQ
#define TFLITE_CHECK_EQ(x, y) ((x) == (y)) ? (void)0 : TFLITE_ABORT
#endif

#ifndef TFLITE_CHECK_NE
#define TFLITE_CHECK_NE(x, y) ((x) != (y)) ? (void)0 : TFLITE_ABORT
#endif

#ifndef TFLITE_CHECK_GE
#define TFLITE_CHECK_GE(x, y) ((x) >= (y)) ? (void)0 : TFLITE_ABORT
#endif

#ifndef TFLITE_CHECK_GT
#define TFLITE_CHECK_GT(x, y) ((x) > (y)) ? (void)0 : TFLITE_ABORT
#endif

#ifndef TFLITE_CHECK_LE
#define TFLITE_CHECK_LE(x, y) ((x) <= (y)) ? (void)0 : TFLITE_ABORT
#endif

#ifndef TFLITE_CHECK_LT
#define TFLITE_CHECK_LT(x, y) ((x) < (y)) ? (void)0 : TFLITE_ABORT
#endif

using int8 = std::int8_t;
using uint8 = std::uint8_t;
using int16 = std::int16_t;
using uint16 = std::uint16_t;
using int32 = std::int32_t;
using uint32 = std::uint32_t;

#endif