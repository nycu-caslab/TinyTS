#ifndef GEN_INCLUDE_QUANTIZATION_UTIL_H
#define GEN_INCLUDE_QUANTIZATION_UTIL_H

#include <cmath>
#include <cstdint>
#include <limits>

#include "gen_lib/include/compatibility.h"
#include "gen_lib/include/types.h"

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of its exponent.
//
// Handles an arbitrary positive multiplier. The 'shift' output-value is
// basically the 'floating-point exponent' of the multiplier:
// Negative for a right-shift (when the multiplier is <1), positive for a
// left-shift (when the multiplier is >1)
void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier,
                        int* shift);

// Splits a double input value into a returned fraction, and a shift value from
// the exponent, using only bitwise and integer operations to support
// microcontrollers and other environments without floating-point support.
//
// This is designed to be a replacement for how std::frexp() is used within the
// QuantizeMultiplier() function, and so has a different signature than the
// standard version, returning a 64-bit integer rather than a double. This
// result has a maximum value of 1<<31, with the fraction expressed as a
// proportion of that maximum.
//
// std::frexp() returns NaNs and infinities unmodified, but since we're
// returning integers that can't represent those values, instead we return
// a shift of std::numeric_limits<int>::max() for all bad numbers, with an int64
// result of 0 for NaNs, std:numeric_limits<int64_t>::max() for +INFINITY, and
// std::numeric_limits<int64_t>::min() for -INFINITY. Denormalized inputs will
// result in return values that end up truncating some bits at the end,
// reflecting the loss of precision inherent in denormalization.
int64_t IntegerFrExp(double input, int* shift);

void QuantizeMultiplierSmallerThanOneExp(double double_multiplier,
                                         int32_t* quantized_multiplier,
                                         int* left_shift);

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of its exponent.
//
// Restricted to the case where the multiplier > 1.
void QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift);

// This first creates a multiplier in a double equivalent of
// Q(input_integer_bits).(31-input_integer_bits) representation, with extra
// precision in the double's fractional bits.  It then splits the result into
// significand and exponent.
void PreprocessSoftmaxScaling(double beta, double input_scale,
                              int input_integer_bits,
                              int32_t* quantized_multiplier, int* left_shift);

// Calculate the largest input that will result in a within-bounds intermediate
// result within MultiplyByQuantizedMultiplierGreaterThanOne.  In other words,
// it must not overflow before we reduce the value by multiplication by the
// input multiplier.  The negative radius is used as the minimum difference in
// Softmax.
int CalculateInputRadius(int input_integer_bits, int input_left_shift,
                         int total_signed_bits = 31);

#endif