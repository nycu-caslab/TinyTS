#ifndef _ARM_MATH_DEPRECATED_H
#define _ARM_MATH_DEPRECATED_H

#include "cmsis_compiler.h"
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>

/**
  @brief         Write 2 S16 to S16 pointer and increment pointer afterwards.
  @param[in]     pS16      points to input value
  @param[in]     value     S32 value
  @return        none
 */
__STATIC_FORCEINLINE void write_s16x2_ia (
  int16_t ** pS16,
  int32_t    value)
{
  int32_t val = value;
#ifdef __ARM_FEATURE_UNALIGNED
  memcpy (*pS16, &val, 4);
#else
  (*pS16)[0] = (val & 0x0FFFF);
  (*pS16)[1] = (val >> 16) & 0x0FFFF;
#endif

 *pS16 += 2;
}

/**
  @brief         Write 2 S16 to S16 pointer.
  @param[in]     pS16      points to input value
  @param[in]     value     S32 value
  @return        none
 */
__STATIC_FORCEINLINE void write_s16x2 (
  int16_t * pS16,
  int32_t   value)
{
  int32_t val = value;

#ifdef __ARM_FEATURE_UNALIGNED
  memcpy (pS16, &val, 4);
#else
  pS16[0] = val & 0x0FFFF;
  pS16[1] = val >> 16;
#endif
}

#endif /* _ARM_MATH_DEPRECATED_H */

/**
 *
 * End of file.
 */