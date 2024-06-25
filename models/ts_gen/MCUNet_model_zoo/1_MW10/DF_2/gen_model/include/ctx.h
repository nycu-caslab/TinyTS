#ifndef _CTX_H_
#define _CTX_H_
#include "gen_lib/include/types.h"
#include <stdint.h>

#define MODEL_NAME "1_MW10_splitted_DF_2"

#ifndef SPLIT_HEIGHT
#define SPLIT_HEIGHT 2
#endif

extern int8_t arena[36736];
extern const int32_t arena_size;
extern const int32_t input_tid;
extern const int32_t output_tid;
extern const Tensor tensors[150];
extern const int all_0_zp_cursor;
extern const int only_zp_cursor;
extern const int only_zp_start;
extern const int32_t quant_scale[7];
extern const int8_t quant_zeropoint[531];
extern const int32_t split_offset[241];
extern const uint8_t offline_tensor_data[379376];
int CtxSummary();
#endif
