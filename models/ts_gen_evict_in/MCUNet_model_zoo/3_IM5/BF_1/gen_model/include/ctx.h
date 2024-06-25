#ifndef _CTX_H_
#define _CTX_H_
#include "gen_lib/include/types.h"
#include <stdint.h>

#define MODEL_NAME "3_IM5_splitted_BF_1"

#ifndef SPLIT_HEIGHT
#define SPLIT_HEIGHT 1
#endif
#define EVICT_IN

extern int8_t arena[74656];
extern const int32_t arena_size;
extern const int32_t input_tid;
extern const int32_t output_tid;
extern int8_t *model_input_data;
extern const Tensor tensors[153];
extern const int all_0_zp_cursor;
extern const int only_zp_cursor;
extern const int only_zp_start;
extern const int32_t quant_scale[1005];
extern const int8_t quant_zeropoint[1054];
extern const int32_t split_offset[687];
extern const uint8_t offline_tensor_data[651456];
int CtxSummary();
#endif
