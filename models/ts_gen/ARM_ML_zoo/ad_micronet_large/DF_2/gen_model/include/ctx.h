#ifndef _CTX_H_
#define _CTX_H_
#include "gen_lib/include/types.h"
#include <stdint.h>

#define MODEL_NAME "ad_micronet_large_splitted_DF_2"

#ifndef SPLIT_HEIGHT
#define SPLIT_HEIGHT 2
#endif

extern int8_t arena[125312];
extern const int32_t arena_size;
extern const int32_t input_tid;
extern const int32_t output_tid;
extern const Tensor tensors[37];
extern const int all_0_zp_cursor;
extern const int only_zp_cursor;
extern const int only_zp_start;
extern const int32_t quant_scale[13];
extern const int8_t quant_zeropoint[293];
extern const int32_t split_offset[92];
extern const uint8_t offline_tensor_data[361680];
int CtxSummary();
#endif
