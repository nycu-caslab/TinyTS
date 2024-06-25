#ifndef _CTX_H_
#define _CTX_H_
#include "gen_lib/include/types.h"
#include <stdint.h>

#define MODEL_NAME "2_MW320_splitted_DF_1"

#ifndef SPLIT_HEIGHT
#define SPLIT_HEIGHT 1
#endif
#define EVICT_IN

extern int8_t arena[105216];
extern const int32_t arena_size;
extern const int32_t input_tid;
extern const int32_t output_tid;
extern int8_t *model_input_data;
extern const Tensor tensors[233];
extern const int all_0_zp_cursor;
extern const int only_zp_cursor;
extern const int only_zp_start;
extern const int32_t quant_scale[7];
extern const int8_t quant_zeropoint[563];
extern const int32_t split_offset[1597];
extern const uint8_t offline_tensor_data[657712];
int CtxSummary();
#endif
