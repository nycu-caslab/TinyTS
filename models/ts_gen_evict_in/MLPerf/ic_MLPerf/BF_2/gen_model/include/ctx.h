#ifndef _CTX_H_
#define _CTX_H_
#include "gen_lib/include/types.h"
#include <stdint.h>

#define MODEL_NAME "ic_MLPerf_splitted_BF_2"

#ifndef SPLIT_HEIGHT
#define SPLIT_HEIGHT 2
#endif
#define EVICT_IN

extern int8_t arena[35392];
extern const int32_t arena_size;
extern const int32_t input_tid;
extern const int32_t output_tid;
extern int8_t *model_input_data;
extern const Tensor tensors[44];
extern const int all_0_zp_cursor;
extern const int only_zp_cursor;
extern const int only_zp_start;
extern const int32_t quant_scale[8];
extern const int8_t quant_zeropoint[83];
extern const int32_t split_offset[128];
extern const uint8_t offline_tensor_data[78832];
int CtxSummary();
#endif
