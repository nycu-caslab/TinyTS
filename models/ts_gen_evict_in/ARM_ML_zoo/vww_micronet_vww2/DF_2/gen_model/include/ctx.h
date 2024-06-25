#ifndef _CTX_H_
#define _CTX_H_
#include "gen_lib/include/types.h"
#include <stdint.h>

#define MODEL_NAME "vww_micronet_vww2_splitted_DF_2"

#ifndef SPLIT_HEIGHT
#define SPLIT_HEIGHT 2
#endif
#define EVICT_IN

extern int8_t arena[21728];
extern const int32_t arena_size;
extern const int32_t input_tid;
extern const int32_t output_tid;
extern int8_t *model_input_data;
extern const Tensor tensors[190];
extern const int all_0_zp_cursor;
extern const int only_zp_cursor;
extern const int only_zp_start;
extern const int32_t quant_scale[7];
extern const int8_t quant_zeropoint[225];
extern const int32_t split_offset[266];
extern const uint8_t offline_tensor_data[99952];
int CtxSummary();
#endif
