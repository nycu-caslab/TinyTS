#ifndef _CTX_H_
#define _CTX_H_
#include "gen_lib/include/types.h"
#include <stdint.h>

#define MODEL_NAME "kws_ds_cnn_l_splitted_BF_1"

#ifndef SPLIT_HEIGHT
#define SPLIT_HEIGHT 1
#endif

extern int8_t arena[70640];
extern const int32_t arena_size;
extern const int32_t input_tid;
extern const int32_t output_tid;
extern const Tensor tensors[52];
extern const int all_0_zp_cursor;
extern const int only_zp_cursor;
extern const int only_zp_start;
extern const int32_t quant_scale[9];
extern const int8_t quant_zeropoint[295];
extern const int32_t split_offset[204];
extern const uint8_t offline_tensor_data[420048];
int CtxSummary();
#endif