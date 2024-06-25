#ifndef _OP_PARAM_H_
#define _OP_PARAM_H_
#include "gen_lib/include/types.h"

#ifndef SPLIT_HEIGHT
#define SPLIT_HEIGHT 2
#endif
extern const SharedParam_Add shared_param_add[10];
extern const SharedParam_AvgPool shared_param_avgpool[1];
extern const SharedParam_Concat shared_param_concat[1];
extern const SharedParam_Conv shared_param_conv[37];
extern const SharedParam_Depthwise_Conv shared_param_dwconv[17];
extern const SharedParam_Pad shared_param_pad[8];
extern const SharedParam_Reshape shared_param_reshape[1];
extern const SharedParam_Split shared_param_split[1];
extern const int32_t op_data[16848];
int OpParamSummary();
#endif
