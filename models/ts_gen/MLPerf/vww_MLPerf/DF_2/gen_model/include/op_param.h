#ifndef _OP_PARAM_H_
#define _OP_PARAM_H_
#include "gen_lib/include/types.h"

#ifndef SPLIT_HEIGHT
#define SPLIT_HEIGHT 2
#endif
extern const SharedParam_AvgPool shared_param_avgpool[1];
extern const SharedParam_Concat shared_param_concat[1];
extern const SharedParam_Conv shared_param_conv[14];
extern const SharedParam_Depthwise_Conv shared_param_dwconv[13];
extern const SharedParam_FC shared_param_fc[1];
extern const SharedParam_Reshape shared_param_reshape[1];
extern const SharedParam_Softmax shared_param_softmax[1];
extern const SharedParam_Split shared_param_split[1];
extern const int32_t op_data[8624];
int OpParamSummary();
#endif
