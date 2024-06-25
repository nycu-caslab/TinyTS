#ifndef _RESHAPE_H_
#define _RESHAPE_H_
#include "gen_model/include/op_param.h"
#include "gen_lib/include/ctx_util.h"
void reshape(int shared_param_idx){
    const SharedParam_Reshape* param = &shared_param_reshape[shared_param_idx];
    int8_t *input = GetTensorData(param->input);
    int8_t *output = GetTensorData(param->output);
    int tensor_size = GetTensorSize(param->input);

    for(int i = 0; i < tensor_size; ++i){
        output[i] = input[i];
    }
}
#endif