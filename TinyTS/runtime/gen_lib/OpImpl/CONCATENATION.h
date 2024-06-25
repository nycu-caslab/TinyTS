#ifndef _CONCATENATION_H_
#define _CONCATENATION_H_
#include "gen_model/include/op_param.h"
#include "gen_lib/include/ctx_util.h"
void concatenation(int shared_param_idx){
    const SharedParam_Concat* param = &shared_param_concat[shared_param_idx];
    if (param->axis != 1){
        printf("Concat on dim other than H is currently not supported.\n");
        while(1){}
    }
    #ifdef DEBUG
    if (-tensors[param->input].data_offset != tensors[param->output].dim.s.H){
        DEBUG_ERR_MSG("Invalid number of splits.");
    }
    #endif
    int8_t *dest = GetTensorData(param->output);
    #ifdef DEBUG
    int tensor_size = GetTensorSize(param->output);
    #endif
    int current = 0;
    int split_num = -tensors[param->input].data_offset;
    for (int sid = 0; sid < split_num; sid++){
        int8_t *src = GetSplitData(param->input, sid);
        int split_size = GetSplitSize(param->input, sid);
        #ifdef DEBUG
        for (int i=0; i<split_size && current<tensor_size; i++)
        #else
        for (int i=0; i<split_size; i++)
        #endif
        {
            dest[current] = src[i];
            current++;
        }
    }
    #ifdef DEBUG
    DEBUG_ERR_MSG("Invalid number of splits.");
    #endif
}
#endif