#ifndef _SPLIT_H_
#define _SPLIT_H_
#include "gen_model/include/op_param.h"
#include "gen_lib/include/ctx_util.h"
void split(int shared_param_idx){
    const SharedParam_Split* param = &shared_param_split[shared_param_idx];
    #ifdef DEBUG
    if (-tensors[param->output].data_offset != param->num_splits){
        DEBUG_ERR_MSG("Number of splits does not match the height of destination tensor.");
    }
    #endif
    int8_t *input = GetTensorData(param->input);
    int current = 0;
    #ifdef DEBUG
    int tensor_size = GetTensorSize(param->input);
    #endif
    #ifdef EVICT_IN
    printf("test evict\n");
    if (isEvicted(param->output, 0))
        return;
    #endif
    for (int sid = 0; sid < param->num_splits; sid++){
        int8_t *dest = GetSplitData(param->output, sid);
        int split_size = GetSplitSize(param->output, sid);
        #ifdef DEBUG
        for (int i=0; i<split_size && current<tensor_size; i++)
        #else
        for (int i=0; i<split_size; i++)
        #endif
        {
            dest[i] = input[current++];
        }
    }
}

#endif