#include "gen_lib/include/ctx_util.h"
#include <cstdio>

int8_t* GetTensorData(int tid){
    #ifdef DEBUG
    if (tensors[tid].data_offset > 0){
        // offline tensor
        
        Should use GetOfflineTensorData() instead.
    }else if (tensors[tid].data_offset == -1){
    #endif
        // online tensor
        // return arena;
        return &arena[tensors[tid].split_offset];
    #ifdef DEBUG
    }else{
        DEBUG_ERR_MSG("Invalid tensor type.");
    }
    #endif
}

const int8_t* GetOfflineTensorData(int tid){
    #ifdef DEBUG
    if (tensors[tid].data_offset > 0){
    #endif
        // return arena;
        return reinterpret_cast<const int8_t*>(&offline_tensor_data[tensors[tid].data_offset]);
    #ifdef DEBUG
    }else{
        DEBUG_ERR_MSG("Invalid tensor type.");
    }
    #endif
}

int8_t* GetSplitData(int tid, int sid){
    #ifdef DEBUG
    if(sid >= -tensors[tid].data_offset)
        DEBUG_ERR_MSG("sid out of range.");
    #endif
    #ifdef EVICT_IN
    if (split_offset[tensors[tid].split_offset+sid] == -1){
        int32_t offset = GetSplitSize(tid, sid) * sid;
        return &model_input_data[offset];
    }else
    #endif
        return &arena[split_offset[tensors[tid].split_offset+sid]];
}

int GetTensorSize(int tid){
    const DIM_TYPE *dim = tensors[tid].dims;
    int size = 1;
    for (int i = 0; i<4; i++){
        size*=dim[i];
    }
    return size;
}

int GetSplitSize(int tid, int sid){
    const DIM_TYPE *dim = tensors[tid].dims;
    int size = 1;
    for (int i = 2; i<4; i++){
        size*=dim[i];
    }
    #if SPLIT_HEIGHT == 1
        return size;
    #elif SPLIT_HEIGHT == 2
        int split_cnt = -tensors[tid].data_offset;
        if(__builtin_expect(sid < split_cnt - 1, 1)){
            return SPLIT_HEIGHT*size;
        }
        else if(sid == split_cnt - 1 ){
            if(dim[1]&1) return size;
            else return SPLIT_HEIGHT*size;
        }else if(sid > split_cnt - 1){
            printf("Error: sid out of range.\n");
            while(1){}
        }
    #elif (SPLIT_HEIGHT != 0) && ((SPLIT_HEIGHT & (SPLIT_HEIGHT-1)) == 0)
        int split_cnt = -tensors[tid].data_offset;
        if(__builtin_expect(sid < split_cnt - 1, 1)){
            return SPLIT_HEIGHT*size;
        }
        else if(sid == split_cnt - 1){
            if(dim[1]&(SPLIT_HEIGHT-1)) return (dim[1]&(SPLIT_HEIGHT-1))*size;
            else return SPLIT_HEIGHT*size;
        }else if(sid > split_cnt - 1){
            printf("Error: sid out of range.\n");
            while(1){}
        }
    #elif SPLIT_HEIGHT > 2
        int split_cnt = -tensors[tid].data_offset;
        if(__builtin_expect(sid < split_cnt - 1, 1)){
            return SPLIT_HEIGHT*size;
        }
        else if(sid == split_cnt - 1){
            if(dim[1]%SPLIT_HEIGHT) return (dim[1]%SPLIT_HEIGHT)*size;
            else return SPLIT_HEIGHT*size;
        }else if(sid > split_cnt - 1){
            printf("Error: sid out of range.\n");
            while(1){}
        }
    #else
        printf("Error: Unknown split height.\n");
        while(1){}
    #endif
    return size;
}

int GetSplitNum(int tid){
    return -tensors[tid].data_offset;
}

// const int32_t* GetTensorQuantMin(int tid){
//     return quant_min + tensors[tid].quant_offset;
// }

// const int32_t* GetTensorQuantMax(int tid){
//     return quant_max + tensors[tid].quant_offset;
// }

const float* GetTensorQuantScale(int tid){
    return reinterpret_cast<const float*>(quant_scale + tensors[tid].quant_offset);
}

const int8_t* GetTensorQuantZP(int tid){
    const Tensor *t = &tensors[tid];
    if (t->quant_offset >= only_zp_cursor)
        return quant_zeropoint + only_zp_start + t->quant_offset - only_zp_cursor;
    else if (t->quant_offset >= all_0_zp_cursor)
        return quant_zeropoint + all_0_zp_cursor;
    else
        return quant_zeropoint + t->quant_offset;
}

// const int32_t* GetTensorQuantMin(const Tensor *t){
//     return quant_min + t->quant_offset;
// }

// const int32_t* GetTensorQuantMax(const Tensor *t){
//     return quant_max + t->quant_offset;
// }

const float* GetTensorQuantScale(const Tensor *t){
    return reinterpret_cast<const float*>(quant_scale + t->quant_offset);
}

const int8_t* GetTensorQuantZP(const Tensor *t){
    if (t->quant_offset >= only_zp_cursor)
        return quant_zeropoint + only_zp_start + t->quant_offset - only_zp_cursor;
    else if (t->quant_offset >= all_0_zp_cursor)
        return quant_zeropoint + all_0_zp_cursor;
    else
        return quant_zeropoint + t->quant_offset;
}

const int32_t* GetOpData(int offset){
    return op_data + offset;
}

uint8_t* GetOpScratchBuf(int out_tid, int out_sid){
    return nullptr;
}

void print_tensor(int tid){
    int8_t *tensor_data = GetTensorData(tid);
    int tensor_size = GetTensorSize(tid);
    printf("Tensor %2d\n\t", tid);
    for (int i = 0; i < tensor_size; i++) {
        printf("%4d ",tensor_data[i]);
        if(i%16==15 && !(i == tensor_size-1)) 
            printf("\n\t");
    }
    printf("\nEnd of Tensor %2d\n", tid);
}

void print_all_split(int tid){
    int split_num = GetSplitNum(tid);
    printf("Tensor %2d:\n\t", tid);
    int cnt = 0;
    for (int sid = 0; sid < split_num; sid++){
        int8_t *split_data = GetSplitData(tid, sid);
        int split_size = GetSplitSize(tid, sid);
        for (int i = 0; i < split_size; i++, cnt++) {
            printf("%4d ",split_data[i]);
            if(cnt%16==15 && !(i == split_size-1 && sid == split_num-1))
                printf("\n\t");
        }
    }
    printf("\nEnd of Tensor %2d\n", tid);
}

void print_all_split_v(int tid){
    int split_num = GetSplitNum(tid);
    printf("Tensor %2d:\n\t", tid);
    int cnt = 0;
    for (int sid = 0; sid < split_num; sid++){
        print_split(tid, sid);
    }
    printf("\nEnd of Tensor %2d\n", tid);
}

void print_all_split_as_normal_tensor(int tid){
    int split_num = GetSplitNum(tid);
    int tensor_size = GetTensorSize(tid);
    printf("Tensor %2d:\n\t", tid);
    int cnt = 0;
    for (int sid = 0; sid < split_num; sid++){
        int8_t *split_data = GetSplitData(tid, sid);
        int split_size = GetSplitSize(tid, sid);
        for (int i = 0; i < split_size; i++, cnt++) {
            printf("%4d ",split_data[i]);
            if(cnt%16==15 && !(cnt == tensor_size-1))
                printf("\n\t");
        }
    }
    printf("\nEnd of Tensor %2d\n", tid);
}

void print_split(int tid, int sid){
    int8_t *split_data = GetSplitData(tid, sid);
    int split_size = GetSplitSize(tid, sid);
    printf("Tensor %2d, Split %2d:\n\t", tid, sid);
    for (int i = 0; i < split_size; i++) {
        printf("%4d ",split_data[i]);
        if(i%16==15 ^ i == split_size-1)
            printf("\n\t");
    }
    printf("\nEnd of Tensor %2d, Split %2d\n", tid, sid);
}

void fill_input_tensor(const uint8_t *input_data){
    int8_t *input_tensor = GetTensorData(input_tid);
    int input_size = GetTensorSize(input_tid);
    for (int i = 0; i < input_size; i++) {
        if(input_data[i]<=127)
            input_tensor[i] = ((int8_t)input_data[i]) - 128;
        else
            input_tensor[i] = (int8_t)(input_data[i] - 128);
    }
}
void fill_input_tensor(const int8_t *input_data){
    int8_t *input_tensor = GetTensorData(input_tid);
    int input_size = GetTensorSize(input_tid);
    for (int i = 0; i < input_size; i++) {
        input_tensor[i] = input_data[i];
    }
}

void fill_input_tensor_w_val(int8_t val){
    int8_t *input_tensor = GetTensorData(input_tid);
    int input_size = GetTensorSize(input_tid);
    for (int i = 0; i < input_size; i++) {
        input_tensor[i] = val;
    }
}

bool isEvicted(int tid, int sid){
    if(sid >= 0 && split_offset[tensors[tid].split_offset+sid] == -1){
        printf("%d-%d Evicted\n", tid, sid);
        return true;
    }
    printf("%d-%d Not evicted\n", tid, sid);
    return false;
}

void CodeGenSummary(){
    int byte_tensor = CtxSummary();
    int byte_op_param = OpParamSummary();
    printf("Total: %d\n", byte_tensor + byte_op_param);
}