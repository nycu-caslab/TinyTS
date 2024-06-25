#ifndef _CTX_UTIL_H_
#define _CTX_UTIL_H_
#include "gen_model/include/ctx.h"
#include "gen_model/include/op_param.h"

int8_t* GetTensorData(int tid);
const int8_t* GetOfflineTensorData(int tid);
int8_t* GetSplitData(int tid, int sid);
int GetTensorSize(int tid);
int GetSplitSize(int tid, int sid);
int GetSplitNum(int tid);
const int32_t* GetTensorQuantMin(int tid);
const int32_t* GetTensorQuantMax(int tid);
const float*   GetTensorQuantScale(int tid);
const int8_t* GetTensorQuantZP(int tid);
const int32_t* GetTensorQuantMin(const Tensor *t);
const int32_t* GetTensorQuantMax(const Tensor *t);
const float*   GetTensorQuantScale(const Tensor *t);
const int8_t* GetTensorQuantZP(const Tensor *t);
const int32_t* GetOpData(int offset);
uint8_t* GetOpScratchBuf(int out_tid, int out_sid);
void print_tensor(int tid);
void print_all_split(int tid);
void print_all_split_v(int tid);
void print_all_split_as_normal_tensor(int tid);
void print_split(int tid, int sid);
void fill_input_tensor(const uint8_t *input_data);
void fill_input_tensor(const int8_t *input_data);
void fill_input_tensor_w_val(int8_t val);
bool isEvicted(int tid, int sid);
void CodeGenSummary();

#endif