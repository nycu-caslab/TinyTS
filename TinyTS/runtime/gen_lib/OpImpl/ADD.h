#ifndef _ADD_H_
#define _ADD_H_
#include "gen_model/include/op_param.h"
#include "gen_lib/include/ctx_util.h"
#include "gen_lib/include/quantization_util.h"
#include "gen_lib/include/kernel_util.h"
#include "gen_lib/include/common.h"
#include <algorithm>

// #include "gen_lib/include/arm_nnsupportfunctions.h"
#include "cmsis/CMSIS/NN/Include/arm_nnsupportfunctions.h"
#include "cmsis/CMSIS/NN/Include/arm_nnfunctions.h"

#define SAT_INPUT(__INPUT, __MULT, __SHIFT)                 \
  __INPUT = arm_nn_sat_doubling_high_mult(__INPUT, __MULT); \
  __INPUT = arm_nn_divide_by_power_of_two(__INPUT, -__SHIFT);
// #define MAX(A,B) ((A) > (B) ? (A) : (B))
// #define MIN(A,B) ((A) < (B) ? (A) : (B))

namespace OP_utils{
namespace add{

typedef struct OpData{
  bool requires_broadcast;

  // These fields are used in both the general 8-bit -> 8bit quantized path,
  // and the special 16-bit -> 16bit quantized path
  int input1_shift;
  int input2_shift;
  int32 output_activation_min;
  int32 output_activation_max;

  // These fields are used only in the general 8-bit -> 8bit quantized path
  int32 input1_multiplier;
  int32 input2_multiplier;
  int32 output_multiplier;
  int output_shift;
  int left_shift;
  int32 input1_offset;
  int32 input2_offset;
  int32 output_offset;
}OpData;

bool HaveSameShapes (const DIM_TYPE dim1[4], const DIM_TYPE dim2[4]){
    for(int i = 0; i < 4; i++){
        if (dim1[i] != dim2[i]) return false;
    }
    return true;
}

inline void CalculateOpData( const Tensor* input1, const Tensor* input2, 
                             const Tensor* output, const SharedParam_Add *params,
                             OpData* data) {
    data->requires_broadcast = !HaveSameShapes(input1->dims, input2->dims);

    // 8bit -> 8bit general quantized path, with general rescalings
    data->input1_offset = -GetTensorQuantZP(input1)[0];
    data->input2_offset = -GetTensorQuantZP(input2)[0];
    data->output_offset = GetTensorQuantZP(output)[0];
    data->left_shift = 20;
    const double twice_max_input_scale =
        2 * static_cast<double>(
                std::max(GetTensorQuantScale(input1)[0], GetTensorQuantScale(input2)[0]));
    const double real_input1_multiplier =
        static_cast<double>(GetTensorQuantScale(input1)[0]) / twice_max_input_scale;
    const double real_input2_multiplier =
        static_cast<double>(GetTensorQuantScale(input2)[0]) / twice_max_input_scale;
    const double real_output_multiplier =
        twice_max_input_scale /
        ((1 << data->left_shift) * static_cast<double>(GetTensorQuantScale(output)[0]));

    QuantizeMultiplierSmallerThanOneExp(
        real_input1_multiplier, &data->input1_multiplier, &data->input1_shift);

    QuantizeMultiplierSmallerThanOneExp(
        real_input2_multiplier, &data->input2_multiplier, &data->input2_shift);

    QuantizeMultiplierSmallerThanOneExp(
        real_output_multiplier, &data->output_multiplier, &data->output_shift);

    CalculateActivationRangeQuantized(
        params->fused_ActFunc, output, &data->output_activation_min,
        &data->output_activation_max);
}

} // namespace ADD
} // namespace OP_utils

/**
 * @brief           Saturating doubling high multiply. Result matches
 *                  NEON instruction VQRDMULH.
 * @param[in]       m1        Multiplicand
 * @param[in]       m2        Multiplier
 * @return          Result of multiplication.
 *
 */

void add(int shared_param_idx){
    const SharedParam_Add *param = &shared_param_add[shared_param_idx];
    const int8_t *input_A = GetTensorData(param->input_A);
    const int8_t *input_B = GetTensorData(param->input_B);
    int8_t *output = GetTensorData(param->output);

    OP_utils::add::OpData data;
    #if !OPT_OFFLOAD_ENABLE
    OP_utils::add::CalculateOpData( &tensors[param->input_A],
                                    &tensors[param->input_B],
                                    &tensors[param->output],
                                    param, &data);
    #else
    const int32_t *op_data_buffer = GetOpData(param->op_data_offset);
    data.output_activation_min = op_data_buffer[0];
    data.output_activation_max = op_data_buffer[1];
    data.input1_shift = op_data_buffer[2];
    data.input2_shift = op_data_buffer[3];
    data.output_shift = op_data_buffer[4];
    data.input1_multiplier = op_data_buffer[5];
    data.input2_multiplier = op_data_buffer[6];
    data.output_multiplier = op_data_buffer[7];
    data.input1_offset = op_data_buffer[8];
    data.input2_offset = op_data_buffer[9];
    data.output_offset = op_data_buffer[10];
    data.left_shift = 20;
    data.requires_broadcast = !OP_utils::add::HaveSameShapes(
                                        tensors[param->input_A].dims,
                                        tensors[param->input_B].dims);
    #endif

    if (data.requires_broadcast){
        DEBUG_ERR_MSG("broadcast add haven't been implemented yet.");
    }

    arm_elementwise_add_s8(input_A, input_B,
                            data.input1_offset, data.input1_multiplier,
                            data.input1_shift, data.input2_offset,
                            data.input2_multiplier, data.input2_shift,
                            data.left_shift, output,
                            data.output_offset, data.output_multiplier,
                            data.output_shift, data.output_activation_min,
                            data.output_activation_max,
                            GetTensorSize(param->output));
    
    // print_split(param->output, sid);
}

void add(int shared_param_idx, int sid){
    if (sid == -1) {
        // unsplitted tensor
        add(shared_param_idx);
        return;
    }
    const SharedParam_Add *param = &shared_param_add[shared_param_idx];
    const int8_t *input_A = GetSplitData(param->input_A, sid);
    const int8_t *input_B = GetSplitData(param->input_B, sid);
    int8_t *output = GetSplitData(param->output, sid);

    OP_utils::add::OpData data;
    #if !OPT_OFFLOAD_ENABLE
    // OP_utils::add::CalculateOpData( &tensors[param->input_A],
    //                                 &tensors[param->input_B],
    //                                 &tensors[param->output],
    //                                 param, &data);
    #else
    const int32_t *op_data_buffer = GetOpData(param->op_data_offset);
    data.output_activation_min = op_data_buffer[0];
    data.output_activation_max = op_data_buffer[1];
    data.input1_shift = op_data_buffer[2];
    data.input2_shift = op_data_buffer[3];
    data.output_shift = op_data_buffer[4];
    data.input1_multiplier = op_data_buffer[5];
    data.input2_multiplier = op_data_buffer[6];
    data.output_multiplier = op_data_buffer[7];
    data.input1_offset = op_data_buffer[8];
    data.input2_offset = op_data_buffer[9];
    data.output_offset = op_data_buffer[10];
    data.left_shift = 20;
    data.requires_broadcast = !OP_utils::add::HaveSameShapes(
                                        tensors[param->input_A].dims,
                                        tensors[param->input_B].dims);
    #endif

    if (data.requires_broadcast){
        DEBUG_ERR_MSG("broadcast add haven't been implemented yet.");
    }

    // serial test
    // float input1_scale = GetTensorQuantScale(&tensors[param->input_A])[0];
    // float input2_scale = GetTensorQuantScale(&tensors[param->input_B])[0];
    // float output_scale = GetTensorQuantScale(&tensors[param->output])[0];
    // float zero_y = GetTensorQuantZP(&tensors[param->output])[0];
    // float input1_scale = quant_scale[0];
    // float input2_scale = quant_scale[0];
    // float output_scale = quant_scale[0];
    // int32_t zero_y = quant_zeropoint[0];

    // for (int i = 0; i < GetSplitSize(param->output, sid); ++i) {
    //     float input1_fp = ((float)*input_A++ + data.input1_offset) * input1_scale;
    //     float input2_fp = ((float)*input_B++ + data.input2_offset) * input2_scale;
    //     int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
    //     clamped_output = (clamped_output<-128)? -128:clamped_output;
    //     clamped_output = (clamped_output>127)? 127:clamped_output;
    //     output[i] = (int8_t)(clamped_output);
    // }

    // SIMD
    arm_elementwise_add_s8(input_A, input_B,
                            data.input1_offset, data.input1_multiplier,
                            data.input1_shift, data.input2_offset,
                            data.input2_multiplier, data.input2_shift,
                            data.left_shift, output,
                            data.output_offset, data.output_multiplier,
                            data.output_shift, data.output_activation_min,
                            data.output_activation_max,
                            GetSplitSize(param->output, sid));
    
    // print_split(param->input_A, sid);
}

#undef SAT_INPUT
// #undef MAX
// #undef MIN

#endif