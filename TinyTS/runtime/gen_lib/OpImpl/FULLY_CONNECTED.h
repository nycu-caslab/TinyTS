#ifndef _FULLY_CONNECTED_H_
#define _FULLY_CONNECTED_H_

#include <algorithm>

#include "gen_model/include/op_param.h"
#include "gen_lib/include/ctx_util.h"
#include "gen_lib/include/kernel_util.h"
#include "gen_lib/include/quantization_util.h"
#include "gen_lib/include/kernel_common.h"

#include "cmsis/CMSIS/NN/Include/arm_nnfunctions.h"

namespace OP_utils{
namespace FC{
struct OpData {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // The index of the temporary tensor where the quantized inputs are cached.
  int input_quantized_index;
};

TfLiteStatus CalculateOpDataFC(/* TfLiteContext* context, */
                             const SharedParam_FC* params,
                             /* TfLiteType data_type, */ const Tensor* input,
                             const Tensor* filter,
                             const Tensor* bias, const Tensor* output,
                             OpData* data) {
  TfLiteStatus status = kTfLiteOk;
  // if (data_type != kTfLiteFloat32) {
    double real_multiplier = 0.0;
    GetQuantizedConvolutionMultipler(
        input, filter, bias, output, &real_multiplier);
    QuantizeMultiplier(real_multiplier, &data->output_multiplier, &data->output_shift);
    CalculateActivationRangeQuantized(
        params->fused_ActFunc, output, &data->output_activation_min,
        &data->output_activation_max);
  // }
  return status;
}
} // namespace FC
} // namespace OP_utils

inline void fully_connected_serial(const SharedParam_FC* params, OP_utils::FC::OpData* data, 
                                  const Tensor* input, const Tensor* filter, const Tensor* output){
  const DIM_TYPE* output_shape = output->dims;
  const int batches = output_shape[0];
  const int output_depth = output_shape[1];
  const DIM_TYPE* filter_shape = filter->dims;
  const int filter_dim_count = 2 /* filter_shape.DimensionsCount() */;
  const int accum_depth = filter_shape[filter_dim_count - 1];

  // Get data
  const int8* input_data = GetTensorData(params->input);
  const int8* filter_data = GetOfflineTensorData(params->weight);
  const int32* bias_data= reinterpret_cast<const int32*>(GetOfflineTensorData(params->bias));
  int8* output_data = GetTensorData(params->output);

  // integer ops: fully connected
  const int32 input_offset = -*GetTensorQuantZP(params->input);
  const int32 filter_offset = -*GetTensorQuantZP(params->weight);
  const int32 output_offset = *GetTensorQuantZP(params->output);
  const int32 output_multiplier = data->output_multiplier;
  const int output_shift = -data->output_shift;
  const int32 output_activation_min = data->output_activation_min;
  const int32 output_activation_max = data->output_activation_max;
  /* TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2); */

//   TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
//   const int filter_dim_count = filter_shape.DimensionsCount();
//   const int batches = output_shape[0];
//   const int output_depth = output_shape[1];
  /* TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2)); */
//   const int accum_depth = filter_shape[filter_dim_count - 1];
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32 input_val = input_data[b * accum_depth + d];
        int32 filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
    }
  }
}

void fully_connected_simd(const SharedParam_FC* params, OP_utils::FC::OpData* data, 
                          const Tensor* input, const Tensor* filter, const Tensor* output){
  const DIM_TYPE* output_shape = output->dims;
  const DIM_TYPE* filter_shape = filter->dims;
  const int batches = output_shape[0];
  const int accum_depth = filter_shape[1];
  const int output_depth = output_shape[1];

  // Get data
  const int8* input_data = GetTensorData(params->input);
  const int8* filter_data = GetOfflineTensorData(params->weight);
  const int32* bias_data= reinterpret_cast<const int32*>(GetOfflineTensorData(params->bias));
  int8* output_data = GetTensorData(params->output);
  
  cmsis_nn_per_tensor_quant_params quant_params;
  cmsis_nn_dims input_dims;
  cmsis_nn_dims filter_dims;
  cmsis_nn_dims bias_dims;
  cmsis_nn_dims output_dims;
  cmsis_nn_context ctx;

  quant_params.multiplier = data->output_multiplier;
  quant_params.shift = data->output_shift;

  input_dims.n = batches;
  input_dims.h = 1;
  input_dims.w = 1;
  input_dims.c = accum_depth;

  filter_dims.n = accum_depth;
  filter_dims.h = 1;
  filter_dims.w = 1;
  filter_dims.c = output_depth;

  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = output_depth;

  output_dims.n = batches;
  output_dims.h = 1;
  output_dims.w = 1;
  output_dims.c = output_depth;

  int input_offset = -(*(GetTensorQuantZP(params->input)));
  int output_offset = (*(GetTensorQuantZP(params->output)));
  int activation_min = data->output_activation_min;
  int activation_max = data->output_activation_max;



  // aussme otuput dimension count == 2
  // if (output_dim_count > 2 && accum_depth % 4 == 0) {
  //   cmsis_nn_conv_params conv_params;
  //   conv_params.dilation.h = 1;
  //   conv_params.dilation.w = 1;
  //   conv_params.input_offset = input_offset;
  //   conv_params.output_offset = output_offset;
  //   conv_params.stride.h = 1;
  //   conv_params.stride.w = 1;
  //   conv_params.padding.h = 0;
  //   conv_params.padding.w = 0;
  //   conv_params.activation.min = activation_min;
  //   conv_params.activation.max = activation_max;

  //   cmsis_nn_per_channel_quant_params per_channel_quant_params;
  //   per_channel_quant_params.multiplier =
  //       const_cast<int32_t*>(data.per_channel_output_multiplier);
  //   per_channel_quant_params.shift =
  //       const_cast<int32_t*>(data.per_channel_output_shift);

  //   for (int i = 0; i < output_depth; i++) {
  //     per_channel_quant_params.multiplier[i] = quant_params.multiplier;
  //     per_channel_quant_params.shift[i] = quant_params.shift;
  //   }

  //   arm_convolve_1x1_s8_fast(
  //       &ctx, &conv_params, &per_channel_quant_params, &input_dims,
  //       input_data, &filter_dims,
  //       filter_data, &bias_dims, bias_data,
  //       &output_dims, output_data);
  // } else {
    cmsis_nn_fc_params fc_params;
    fc_params.input_offset = input_offset;
    fc_params.output_offset = output_offset;
    fc_params.filter_offset = 0;
    fc_params.activation.min = data->output_activation_min;
    fc_params.activation.max = data->output_activation_max;
// printf(
// "fc_params:\n"
// "\tfc_params.input_offset: %d\n"
// "\tfc_params.output_offset: %d\n"
// "\tfc_params.filter_offset: %d\n"
// "\tfc_params.activation.min: %d\n"
// "\tfc_params.activation.max: %d\n"
// "quant_params:\n"
// "\tquant_params.multiplier: %d\n"
// "\tquant_params.shift: %d\n"
// "input_params:\n"
// "\tinput_dims.n: %d\n"
// "\tinput_dims.h: %d\n"
// "\tinput_dims.w: %d\n"
// "\tinput_dims.c: %d\n"
// "filter_params:\n"
// "\tfilter_dims.n: %d\n"
// "\tfilter_dims.h: %d\n"
// "\tfilter_dims.w: %d\n"
// "\tfilter_dims.c: %d\n"
// "bias_params:\n"
// "\tbias_dims.n: %d\n"
// "\tbias_dims.h: %d\n"
// "\tbias_dims.w: %d\n"
// "\tbias_dims.c: %d\n"
// "output_params:\n"
// "\toutput_dims.n: %d\n"
// "\toutput_dims.h: %d\n"
// "\toutput_dims.w: %d\n"
// "\toutput_dims.c: %d\n"
// "\tctx.buf == nullptr: %d\n",
// fc_params.input_offset,
// fc_params.output_offset,
// fc_params.filter_offset,
// fc_params.activation.min,
// fc_params.activation.max,
// quant_params.multiplier,
// quant_params.shift,
// input_dims.n,
// input_dims.h,
// input_dims.w,
// input_dims.c,
// filter_dims.n,
// filter_dims.h,
// filter_dims.w,
// filter_dims.c,
// bias_dims.n,
// bias_dims.h,
// bias_dims.w,
// bias_dims.c,
// output_dims.n,
// output_dims.h,
// output_dims.w,
// output_dims.c,
// ctx.buf == nullptr
// );
    arm_fully_connected_s8(
        &ctx, &fc_params, &quant_params, &input_dims,
        input_data, &filter_dims,
        filter_data, &bias_dims, bias_data,
        &output_dims, output_data);
  // }

}

void fully_connected(int shared_param_idx) {
  const SharedParam_FC* params = &shared_param_fc[shared_param_idx];

  const Tensor& input = tensors[params->input];
  const Tensor& filter = tensors[params->weight];
  const Tensor& bias = tensors[params->bias];
  const Tensor& output= tensors[params->output];

  /* TfLiteType data_type = input->type; */
  OP_utils::FC::OpData local_data_object;
  OP_utils::FC::OpData* data = &local_data_object;
  /* TF_LITE_ENSURE_STATUS( */OP_utils::FC::CalculateOpDataFC(/* context, */ params/* , data_type */, &input,
                                        &filter, &bias, &output, data);
  
#if SIMD
    fully_connected_simd(params, data, &input, &filter, &output);
#else
    fully_connected_serial(params, data, &input, &filter, &output);
#endif
}
#endif