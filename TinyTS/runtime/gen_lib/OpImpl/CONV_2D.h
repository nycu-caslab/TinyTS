#ifndef _CONV_2D_H_
#define _CONV_2D_H_
#include "gen_model/include/op_param.h"
#include "gen_lib/include/ctx_util.h"

#include "gen_lib/include/types.h"
#include "gen_lib/include/padding.h"
#include "gen_lib/include/kernel_common.h"
#include "gen_lib/include/kernel_util.h"
#include "gen_lib/include/quantization_util.h"
#include "gen_lib/include/virtualfp.h"

#include "cmsis/CMSIS/NN/Include/arm_nn_types.h"
#include "cmsis/CMSIS/NN/Include/arm_nnfunctions.h"

#include <algorithm>



constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kMaxChannels = 1024;

// Conv is quantized along dimension 0:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kConvQuantizedDimension = 0;

/* TODO */
struct OpData {
  PaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  // TODO(b/141139247): Allocate these dynamically when possible.

  #if !OPT_OFFLOAD_ENABLE
    int32_t per_channel_output_multiplier[kMaxChannels];
    int32_t per_channel_output_shift[kMaxChannels];
  #else
    const int32_t *per_channel_output_multiplier;
    const int32_t *per_channel_output_shift;
    const int32_t *contribs;
  #endif

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
};

/* TODO */
TfLiteStatus CalculateOpData(/* TfLiteContext* context, TfLiteNode* node, */
                             const SharedParam_Conv* params, int width, int height,
                             int filter_width, int filter_height, int out_width,
                             int out_height, /* const TensorDataType data_type, */
                             OpData* data) {

//   bool has_bias = node->inputs->size == 3;
//   bool custom_padding = node->inputs->size >= 4;
//   bool multiple_input_feature_maps = node->inputs->size > 4;
  // Check number of inputs/outputs
//   TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2 || custom_padding || multiple_input_feature_maps);
//   TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // original
  // bool has_bias = node->inputs->size == 3;
  // // Check number of inputs/outputs
  // TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  // TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      params->dilation_height_factor, params->dilation_width_factor, height,
      width, filter_height, filter_width, padding, &out_height, &out_width);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  /* if (data_type != kFLOAT32) { */
    const Tensor& input = tensors[params->input];
    const Tensor& filter = tensors[params->filter];
    const Tensor& bias = tensors[params->bias];
    Tensor output = tensors[params->output];
    int num_channels = filter.dims[0];

    // TF_LITE_ENSURE_STATUS(PopulateConvolutionQuantizationParams(
    //     /* context, */ &input, &filter, &bias, &output, params->activation,
    //     &data->output_multiplier, &data->output_shift,
    //     &data->output_activation_min, &data->output_activation_max,
    //     data->per_channel_output_multiplier,
    //     reinterpret_cast<int*>(data->per_channel_output_shift), num_channels));
  /* } */
  return kTfLiteOk;
};

void conv_2d_serial(const SharedParam_Conv* params, OpData* data, const VirtualFp<int8_t>& combined_ifmap, const int8* filter_data, const int32* bias_data, int8* output_data, int output_height)
{
  // Set input_shape to the ifmap which is going to go through conv2d calculation
  const int* input_shape = combined_ifmap.combined_dim;
  const DIM_TYPE* filter_shape = tensors[params->filter].dims;
  const DIM_TYPE* output_shape = tensors[params->output].dims;

  // Get parameters.
  const int32 input_offset = -(*GetTensorQuantZP(params->input));// r = s(q - Z)
  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int dilation_width_factor = params->dilation_width_factor;
  const int dilation_height_factor = params->dilation_height_factor;
  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int32 output_offset = *GetTensorQuantZP(params->output);
  const int32* output_multiplier = data->per_channel_output_multiplier;
  const int32* output_shift = data->per_channel_output_shift;

  // Set min and max value of the output.
  const int32 output_activation_min = data->output_activation_min;
  const int32 output_activation_max = data->output_activation_max;

  // Sanity check.
  const int batches = output_shape[0];  // MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape[3];  // MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = std::min(filter_shape[0], output_shape[3]); // MatchingDim(filter_shape, 0, output_shape, 3);

  int input_width = input_shape[2];
  int input_height = input_shape[1];  // split_id_end - split_id_begin + 1
  int filter_width = filter_shape[2];
  int filter_height = filter_shape[1];
  int output_width = output_shape[2];
  // int output_height = GetSplitSize(params->output, output_split_id)/(output_width*output_shape[3]);

  // printf("Min,Max: (%d, %d)\n", output_activation_min, output_activation_max);
  // for (int i = 0; i < output_depth; i++) {
  //   printf("(%d,%d) ", output_multiplier[i], output_shift[i]);
  //   if(i%8==7) printf("\n");
  // }printf("\n");

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          int32 acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);
                    // (in_y < combined_height); //DEBUG 2022/06/06
                if (is_point_inside_image) {
                  // int32 input_val = input_data[(out_y*output_width + out_x) * output_depth + out_channel];
                  int32 input_val = combined_ifmap.inputs_data[in_y][in_x*input_depth + in_channel];
                  int32 filter_val =
                      filter_data[((out_channel*filter_height+filter_y)*filter_width + filter_x) * input_depth + in_channel];
                  // Accumulate with 32 bits accumulator.
                  // In the nudging process during model quantization, we force
                  // real value of 0.0 be represented by a quantized value. This
                  // guarantees that the input_offset is a int8, even though it
                  // is represented using int32.
                  // int32 += int8 * (int8 - int8) so the highest value we can
                  // get from each accumulation is [-127, 127] * ([-128, 127] -
                  // [-128, 127]), which is [-32512, 32512]. log2(32512)
                  // = 14.98, which means we can accumulate at least 2^16
                  // multiplications without overflow. The accumulator is
                  // applied to a filter so the accumulation logic will hold as
                  // long as the filter size (filter_y * filter_x * in_channel)
                  // does not exceed 2^16, which is the case in all the models
                  // we have seen so far.
                  // TODO(jianlijianli): Add a check to make sure the
                  // accumulator depth is smaller than 2^16.
                  acc += filter_val * (input_val + input_offset);
                }
              }
            }
          }

          if (bias_data) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[(out_y*output_width + out_x) * output_depth + out_channel] =
              static_cast<int8_t>(acc);
        }
      }
    }
  }
  // if(shared_param_idx==4){
  //   print_split(params->output, output_split_id);
  // }
}

void conv_2d_simd(const SharedParam_Conv* params, const OpData* data, int scratch_buffer_offset,
                        const VirtualFp<int8_t>& combined_ifmap, const int8* filter_data, const int32* bias_data, int8* output_data, int output_height)
{
  /* Prepare */

  const int* input_shape = combined_ifmap.combined_dim;
  const DIM_TYPE* output_shape = tensors[params->output].dims;
  const DIM_TYPE* filter_shape = tensors[params->filter].dims;

  // Sanity check.
  // TFLITE_DCHECK_LE(conv_params.activation.min, conv_params.activation.max);
  // TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  // TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  // TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batch_size = output_shape[0]/* MatchingDim(input_shape, 0, output_shape, 0) */;
  const int input_depth = input_shape[3];/* MatchingDim(input_shape, 3, filter_shape, 3) */;
  const int output_depth = output_shape[3]/* MatchingDim(filter_shape, 0, output_shape, 3) */;
  // if (GetTensorData<int8_t>(bias)) {
  //   TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  // }


  // Initialize cmsis-nn input dimensions
  cmsis_nn_dims input_dims;
  input_dims.n = batch_size;
  input_dims.h = input_shape[1]/* input->dims->data[1]*(is_custom?node->inputs->size - 4 : 1) */;
  input_dims.w = input_shape[2]/* input->dims->data[2] */;
  input_dims.c = input_shape[3];

  // Initialize cmsis-nn filter dimensions
  cmsis_nn_dims filter_dims;
  filter_dims.n = output_depth;
  filter_dims.h = filter_shape[1];
  filter_dims.w = filter_shape[2];
  filter_dims.c = input_depth;

  // Bias
  cmsis_nn_dims bias_dims;
  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = output_depth;

  // Initialize cmsis-nn output dimensions
  cmsis_nn_dims output_dims;
  output_dims.n = batch_size;
  output_dims.h = output_height /* output_shape[1] */;
  output_dims.w = output_shape[2];
  output_dims.c = output_depth;


  // int32_t buf_size = 0;
  cmsis_nn_conv_params conv_params;
  // conv_params.input_offset = -input->params.zero_point;
  conv_params.input_offset = -(*GetTensorQuantZP(params->input));
  // conv_params.output_offset = output->params.zero_point;
  conv_params.output_offset = *GetTensorQuantZP(params->output);
  conv_params.stride.h = params->stride_height;
  conv_params.stride.w = params->stride_width;
  conv_params.dilation.h = params->dilation_height_factor;
  conv_params.dilation.w = params->dilation_width_factor;
  conv_params.padding.h = data->padding.height;
  conv_params.padding.w = data->padding.width;
  conv_params.activation.min = data->output_activation_min;
  conv_params.activation.max = data->output_activation_max;

  // buf_size = arm_convolve_wrapper_s8_get_buffer_size(
  //     &conv_params, &input_dims, &filter_dims, &output_dims);

  // node->user_data = buffer_idx;
  // if (buf_size > 0) {
  //   TF_LITE_ENSURE_STATUS(
  //       context->RequestScratchBufferInArena(context, buf_size, buffer_idx));
  // } else {
  //   *buffer_idx = -1;
  // }

  /* Prepare end */

  // Initialize cmsis-nn per channel quantization parameters
  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = const_cast<int32_t*>(data->per_channel_output_multiplier);
  quant_params.shift = const_cast<int32_t*>(data->per_channel_output_shift);



  // Initialize cmsis-nn context
  cmsis_nn_context ctx;
  ctx.buf = reinterpret_cast<uint8_t*>(arena+scratch_buffer_offset);
  ctx.size = 0;

  // auto* buffer_idx = reinterpret_cast<int*>(node->user_data);
  // if (*buffer_idx > -1) {
  //   ctx.buf = context->GetScratchBuffer(context, *buffer_idx);
  //   // Note: ctx.size is currently not used in cmsis-nn.
  //   // The buffer should be allocated in the Prepare function through
  //   // arm_convolve_wrapper_s8_get_buffer_size
  // }

  arm_cmsis_nn_status status = arm_convolve_wrapper_s8(
    &ctx, &conv_params, &quant_params, &input_dims,
    combined_ifmap.inputs_data, &filter_dims, filter_data,
    &bias_dims, reinterpret_cast<const int32*>(bias_data), &output_dims,
    output_data, data->contribs);
}

void conv_2d(int shared_param_idx, int scratch_buffer_offset)
{
  #if HIDE_CONV
    return;
  #endif
  const SharedParam_Conv* params = &shared_param_conv[shared_param_idx];
  const Tensor& input = tensors[params->input];
  const Tensor& filter = tensors[params->filter];
  const Tensor& output= tensors[params->output];

  // Get shape
  const DIM_TYPE* input_shape = input.dims;
  const DIM_TYPE* filter_shape = filter.dims;
  const DIM_TYPE* output_shape = output.dims;

  int input_width = input_shape[2];
  int input_height = input_shape[1];
  int filter_width = filter_shape[2];
  int filter_height = filter_shape[1];
  int output_width = output_shape[2];
  int output_height = output_shape[1];

  OpData data;

  #if !OPT_OFFLOAD_ENABLE
  // v0.1 method: call CalculateOpData
  // Calculate padding and quantization params
  CalculateOpData(params, input_width, input_height, filter_width,
                  filter_height, output_width, output_shape[1], &data);
  #else
  // v0.2 method: use offloaded OpData
  const int32_t *op_data_buffer = GetOpData(params->op_data_offset);
  data.padding.height = op_data_buffer[0];
  data.padding.width = op_data_buffer[1];
  data.output_activation_min = op_data_buffer[2];
  data.output_activation_max = op_data_buffer[3];
  // Method  I: directly point to data in flash
  data.per_channel_output_multiplier = const_cast<int32_t*>(&op_data_buffer[4]);
  data.per_channel_output_shift = const_cast<int32_t*>(&op_data_buffer[4+output_shape[3]]);
  data.contribs = const_cast<int32_t*>(&op_data_buffer[4+output_shape[3]*2]);
  // // Method II: copy to SRAM
  // for (int i = 0; i < output_depth; i++)
  //   data.per_channel_output_multiplier[i] = op_data_buffer[4+i];
  // for (int i = 0; i < output_depth; i++)
  //   data.per_channel_output_shift[i] = op_data_buffer[4+output_depth+i];
  #endif

  // Sanity check.
  /* TODO: check impl correctness */
  const int batches = std::min(input_shape[0], output_shape[0]);  // MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape[3];  // MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = output_shape[3]; // MatchingDim(filter_shape, 0, output_shape, 3);

#if SIMD
   // Get data
  const int8* filter_data = GetOfflineTensorData(params->filter);
  const int32* bias_data= reinterpret_cast<const int32*>(GetOfflineTensorData(params->bias));
  int8* output_data = GetTensorData(params->output);
  const VirtualFp<int8_t> combined_ifmap(params->input, input);

  cmsis_nn_conv_params conv_params;
  // conv_params.input_offset = -input->params.zero_point;
  conv_params.input_offset = -(*GetTensorQuantZP(params->input));
  // conv_params.output_offset = output->params.zero_point;
  conv_params.output_offset = *GetTensorQuantZP(params->output);
  conv_params.stride.h = params->stride_height;
  conv_params.stride.w = params->stride_width;
  conv_params.dilation.h = params->dilation_height_factor;
  conv_params.dilation.w = params->dilation_width_factor;
  conv_params.padding.h = data.padding.height;
  conv_params.padding.w = data.padding.width;
  conv_params.activation.min = data.output_activation_min;
  conv_params.activation.max = data.output_activation_max;

  // Initialize cmsis-nn per channel quantization parameters
  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = const_cast<int32_t*>(data.per_channel_output_multiplier);
  quant_params.shift = const_cast<int32_t*>(data.per_channel_output_shift);

  // Initialize cmsis-nn dimensions
  // Input
  cmsis_nn_dims input_dims;
  input_dims.n = batches;
  input_dims.h = input_shape[1];
  input_dims.w = input_shape[2];
  input_dims.c = input_depth;

  // Filter
  cmsis_nn_dims filter_dims;
  filter_dims.n = output_depth;
  filter_dims.h = filter_shape[1];
  filter_dims.w = filter_shape[2];
  filter_dims.c = input_depth;

  // Bias
  cmsis_nn_dims bias_dims;
  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = output_depth;

  // Output
  cmsis_nn_dims output_dims;
  output_dims.n = batches;
  output_dims.h = output_shape[1];
  output_dims.w = output_shape[2];
  output_dims.c = output_depth;

  // Initialize cmsis-nn context
  cmsis_nn_context ctx;
  ctx.buf = reinterpret_cast<uint8_t*>(arena+scratch_buffer_offset/* GetOpScratchBuf(scratch_buffer_offset) */);
  ctx.size = 0;

  // printf("activation: %d\n", params->activation);
  // printf("conv_params.input_offset: %d\n", conv_params.input_offset);
  // printf("conv_params.output_offset: %d\n", conv_params.output_offset);
  // printf("conv_params.stride.h: %d\n", conv_params.stride.h);
  // printf("conv_params.stride.w: %d\n", conv_params.stride.w);
  // printf("conv_params.dilation.h: %d\n", conv_params.dilation.h);
  // printf("conv_params.dilation.w: %d\n", conv_params.dilation.w);
  // printf("conv_params.padding.h: %d\n", conv_params.padding.h);
  // printf("conv_params.padding.w: %d\n", conv_params.padding.w);
  // printf("conv_params.padding.w: %d\n", conv_params.padding.w);
  // printf("conv_params.activation.min: %d\n", conv_params.activation.min);
  // printf("conv_params.activation.max: %d\n", conv_params.activation.max);
  // //quant_params 
  // printf("quant_params.multiplier: %d\n", quant_params.multiplier);
  // printf("quant_params.shift: %d\n", quant_params.shift);
  // // input_dims filter_dims bias_dims output_dims
  // printf("input_dims: %d %d %d %d\n", input_dims.n, input_dims.h, input_dims.w, input_dims.c);
  // printf("filter_dims: %d %d %d %d\n", filter_dims.n, filter_dims.h, filter_dims.w, filter_dims.c);
  // printf("bias_dims: %d %d %d %d\n", bias_dims.n, bias_dims.h, bias_dims.w, bias_dims.c);
  // printf("output_dims: %d %d %d %d\n", output_dims.n, output_dims.h, output_dims.w, output_dims.c);

  // arm_convolve_wrapper_s8 dispatches the optimized kernel accordingly with
  // the parameters passed
  arm_cmsis_nn_status status = arm_convolve_wrapper_s8_ori(
      &ctx, &conv_params, &quant_params, &input_dims,
      combined_ifmap.inputs_data[0], &filter_dims, filter_data,
      &bias_dims, bias_data, &output_dims,
      output_data);
#else
  const int32 input_offset = -*GetTensorQuantZP(params->input)/* input->params.zero_point */;  // r = s(q - Z)
  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int dilation_width_factor = params->dilation_width_factor;
  const int dilation_height_factor = params->dilation_height_factor;
  const int pad_width = data.padding.width;
  const int pad_height = data.padding.height;
  const int32 output_offset = *GetTensorQuantZP(params->output);
  const int32* output_multiplier = data.per_channel_output_multiplier;
  const int32* output_shift = data.per_channel_output_shift;

  // Set min and max value of the output.
  const int32 output_activation_min = data.output_activation_min;
  const int32 output_activation_max = data.output_activation_max;

  // Get data
  const int8* input_data = GetTensorData(params->input);
  const int8* filter_data = GetOfflineTensorData(params->filter);
  const int32* bias_data= reinterpret_cast<const int32*>(GetOfflineTensorData(params->bias));
  int8* output_data = GetTensorData(params->output);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          int32 acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);
                if (is_point_inside_image) {
                  int32 input_val = input_data[((batch * input_height + in_y) * input_width + in_x) * input_shape[3] + in_channel];
                  int32 filter_val =
                      filter_data[((out_channel*filter_height+filter_y)*filter_width + filter_x) * input_depth + in_channel];
                  // Accumulate with 32 bits accumulator.
                  // In the nudging process during model quantization, we force
                  // real value of 0.0 be represented by a quantized value. This
                  // guarantees that the input_offset is a int8, even though it
                  // is represented using int32.
                  // int32 += int8 * (int8 - int8) so the highest value we can
                  // get from each accumulation is [-127, 127] * ([-128, 127] -
                  // [-128, 127]), which is [-32512, 32512]. log2(32512)
                  // = 14.98, which means we can accumulate at least 2^16
                  // multiplications without overflow. The accumulator is
                  // applied to a filter so the accumulation logic will hold as
                  // long as the filter size (filter_y * filter_x * in_channel)
                  // does not exceed 2^16, which is the case in all the models
                  // we have seen so far.
                  // TODO(jianlijianli): Add a check to make sure the
                  // accumulator depth is smaller than 2^16.
                  acc += filter_val * (input_val + input_offset);
                }
              }
            }
          }

          if (bias_data) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[(out_y*output_width + out_x) * output_depth + out_channel] =
              static_cast<int8_t>(acc);
        }
      }
    }
  }
#endif
// print_tensor(params->input);
// print_tensor(params->output);
}

void conv_2d(int shared_param_idx, int output_split_id, int scratch_buffer_offset)
{
  #if HIDE_CONV
    return;
  #endif
  if (output_split_id == -1) {
    // unsplitted tensor
    conv_2d(shared_param_idx, scratch_buffer_offset);
    return;
  }
  const SharedParam_Conv* params = &shared_param_conv[shared_param_idx];
  const Tensor& input = tensors[params->input];
  const Tensor& filter = tensors[params->filter];
  const Tensor& output= tensors[params->output];

  // Get shape
  const DIM_TYPE* whole_input_shape = input.dims;
  const DIM_TYPE* filter_shape = filter.dims;
  const DIM_TYPE* output_shape = output.dims;

  int input_width = whole_input_shape[2];
  int input_height = whole_input_shape[1];
  int filter_width = filter_shape[2];
  int filter_height = filter_shape[1];
  int output_width = output_shape[2];
  int output_height = (output_split_id>=(output_shape[1]/SPLIT_HEIGHT))? (output_shape[1]%SPLIT_HEIGHT):(SPLIT_HEIGHT);
  OpData data;

  #if !OPT_OFFLOAD_ENABLE
  // v0.1 method: call CalculateOpData
  // Calculate padding and quantization params
  CalculateOpData(params, input_width, input_height, filter_width,
                  filter_height, output_width, output_shape[1], &data);
  #else
  // v0.2 method: use offloaded OpData
  const int32_t *op_data_buffer = GetOpData(params->op_data_offset);
  data.padding.height = op_data_buffer[0];
  data.padding.width = op_data_buffer[1];
  data.output_activation_min = op_data_buffer[2];
  data.output_activation_max = op_data_buffer[3];
  // Method  I: directly point to data in flash
  data.per_channel_output_multiplier = /* const_cast<int32_t*> */(&op_data_buffer[4]);
  data.per_channel_output_shift = /* const_cast<int32_t*> */(&op_data_buffer[4+output_shape[3]]);
  data.contribs = /* const_cast<int32_t*> */(&op_data_buffer[4+output_shape[3]*2]);
  // printf("HERE: %d %d %d\n", data.per_channel_output_multiplier, data.per_channel_output_shift, data.contribs);
  // // Method II: copy to SRAM
  // for (int i = 0; i < output_depth; i++)
  //   data.per_channel_output_multiplier[i] = op_data_buffer[4+i];
  // for (int i = 0; i < output_depth; i++)
  //   data.per_channel_output_shift[i] = op_data_buffer[4+output_depth+i];
  #endif

  // Get data

  // calculate the receptive field base on output_split_id
  // calculate the needed input split range
  int input_row_begin = (output_split_id*SPLIT_HEIGHT) * params->stride_height - data.padding.height;
  int input_row_end = input_row_begin + (filter_height-1) + params->stride_height*(SPLIT_HEIGHT-1);
  input_row_end = input_row_end < input_height? input_row_end:input_height-1;
  data.padding.height = input_row_begin >= 0 ? 0:abs(input_row_begin);
  input_row_begin = input_row_begin > 0 ? input_row_begin: 0;
  const VirtualFp<int8_t> combined_ifmap(params->input, input, input_row_begin, input_row_end, input_width*whole_input_shape[3]);
  int combined_height = input_row_end-input_row_begin+1;

  // Set input_shape to the ifmap which is going to go through conv2d calculation
  const int* input_shape = combined_ifmap.combined_dim;

  const int8* filter_data = GetOfflineTensorData(params->filter);
  const int32* bias_data = reinterpret_cast<const int32*>(GetOfflineTensorData(params->bias));
  int8* output_data = GetSplitData(params->output, output_split_id);


#if defined(SIMD)
    conv_2d_simd(params, &data, scratch_buffer_offset, combined_ifmap, filter_data, bias_data, output_data, output_height);
#else
    conv_2d_serial(params, &data, combined_ifmap, filter_data, bias_data, output_data, output_height);
#endif
  // if (shared_param_idx == 1)
  //   print_split(params->output, output_split_id);
}

extern "C" {
#include "tiny_conv1x1.h"
#include "tiny_conv_3x3_ich3_st2_padw1_flexiblepadh.h"
}

#include "tiny_conv1x1_header.h"
#include "tiny_conv_3x3_ich3_st2_padw1_flexiblepadh_header.h"


#endif