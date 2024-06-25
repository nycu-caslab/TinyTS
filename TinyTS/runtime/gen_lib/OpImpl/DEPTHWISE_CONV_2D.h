#ifndef _DEPTHWISE_CONV_H_
#define _DEPTHWISE_CONV_H_
#include "gen_model/include/op_param.h"
#include "gen_lib/include/ctx_util.h"

#include "gen_lib/include/types.h"
#include "gen_lib/include/padding.h"
#include "gen_lib/include/kernel_common.h"
#include "gen_lib/include/kernel_util.h"
#include "gen_lib/include/quantization_util.h"
#include "gen_lib/include/virtualfp.h"

#include "cmsis/CMSIS/NN/Include/arm_nnfunctions.h"



#include <algorithm>

namespace OP_utils {
namespace DepthwiseConv {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kMaxChannels = 256;

// Depthwise conv is quantized along dimension 3:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kDepthwiseConvQuantizedDimension = 3;

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
  int32_t *per_channel_output_multiplier;
  int32_t *per_channel_output_shift;
  int32_t *contribs;
  #endif
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
};

TfLiteStatus CalculateOpData(/* TfLiteContext* context, TfLiteNode* node, */
                             const SharedParam_Depthwise_Conv* params, int width,
                             int height, int filter_width, int filter_height,
                             /* const TfLiteType data_type ,*/
                             int out_width, int out_height, OpData* data) {

  // bool has_bias = node->inputs->size == 3;
  // bool custom_padding = node->inputs->size >= 4;
  // bool multiple_input_feature_maps = node->inputs->size > 4;
  // Check number of inputs/outputs
  /* TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2 || custom_padding || multiple_input_feature_maps);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1); */

  // int unused_output_height, unused_output_width;
  // data->padding = ComputePaddingHeightWidth(
  //     params->stride_height, params->stride_width, 1, 1, height, width,
  //     filter_height, filter_width, params->padding, &unused_output_height,
  //     &unused_output_width);

  int offset = 0;
  data->padding.height =
      ComputePaddingWithOffset(params->stride_height, params->dilation_height_factor, height,
                               filter_height, out_height, &offset);
  data->padding.height_offset = offset;
  data->padding.width =
      ComputePaddingWithOffset(params->stride_width, params->dilation_width_factor, width,
                               filter_width, out_width, &offset);
  // data->padding.width_offset = offset;
  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  /* if (data_type != kTfLiteFloat32) { */
    const Tensor& input = tensors[params->tid_input];
    const Tensor& filter = tensors[params->tid_filter];
    const Tensor& bias = tensors[params->tid_bias];
    Tensor output = tensors[params->tid_output];
    int num_channels = filter.dims[kDepthwiseConvQuantizedDimension];

    TF_LITE_ENSURE_STATUS(PopulateConvolutionQuantizationParams(
        /* context, */ &input, &filter, &bias, &output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift), num_channels));
  /* } */
  return kTfLiteOk;
}

} // namespace DepthwiseConv
} // namespace OP_utils

void depthwise_conv_serial(const SharedParam_Depthwise_Conv* params, OP_utils::DepthwiseConv::OpData* data, const VirtualFp<int8_t>& combined_ifmap, const int8* filter_data, const int32* bias_data, int8* output_data, int output_height){

  const int* input_shape = combined_ifmap.combined_dim; 
  const DIM_TYPE* filter_shape = tensors[params->tid_filter].dims;
  const DIM_TYPE* output_shape = tensors[params->tid_output].dims;

  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int dilation_width_factor = params->dilation_width_factor;
  const int dilation_height_factor = params->dilation_height_factor;
  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int depth_multiplier = params->depth_multiplier;
  const int32 input_offset = -(*GetTensorQuantZP(params->tid_input));
  // const int32 weight_offset = 0;
  const int32 output_offset = *GetTensorQuantZP(params->tid_output);
  const int32 output_activation_min = std::numeric_limits<int8_t>::min();
  const int32 output_activation_max = std::numeric_limits<int8_t>::max();
  const int32* output_multiplier = data->per_channel_output_multiplier;
  const int32* output_shift = data->per_channel_output_shift;

  const int filter_height = filter_shape[1];
  const int filter_width = filter_shape[2];
  const int input_height = input_shape[1];
  const int input_width = input_shape[2];
  const int input_depth = input_shape[3];

  const int output_width = output_shape[2];
  // Check dimensions of the tensors.
  // TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  // TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  // TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  // TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int batches = output_shape[0];// MatchingDim(input_shape, 0, output_shape, 0);
  // const int output_depth = std::min(filter_shape[3], output_shape[3]); // MatchingDim(filter_shape, 3, output_shape, 3);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          for (int m = 0; m < depth_multiplier; ++m) {
            const int output_channel = m + in_channel * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int32 acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);
                if (is_point_inside_image) {
                  // int32 input_val = inputs_data[Offset(input_shape, batch, in_y,
                  //                                     in_x, in_channel)];
                  int32 input_val =
                      combined_ifmap.inputs_data[in_y][in_x*input_depth + in_channel];
                  // int32 filter_val = filter_data[Offset(
                  //     filter_shape, 0, filter_y, filter_x, output_channel)];
                  int32 filter_val =
                      filter_data[((0 * filter_height + filter_y) * filter_width + filter_x) * filter_shape[3] + output_channel];
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
            if (bias_data) {
              acc += bias_data[output_channel];
            }
            acc = MultiplyByQuantizedMultiplier(
                acc, output_multiplier[output_channel],
                output_shift[output_channel]);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            // output_data[Offset(output_shape, batch, out_y, out_x,
            //                    output_channel)] = static_cast<int8_t>(acc);
            output_data[((batch * output_height + out_y) * output_width + out_x) * output_shape[3] + output_channel] =
                static_cast<int8_t>(acc);
          }
        }
      }
    }
  }
}

void depthwise_conv_simd(const SharedParam_Depthwise_Conv* params, OP_utils::DepthwiseConv::OpData* data, 
                         int scratch_buffer_offset, const VirtualFp<int8_t>& combined_ifmap, 
                         const int8* filter_data, const int32* bias_data, int8* output_data, int output_height){
  
  cmsis_nn_dw_conv_params dw_conv_params;
  dw_conv_params.input_offset   = -(*GetTensorQuantZP(params->tid_input));
  dw_conv_params.input_offset   =  (*GetTensorQuantZP(params->tid_output));
  dw_conv_params.padding.h      = data->padding.height;
  dw_conv_params.padding.w      = data->padding.width;
  dw_conv_params.activation.min = data->output_activation_min;
  dw_conv_params.activation.max = data->output_activation_max;
  dw_conv_params.ch_mult        = params->depth_multiplier;
  dw_conv_params.dilation.h     = params->dilation_height_factor;
  dw_conv_params.dilation.w     = params->dilation_width_factor;
  dw_conv_params.stride.h       = params->stride_height;
  dw_conv_params.stride.w       = params->stride_width;

  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = data->per_channel_output_multiplier;
  quant_params.shift = data->per_channel_output_shift;

  const int* input_shape = combined_ifmap.combined_dim;
  cmsis_nn_dims input_dims;
  input_dims.n = input_shape[0];
  input_dims.h = input_shape[1];
  input_dims.w = input_shape[2];
  input_dims.c = input_shape[3];

  const DIM_TYPE* filter_shape = tensors[params->tid_filter].dims;
  cmsis_nn_dims filter_dims;
  filter_dims.n = input_shape[3];
  filter_dims.h = filter_shape[1];
  filter_dims.w = filter_shape[2];
  filter_dims.c = input_shape[3];

  cmsis_nn_dims bias_dims;
  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = input_shape[3];

  const DIM_TYPE* output_shape = tensors[params->tid_output].dims;
  cmsis_nn_dims output_dims;
  output_dims.n = output_shape[0];
  output_dims.h = output_shape[1];
  output_dims.w = output_shape[2];
  output_dims.c = input_shape[3];
  
  cmsis_nn_context ctx;
  /* 'size' is unused */
  ctx.size = 0;
  if(scratch_buffer_offset >= 0){
    ctx.buf = reinterpret_cast<int16_t*>(arena+scratch_buffer_offset);
  }
  else{
    ctx.buf = nullptr;
  }

  arm_depthwise_conv_wrapper_s8(
      &ctx, &dw_conv_params, &quant_params, &input_dims,
      combined_ifmap.inputs_data, &filter_dims,
      filter_data, &bias_dims,
      bias_data, &output_dims,
      output_data);
  // if (params->depth_multiplier == 1) {
  //   arm_depthwise_conv_s8_opt(
  //       combined_ifmap.inputs_data, input_width, input_height,
  //       input_depth, filter_data, input_depth,
  //       filter_width, filter_height, data->padding.width,
  //       data->padding.height, params->stride_width,
  //       params->stride_height, bias_data,
  //       output_data, data->per_channel_output_shift,
  //       data->per_channel_output_multiplier, output_width, output_height,
  //       *GetTensorQuantZP(params->tid_output)/* op_params.output_offset */,
  //       -(*GetTensorQuantZP(params->tid_input)) /* op_params.input_offset */,
  //       std::numeric_limits<int8_t>::min()/* op_params.quantized_activation_min */,
  //       std::numeric_limits<int8_t>::max()/* op_params.quantized_activation_max */,
  //       params->dilation_width_factor, params->dilation_height_factor, buf);
  // } else {
  //       arm_depthwise_conv_s8(
  //           combined_ifmap.inputs_data, input_width, input_height,
  //           input_depth, filter_data,
  //           params->depth_multiplier * input_depth,
  //           params->depth_multiplier, filter_width, filter_height,
  //           data->padding.width, data->padding.height,
  //           params->stride_width, params->stride_height,
  //           bias_data, output_data,
  //           data->per_channel_output_shift, data->per_channel_output_multiplier,
  //           output_width, output_height, *GetTensorQuantZP(params->tid_output)/* op_params.output_offset */,
  //           -(*GetTensorQuantZP(params->tid_input))/* op_params.input_offset */,
  //           std::numeric_limits<int8_t>::min()/* op_params.quantized_activation_min */,
  //           std::numeric_limits<int8_t>::max()/* op_params.quantized_activation_max */,
  //           params->dilation_width_factor,
  //           params->dilation_height_factor, nullptr);//,
  //       // ARM_MATH_SUCCESS);
  // }

}

void depthwise_conv_2d(int shared_param_idx, int scratch_buffer_offset)
{
  printf("Original dwconv is disabled.");
  while(1){}
//   #if HIDE_DWCONV
//     return;
//   #endif
//   const SharedParam_Depthwise_Conv* params = &shared_param_dwconv[shared_param_idx];

//   const Tensor& output = tensors[params->tid_output];
//   const Tensor& input = tensors[params->tid_input];
//   const Tensor& filter = tensors[params->tid_filter];
//   const Tensor& bias = tensors[params->tid_bias];

//   // Assune data_type is int8
//   // const TfLiteType data_type = input->type;
//   int width = input.dims[2]/* SizeOfDimension(input, 2) */;
//   int height = input.dims[1];
//   int filter_width = filter.dims[2]/* SizeOfDimension(filter, 2) */;
//   int filter_height = filter.dims[1]/* SizeOfDimension(filter, 1) */;

//   OP_utils::DepthwiseConv::OpData data;

//   // All per-channel quantized tensors need valid zero point and scale arrays.
//   /* if (input->type == kTfLiteInt8) {
//     TF_LITE_ENSURE_EQ(context, filter->quantization.type,
//                       kTfLiteAffineQuantization);

//     const auto* affine_quantization =
//         reinterpret_cast<TfLiteAffineQuantization*>(
//             filter->quantization.params);
//     TF_LITE_ENSURE(context, affine_quantization);
//     TF_LITE_ENSURE(context, affine_quantization->scale);
//     TF_LITE_ENSURE(context, affine_quantization->zero_point);
//     TF_LITE_ENSURE(
//         context, affine_quantization->scale->size == 1 ||
//                      affine_quantization->scale->size ==
//                          filter->dims->data[kDepthwiseConvQuantizedDimension]);
//     TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
//                       affine_quantization->zero_point->size);
//   }
//  */
//   const DIM_TYPE* filter_shape = filter.dims;

//   const DIM_TYPE* input_shape = input.dims;
//   const int input_height = input_shape[1];
//   const int input_width = input_shape[2];
//   const int input_depth = input_shape[3];
//   const DIM_TYPE* output_shape = output.dims;
//   const int output_height = output_shape[1];
//   const int output_width = output_shape[2];

//   #if !OPT_OFFLOAD_ENABLE
//   // v0.1 method: call CalculateOpData
//   // Calculate padding and quantization params
//   CalculateOpData(params, input_width, input_height,
//                   filter_width, filter_height,
//                   output_shape[2], output_shape[1], &data);
//   #else
//   // v0.2 method: use offloaded OpData
//   const int32_t *op_data_buffer = GetOpData(params->op_data_offset);
//   data.padding.height = op_data_buffer[0];
//   data.padding.width = op_data_buffer[1];
//   data.output_activation_min = op_data_buffer[2];
//   data.output_activation_max = op_data_buffer[3];
//     // Method  I: directly point to data in flash
//   data.per_channel_output_multiplier = const_cast<int32_t*>(&op_data_buffer[4]);
//   data.per_channel_output_shift = const_cast<int32_t*>(&op_data_buffer[4+output_shape[3]]);
//   // Method II: copy to SRAM
//   // for (int i = 0; i < input.dims[3]; i++)
//   //   data.per_channel_output_multiplier[i] = op_data_buffer[4+i];
//   // for (int i = 0; i < input.dims[3]; i++)
//   //   data.per_channel_output_shift[i] = op_data_buffer[4+input.dims[3]+i];
//   #endif

// #if defined(SIMD)
//   const VirtualFp<int8_t> combined_ifmap(params->tid_input, input);
//   const int8* filter_data = GetOfflineTensorData(params->tid_filter);
//   const int32* bias_data= reinterpret_cast<const int32*>(GetOfflineTensorData(params->tid_bias));
//   int8* output_data = GetTensorData(params->tid_output);

//   if (params->depth_multiplier == 1) {
//     int16_t* buf = reinterpret_cast<int16_t*>(arena+scratch_buffer_offset);
//     // int16_t* buf = nullptr;
//     // auto* buffer_idx = reinterpret_cast<int*>(node->user_data);
//     // if (*buffer_idx > -1) {
//     //   void* raw = context->GetScratchBuffer(context, *buffer_idx);
//     //   buf = reinterpret_cast<int16_t*>(raw);
//     // }
//     // printf("arm_depthwise_conv_s8_opt\n");
//     // TF_LITE_ENSURE_EQ(
//     //     context,
//         arm_depthwise_conv_s8_opt(
//             combined_ifmap.inputs_data, input_width, input_height,
//             input_depth, filter_data, input_depth,
//             filter_width, filter_height, data.padding.width,
//             data.padding.height, params->stride_width,
//             params->stride_height, bias_data,
//             output_data, data.per_channel_output_shift,
//             data.per_channel_output_multiplier, output_width, output_height,
//             *GetTensorQuantZP(params->tid_output)/* op_params.output_offset */,
//             -(*GetTensorQuantZP(params->tid_input)) /* op_params.input_offset */,
//             std::numeric_limits<int8_t>::min()/* op_params.quantized_activation_min */,
//             std::numeric_limits<int8_t>::max()/* op_params.quantized_activation_max */,
//             params->dilation_width_factor, params->dilation_height_factor, buf);//,
//         // ARM_MATH_SUCCESS);
//   } else {
//     // TF_LITE_ENSURE_EQ(
//     //     context,
//         arm_depthwise_conv_s8(
//             combined_ifmap.inputs_data, input_width, input_height,
//             input_depth, filter_data,
//             params->depth_multiplier * input_depth,
//             params->depth_multiplier, filter_width, filter_height,
//             data.padding.width, data.padding.height,
//             params->stride_width, params->stride_height,
//             bias_data, output_data,
//             data.per_channel_output_shift, data.per_channel_output_multiplier,
//             output_width, output_height, *GetTensorQuantZP(params->tid_output)/* op_params.output_offset */,
//             -(*GetTensorQuantZP(params->tid_input))/* op_params.input_offset */,
//             std::numeric_limits<int8_t>::min()/* op_params.quantized_activation_min */,
//             std::numeric_limits<int8_t>::max()/* op_params.quantized_activation_max */,
//             params->dilation_width_factor,
//             params->dilation_height_factor, nullptr);//,
//         // ARM_MATH_SUCCESS);
//   }

// #else
//   // DepthwiseConvPerChannel
//   const int8* input_data = GetTensorData(params->tid_input);
//   const int8* filter_data = GetOfflineTensorData(params->tid_filter);
//   const int32* bias_data= reinterpret_cast<const int32*>(GetOfflineTensorData(params->tid_bias));
//   int8* output_data = GetTensorData(params->tid_output);

//   const int stride_width = params->stride_width;
//   const int stride_height = params->stride_height;
//   const int dilation_width_factor = params->dilation_width_factor;
//   const int dilation_height_factor = params->dilation_height_factor;
//   const int pad_width = data.padding.width;
//   const int pad_height = data.padding.height;
//   const int depth_multiplier = params->depth_multiplier;
//   const int32 input_offset = -*GetTensorQuantZP(params->input);
//   // const int32 weight_offset = 0;
//   const int32 output_offset = *GetTensorQuantZP(params->output);
//   const int32 output_activation_min = std::numeric_limits<int8_t>::min();
//   const int32 output_activation_max = std::numeric_limits<int8_t>::max();
//   const int32* output_multiplier = data.per_channel_output_multiplier;
//   const int32* output_shift = data.per_channel_output_shift;
//   // Check dimensions of the tensors.
//   // TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
//   // TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
//   // TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

//   // TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
//   const int batches = std::min(input_shape[0], output_shape[0]);// MatchingDim(input_shape, 0, output_shape, 0);
//   // const int output_depth = std::min(filter_shape[3], output_shape[3]); // MatchingDim(filter_shape, 3, output_shape, 3);

//   for (int batch = 0; batch < batches; ++batch) {
//     for (int out_y = 0; out_y < output_height; ++out_y) {
//       for (int out_x = 0; out_x < output_width; ++out_x) {
//         for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
//           for (int m = 0; m < depth_multiplier; ++m) {
//             const int output_channel = m + in_channel * depth_multiplier;
//             const int in_x_origin = (out_x * stride_width) - pad_width;
//             const int in_y_origin = (out_y * stride_height) - pad_height;
//             int32 acc = 0;
//             for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
//               for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
//                 const int in_x = in_x_origin + dilation_width_factor * filter_x;
//                 const int in_y =
//                     in_y_origin + dilation_height_factor * filter_y;
//                 // Zero padding by omitting the areas outside the image.
//                 const bool is_point_inside_image =
//                     (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
//                     (in_y < input_height);
//                 if (is_point_inside_image) {
//                   // int32 input_val = input_data[Offset(input_shape, batch, in_y,
//                   //                                     in_x, in_channel)];
//                   int32 input_val =
//                       input_data[((batch * input_height + in_y) * input_width + in_x) * input_shape[3] + in_channel];
//                   // int32 filter_val = filter_data[Offset(
//                   //     filter_shape, 0, filter_y, filter_x, output_channel)];
//                   int32 filter_val =
//                       filter_data[((0 * filter_height + filter_y) * filter_width + filter_x) * filter_shape[3] + output_channel];
//                   // Accumulate with 32 bits accumulator.
//                   // In the nudging process during model quantization, we force
//                   // real value of 0.0 be represented by a quantized value. This
//                   // guarantees that the input_offset is a int8, even though it
//                   // is represented using int32.
//                   // int32 += int8 * (int8 - int8) so the highest value we can
//                   // get from each accumulation is [-127, 127] * ([-128, 127] -
//                   // [-128, 127]), which is [-32512, 32512]. log2(32512)
//                   // = 14.98, which means we can accumulate at least 2^16
//                   // multiplications without overflow. The accumulator is
//                   // applied to a filter so the accumulation logic will hold as
//                   // long as the filter size (filter_y * filter_x * in_channel)
//                   // does not exceed 2^16, which is the case in all the models
//                   // we have seen so far.
//                   // TODO(jianlijianli): Add a check to make sure the
//                   // accumulator depth is smaller than 2^16.
//                   acc += filter_val * (input_val + input_offset);
//                 }
//               }
//             }
//             if (bias_data) {
//               acc += bias_data[output_channel];
//             }
//             acc = MultiplyByQuantizedMultiplier(
//                 acc, output_multiplier[output_channel],
//                 output_shift[output_channel]);
//             acc += output_offset;
//             acc = std::max(acc, output_activation_min);
//             acc = std::min(acc, output_activation_max);
//             // output_data[Offset(output_shape, batch, out_y, out_x,
//             //                    output_channel)] = static_cast<int8_t>(acc);
//             output_data[((batch * output_height + out_y) * output_width + out_x) * output_shape[3] + output_channel] =
//                 static_cast<int8_t>(acc);
//           }
//         }
//       }
//     }
//   }

// #endif

}

void depthwise_conv_2d(int shared_param_idx, int output_split_id, int scratch_buffer_offset){
  #if HIDE_DWCONV
    return;
  #endif
  const SharedParam_Depthwise_Conv* params = &shared_param_dwconv[shared_param_idx];

  const Tensor& output = tensors[params->tid_output];
  const Tensor& input = tensors[params->tid_input];
  const Tensor& filter = tensors[params->tid_filter];
  // const Tensor& bias = tensors[params->tid_bias];

  // Assune data_type is int8
  // const TfLiteType data_type = input->type;
  int total_width = input.dims[2]/* SizeOfDimension(input, 2) */;
  int total_height = input.dims[1];
  int filter_width = filter.dims[2]/* SizeOfDimension(filter, 2) */;
  int filter_height = filter.dims[1]/* SizeOfDimension(filter, 1) */;

  const DIM_TYPE* output_shape = output.dims;
  const int output_width = output_shape[2];
  const int output_height = (output_split_id>=(output_shape[1]/SPLIT_HEIGHT))? (output_shape[1]%SPLIT_HEIGHT):(SPLIT_HEIGHT) ;

  OP_utils::DepthwiseConv::OpData data;

  #if !OPT_OFFLOAD_ENABLE
  // v0.1 method: call CalculateOpData
  // Calculate padding and quantization params
  CalculateOpData(params, total_width, total_height,
                  filter_width, filter_height,
                  output_shape[2], output_shape[1], &data);
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
  // Method II: copy to SRAM
  // for (int i = 0; i < input.dims[3]; i++)
  //   data.per_channel_output_multiplier[i] = op_data_buffer[4+i];
  // for (int i = 0; i < input.dims[3]; i++)
  //   data.per_channel_output_shift[i] = op_data_buffer[4+input.dims[3]+i];
  #endif

  int input_split_id_begin = (output_split_id*SPLIT_HEIGHT) * params->stride_height - data.padding.height;
  int input_split_id_end = input_split_id_begin + (filter_height-1) + params->stride_height*(SPLIT_HEIGHT-1);
  input_split_id_end = input_split_id_end < total_height? input_split_id_end:total_height-1;
  data.padding.height = input_split_id_begin >= 0 ? 0:abs(input_split_id_begin);
  input_split_id_begin = input_split_id_begin > 0 ? input_split_id_begin: 0;
  const VirtualFp<int8_t> combined_ifmap(params->tid_input, input, input_split_id_begin, input_split_id_end, total_width*input.dims[3]);

  const int8* filter_data = GetOfflineTensorData(params->tid_filter);
  const int32* bias_data= reinterpret_cast<const int32*>(GetOfflineTensorData(params->tid_bias));
  int8* output_data = GetSplitData(params->tid_output, output_split_id);


  if (SIMD) {
    depthwise_conv_simd(params, &data, scratch_buffer_offset, combined_ifmap, filter_data, bias_data, output_data, output_height);
  }
  else {
    depthwise_conv_serial(params, &data, combined_ifmap, filter_data, bias_data, output_data, output_height);
  }
  // if (shared_param_idx == 0)
  //   print_split(params->tid_output, output_split_id);
}

extern "C" {
#include "tiny_depthwise_conv.h"
}

#include "tiny_dwconv_headers.h"


#endif