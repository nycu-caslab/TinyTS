#ifndef _AVERAGE_POOL_H_
#define _AVERAGE_POOL_H_

#include <algorithm>

#include "gen_model/include/op_param.h"
#include "gen_lib/include/ctx_util.h"
#include "gen_lib/include/padding.h"
#include "gen_lib/include/kernel_util.h"
#include "gen_lib/tensorflow/cppmath.h"

#include "cmsis/CMSIS/NN/Include/arm_nnfunctions.h"

#include <algorithm>

// Comment serial implementatino for deprication
// inline void average_pool_2d_serial(const SharedParam_AvgPool* params, const Tensor* input, const Tensor* output, 
//                     int32_t& act_min, int32_t& act_max){
// 	int8_t *input_data = GetTensorData(params->input);
// 	int8_t *output_data = GetTensorData(params->output);
//
// 	const int batches = input->dims[0];
// 	const int depth = input->dims[3];
// 	const int input_height = input->dims[1];
// 	const int input_width = input->dims[2];
// 	const int output_height = output->dims[1];
// 	const int output_width = output->dims[2];
// 	const int stride_height = params->stride_h;
// 	const int stride_width = params->stride_w;
// 	const int filter_height = params->filter_height;
// 	const int filter_width = params->filter_width;
// 	int dummy;
// 	const int padding_height = ComputePaddingWithOffset(stride_height, 1,
// 															input_height, filter_height, 
// 															output_height, &dummy);
// 	const int padding_width  = ComputePaddingWithOffset(stride_width, 1,
// 															input_width, filter_width, 
// 															output_width, &dummy);
// 	for (int batch = 0; batch < batches; ++batch) {
// 		for (int out_y = 0; out_y < output_height; ++out_y) {
// 			for (int out_x = 0; out_x < output_width; ++out_x) {
// 				for (int channel = 0; channel < depth; ++channel) {
// 				const int in_x_origin =
// 						(out_x * stride_width) - padding_width;
// 				const int in_y_origin =
// 						(out_y * stride_height) - padding_height;
// 				// Compute the boundaries of the filter region clamped so as to
// 				// ensure that the filter window fits in the input array.
// 				const int filter_x_start = std::max(0, -in_x_origin);
// 				const int filter_x_end =
// 						std::min(filter_width, input_width - in_x_origin);
// 				const int filter_y_start = std::max(0, -in_y_origin);
// 				const int filter_y_end =
// 						std::min(filter_height, input_height - in_y_origin);
// 				int32 acc = 0;
// 				int filter_count = 0;
// 				for (int filter_y = filter_y_start; filter_y < filter_y_end;
// 					++filter_y) {
// 					for (int filter_x = filter_x_start; filter_x < filter_x_end;
// 							++filter_x) {
// 					const int in_x = in_x_origin + filter_x;
// 					const int in_y = in_y_origin + filter_y;
// 					acc +=
// 							input_data[((batch*input_height + in_y)*input_width + in_x)*depth + channel];
// 					filter_count++;
// 					}
// 				}
// 				// Round to the closest integer value.
// 				acc = acc > 0 ? (acc + filter_count / 2) / filter_count
// 												: (acc - filter_count / 2) / filter_count;
// 				acc = std::max(acc, act_min);
// 				acc = std::min(acc, act_max);
// 				output_data[((batch*input_height + out_y)*input_width + out_x)*depth + channel] =
// 						static_cast<int8>(acc);
// 				}
// 			}
// 		}
// 	}
// }

void average_pool_2d_simd(	const SharedParam_AvgPool* params, int scratch_buffer_offset,
							const Tensor* input, const Tensor* output, 
                    		int32_t& activation_min, int32_t& activation_max){

  const DIM_TYPE* input_shape  = input->dims;
  const DIM_TYPE* output_shape = output->dims;
	
  // Calculate padding value
  int height = input_shape[1];
  int width  = input_shape[2];

  int out_height, out_width;

  PaddingValues padding;

  padding = ComputePaddingHeightWidth(
      params->stride_h, params->stride_w,
      /*dilation_rate_height=*/1, /*dilation_rate_width=*/1, 
	  height, width, params->filter_height, params->filter_width,
	  params->padding, &out_height, &out_width);
	
  const int depth = std::min(input_shape[3], output_shape[3]);
  int16_t* scratch_buffer = reinterpret_cast<int16_t*>(arena + scratch_buffer_offset);

  cmsis_nn_dims input_dims;
  input_dims.n = 1;
  input_dims.h = input_shape[1];
  input_dims.w = input_shape[2];
  input_dims.c = depth;

  cmsis_nn_dims output_dims;
  output_dims.n = 1;
  output_dims.h = output_shape[1];
  output_dims.w = output_shape[2];
  output_dims.c = depth;

  cmsis_nn_pool_params pool_params;
  pool_params.stride.h = params->stride_h;
  pool_params.stride.w = params->stride_w;
  pool_params.padding.h = padding.height;
  pool_params.padding.w = padding.width;
  pool_params.activation.min = activation_min;
  pool_params.activation.max = activation_max;

  cmsis_nn_dims filter_dims;
  filter_dims.n = 1;
  filter_dims.h = params->filter_height;
  filter_dims.w = params->filter_width;
  filter_dims.c = 1;

  cmsis_nn_context ctx;
  ctx.buf = scratch_buffer;
  ctx.size = 0;

  arm_avgpool_s8(&ctx, &pool_params, &input_dims,
                 GetTensorData(params->input), &filter_dims,
                 &output_dims, GetTensorData(params->output));
      // arm_avgpool_s8(input_height, input_width, output_height, output_width,
      //                stride_height, stride_width, filter_height, filter_width,
      //                padding_height, padding_width, activation_min,
      //                activation_max, depth, GetTensorData(params->input),
      //                scratch_buffer, GetTensorData(params->output)); //,
} 

void average_pool_2d(int shared_param_idx, int scratch_buffer_offset){
	const SharedParam_AvgPool *params = &shared_param_avgpool[shared_param_idx];
	const Tensor *input = &tensors[params->input];
	const Tensor *output = &tensors[params->output];

	int32_t act_min, act_max;
	CalculateActivationRangeQuantized(params->fused_ActFunc, 
																	output,
																	&act_min, &act_max);

#if SIMD
	average_pool_2d_simd(params, scratch_buffer_offset, input, output, act_min, act_max);
#else
	average_pool_2d_serial(params, input, output, act_min, act_max);
#endif
}
#endif