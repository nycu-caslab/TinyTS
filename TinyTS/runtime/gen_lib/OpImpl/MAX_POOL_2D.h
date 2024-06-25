#ifndef _MAX_POOL_H_
#define _MAX_POOL_H_

#include <algorithm>

#include "gen_model/include/op_param.h"
#include "gen_lib/include/ctx_util.h"
#include "gen_lib/include/padding.h"
#include "gen_lib/include/kernel_util.h"
#include "gen_lib/tensorflow/cppmath.h"

#include "cmsis/CMSIS/NN/Include/arm_nnfunctions.h"

#include <algorithm>

void max_pool_serial(const SharedParam_MaxPool* params, const Tensor* input, const Tensor* output){
  int8_t *input_data = GetTensorData(params->input);
  int8_t *output_data = GetTensorData(params->output);

  const int batches = input->dims[0];
  const int depth = input->dims[3];
  const int input_height = input->dims[1];
  const int input_width = input->dims[2];
  const int output_height = output->dims[1];
  const int output_width = output->dims[2];
  const int stride_height = params->stride_h;
  const int stride_width = params->stride_w;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params->padding_width;
          const int in_y_origin =
              (out_y * stride_height) - params->padding_height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params->filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params->filter_height, input_height - in_y_origin);
          int8_t max = std::numeric_limits<int8_t>::lowest();
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              max = std::max(
                  max,
                  input_data[((batch * input_height + in_y) * input_width + in_x) * depth + channel]);
            }
          }
          max = std::max<int8_t>(max, params->quantized_activation_min);
          max = std::min<int8_t>(max, params->quantized_activation_max);
          output_data[((batch * output_height + out_y) * output_width + out_x) * depth + channel] =
              static_cast<int8_t>(max);
        }
      }
    }
  }
}
void max_pool_simd(const SharedParam_MaxPool* params, uint8_t* scratch_buf, const Tensor* input, const Tensor* output, 
                    int32_t& activation_min, int32_t& activation_max){
  
  const int* input_shape = input->dims;
  // TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);

  const int* output_shape = output->dims;
  // TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  // Calculate padding value
  int height = SizeOfDimension(input, 1);
  int width = SizeOfDimension(input, 2);

  int out_height, out_width;

  PaddingValues padding;

  padding = ComputePaddingHeightWidth(
      params->stride_h, params->stride_w,
      /*dilation_rate_height=*/1,
      /*dilation_rate_width=*/1, height, width, params->filter_height,
      params->filter_width, params->padding, &out_height, &out_width);

  const int depth = std::min(input_shape[3], output_shape[3]);
  const int input_height = input_shape[1];
  const int input_width = input_shape[2];
  const int output_height = output_shape[1];
  const int output_width = output_shape[2];
  const int stride_height = params->stride_h;
  const int stride_width = params->stride_w;

  const int filter_height = params->filter_height;
  const int filter_width = params->filter_width;
  const int padding_height = padding.height;
  const int padding_width = padding.width;

  int16_t* scratch_buffer = reinterpret_cast<int16_t*>(scratch_buf);

  // auto* buffer_idx = reinterpret_cast<int*>(node->user_data);

  // if (*buffer_idx > -1) {
  //   void* raw = context->GetScratchBuffer(context, *buffer_idx);
  //   scratch_buffer = reinterpret_cast<int16_t*>(raw);
  // }

  // TF_LITE_ENSURE_EQ(
  //     context,
      arm_max_pool_s8_opt(input_height, input_width, output_height,
                          output_width, stride_height, stride_width,
                          filter_height, filter_width, padding_height,
                          padding_width, activation_min, activation_max, depth,
                          GetTensorData(params->input), scratch_buffer,
                          GetTensorData(params->output)); //,
      // ARM_MATH_SUCCESS);
}

void max_pool_2d(int shared_param_idx){
    const SharedParam_MaxPool *params = &shared_param_maxpool[shared_param_idx];
    const Tensor *input = &tensors[params->input];
    const Tensor *output = &tensors[params->output];

    int32_t act_min, act_max;
    CalculateActivationRangeQuantized(params->fused_ActFunc, 
                                    output,
                                    &act_min, &act_max);

  if(SIMD){
    max_pool_simd(params, GetOpScratchBuf(params->output, 1), input, output, act_min, act_max);
  }
  else{
    max_pool_serial(params, input, output);
  }
}
#endif