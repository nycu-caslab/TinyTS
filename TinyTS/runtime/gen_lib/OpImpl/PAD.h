#ifndef _PAD_2D_H_
#define _PAD_2D_H_
#include "gen_model/include/op_param.h"
#include "gen_lib/include/ctx_util.h"

#include "gen_lib/include/types.h"
#include "gen_lib/include/padding.h"
#include "gen_lib/include/kernel_common.h"
#include "gen_lib/include/kernel_util.h"
#include "gen_lib/include/quantization_util.h"
#include "gen_lib/include/virtualfp.h"

#include <algorithm>

void pad(int shared_param_idx) {
  const SharedParam_Pad* params = &shared_param_pad[shared_param_idx];

  // Get Data
  const Tensor& padding = tensors[params->padding];
  const Tensor& output = tensors[params->output];

  const int8 *raw_input = GetTensorData(params->input);
  int8 *raw_output = GetTensorData(params->output);
  const int32* padding_data = reinterpret_cast<const int32*>(GetOfflineTensorData(params->padding));

  // Get pad value
  int8_t pad_value;
  pad_value = static_cast<int8_t>(*GetTensorQuantZP(&output));

  // Get shape
  const auto *output_shape = output.dims;
  const auto *padding_shape = padding.dims;

  const int output_batch = output_shape[0];
  const int output_height = output_shape[1];
  const int output_width = output_shape[2];
  const int output_depth = output_shape[3];

  const int padding_height = padding_shape[0];
  const int padding_width = padding_shape[1];

  // pad config
  const int left_b_padding = padding_data[0];
  const int right_b_padding = padding_data[1];

  const int left_h_padding = padding_data[2];
  const int right_h_padding = padding_data[3];

  const int left_w_padding = padding_data[4];
  const int right_w_padding = padding_data[5];

  const int left_d_padding = padding_data[6];
  const int right_d_padding = padding_data[7];

  // pad pad_value
  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_h = 0; out_h < output_height; ++out_h) {
      for (int out_w = 0; out_w < output_width; ++out_w) {
        for (int out_d = 0; out_d < output_depth; ++out_d) {
          if (out_b < left_b_padding ||
              out_b >= output_batch - right_b_padding ||
              out_h < left_h_padding ||
              out_h >= output_height - right_h_padding ||
              out_w < left_w_padding ||
              out_w >= output_width - right_w_padding ||
              out_d < left_d_padding ||
              out_d >= output_depth - right_d_padding) {
            *raw_output++ = pad_value;
          } else {
            *raw_output++ = *raw_input++;
          }
        }
      }
    }
  }
}


void pad(const int shared_param_idx, const int sid) {
  const SharedParam_Pad* params = &shared_param_pad[shared_param_idx];

  // Get Data
  const Tensor& padding = tensors[params->padding];
  const Tensor& output = tensors[params->output];

  const int8 *raw_input = GetSplitData(params->input, sid);
  // printf("(%d,%d): sid before = %d, ", shared_param_idx, sid, sid);
  int8 *raw_output = GetSplitData(params->output, sid);
  // printf("sid after = %d, ", sid);
  const int32* padding_data = reinterpret_cast<const int32*>(GetOfflineTensorData(params->padding));

  // Get pad value
  int8_t pad_value;
  pad_value = static_cast<int8_t>(*GetTensorQuantZP(&output));

  // Get shape
  const auto *output_shape = output.dims;
  const auto *padding_shape = padding.dims;

  const int output_batch = output_shape[0];
  const int output_height = output_shape[1];
  const int output_width = output_shape[2];
  const int output_depth = output_shape[3];

  const int padding_height = padding_shape[0];
  const int padding_width = padding_shape[1];

  // Get data
  const int8* input_data = GetTensorData(params->input);
  int8* output_data = GetTensorData(params->output);

  // pad config
  const int left_b_padding = padding_data[0];
  const int right_b_padding = padding_data[1];

  const int left_h_padding = padding_data[2];
  const int right_h_padding = padding_data[3];

  const int left_w_padding = padding_data[4];
  const int right_w_padding = padding_data[5];

  const int left_d_padding = padding_data[6];
  const int right_d_padding = padding_data[7];

  // calc height of this split
  int8_t split_height = output_height - SPLIT_HEIGHT*sid;
  if (split_height > SPLIT_HEIGHT){
    split_height = SPLIT_HEIGHT;
  }
  // printf("split_height = %d\n", split_height);

  // pad pad_value
  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_h = 0; out_h < split_height; ++out_h) {
      for (int out_w = 0; out_w < output_width; ++out_w) {
        for (int out_d = 0; out_d < output_depth; ++out_d) {
          if (out_b < left_b_padding ||
              out_b >= output_batch - right_b_padding ||
              out_h < left_h_padding ||
              out_h >= output_height - right_h_padding ||
              out_w < left_w_padding ||
              out_w >= output_width - right_w_padding ||
              out_d < left_d_padding ||
              out_d >= output_depth - right_d_padding) {
            *raw_output++ = pad_value;
          } else {
            *raw_output++ = *raw_input++;
          }
        }
      }
    }
  }
}

#endif _PAD_2D_H