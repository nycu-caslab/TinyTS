#ifndef _SOFTMAX_H_
#define _SOFTMAX_H_
#include "gen_model/include/op_param.h"
#include "gen_lib/include/ctx_util.h"

#include "cmsis/CMSIS/NN/Include/arm_nnfunctions.h"
#include "cmsis/CMSIS/NN/Include/arm_nnsupportfunctions.h"
// #include "gen_lib/include/arm_nnsupportfunctions.h"
// #include "arm_nnsupportfunctions.h"

#define ACCUM_BITS 12

namespace OP_utils{
namespace softmax{
}
}
struct SoftmaxParams {
  // beta is not really used (not a Tensorflow parameter) and not implemented
  // for LogSoftmax.
  double beta;
  // uint8 inference params.  Used even when beta defaults to 1.0.
  int32 input_multiplier;
  int32 input_left_shift;
  // Reverse scaling is only used by LogSoftmax.
  int32 reverse_scaling_divisor;
  int32 reverse_scaling_right_shift;
  int diff_min;
  int32_t zero_point;
  float scale;
  float* table;
  int16_t* exp_lut;
  int16_t* one_over_one_plus_x_lut;
  uint8_t* uint8_table1;
  uint8_t* uint8_table2;
};

TfLiteStatus CalculateSoftmaxParams(/* TfLiteContext* context, */
                                    const Tensor* input,
                                    const Tensor* output,
                                    const SharedParam_Softmax* params,
                                    SoftmaxParams* op_data) {
  /* if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) { */
    /* if (input->type == kTfLiteUInt8) {
      TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteUInt8);
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    } else {
      TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt8);
      TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt8);
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, -128);
    } */
    /* output->params.scale = 1.f / 256; */
    /* TF_LITE_ENSURE(context, (output->params.scale == 1.f / 256) ||
                                (output->params.scale == 1.f / 255));
 */
    static const int kScaledDiffIntegerBits = 5;

    int input_left_shift;
    PreprocessSoftmaxScaling(
        static_cast<double>(*reinterpret_cast<const float*>(&params->beta)),
        static_cast<double>(*GetTensorQuantScale(input)/* input->params.scale */), kScaledDiffIntegerBits,
        &op_data->input_multiplier, &input_left_shift);
    op_data->input_left_shift = input_left_shift;
    op_data->diff_min =
        -1.0 * CalculateInputRadius(kScaledDiffIntegerBits,
                                            op_data->input_left_shift);
  /* } else {
    TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
    op_data->beta = static_cast<double>(params->beta);
  } */
  return kTfLiteOk;
}

// A combination of MatchingFlatSize() and FlatSizeSkipDim().
// template <int N>
// inline int MatchingFlatSizeSkipDim(const Dims<N>& dims, int skip_dim,
//                                    const Dims<N>& check_dims_0) {
//   for (int i = 0; i < N; ++i) {
//     if (i != skip_dim) {
//       TFLITE_DCHECK_EQ(ArraySize(dims, i), ArraySize(check_dims_0, i));
//     }
//   }
//   return FlatSizeSkipDim(dims, skip_dim);
// }

// Data is required to be contiguous, and so many operators can use either the
// full array flat size or the flat size with one dimension skipped (commonly
// the depth).
// template <int N>
inline int FlatSizeSkipDim(const DIM_TYPE* dims/* Dims<N>& dims */, int skip_dim, int dimention_count) {
  /* TFLITE_DCHECK(skip_dim >= 0 && skip_dim < N); */
  int flat_size = 1;
  for (int i = 0; i < dimention_count/* N */; ++i) {
    flat_size *= (i == skip_dim) ? 1 : dims[i];
  }
  return flat_size;
}

void softmax(int shared_param_idx){
  const SharedParam_Softmax* params = &shared_param_softmax[shared_param_idx];

  const Tensor& input = tensors[params->input]; 
  const Tensor& output = tensors[params->output];

  const int8* input_data = GetTensorData(params->input);
  int8* output_data = GetTensorData(params->output);

  SoftmaxParams op_data;
  /* TF_LITE_ENSURE_STATUS( */
      CalculateSoftmaxParams(/* context,  */&input, &output, params, &op_data);
  
  /* SoftmaxQuantized */
  const auto input_shape = input.dims;
  const auto output_shape = output.dims;

  /* const unsigned int num_dims = NumDimensions(input); */

  const int trailing_dim = 2 /* input_shape.DimensionsCount() */ - 1;
  const int outer_size =
      /* Matching */FlatSizeSkipDim(input_shape, trailing_dim, 2/* , output_shape */);
  const int depth = std::min(input.dims[trailing_dim], output.dims[trailing_dim])
      /* MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim) */;

  arm_softmax_s8(input_data, outer_size, depth,
                  op_data.input_multiplier, op_data.input_left_shift,
                  op_data.diff_min, output_data);
}
#endif