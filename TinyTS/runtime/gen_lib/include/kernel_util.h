#ifndef GEN_INCLUDE_KERNEL_UTIL_H_
#define GEN_INCLUDE_KERNEL_UTIL_H_

#include "gen_lib/include/common.h"
#include "gen_lib/include/builtin_op_data.h"

// Check dimensionality match and populate OpData for Conv and DepthwiseConv.
TfLiteStatus PopulateConvolutionQuantizationParams(
    /* TfLiteContext* context, */ const Tensor* input,
    const Tensor* filter, const Tensor* bias, Tensor* output,
    const int& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int* per_channel_shift);

TfLiteStatus PopulateConvolutionQuantizationParams(
    /* TfLiteContext* context, */ const Tensor* input,
    const Tensor* filter, const Tensor* bias, Tensor* output,
    const int& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int* per_channel_shift, int num_channels);

// Calculates the multiplication factor for a quantized convolution (or
// quantized depthwise convolution) involving the given tensors. Returns an
// error if the scales of the tensors are not compatible.
TfLiteStatus GetQuantizedConvolutionMultipler(/* TfLiteContext* context, */
                                              const Tensor* input,
                                              const Tensor* filter,
                                              const Tensor* bias,
                                              const Tensor* output,
                                              double* multiplier);

TfLiteStatus GetQuantizedConvolutionMultipler(/* TfLiteContext* context, */
                                              const Tensor* input,
                                              const Tensor* filter,
                                              const Tensor* output,
                                              double* multiplier);

// Calculates the useful quantized range of an activation layer given its
// activation tensor.
TfLiteStatus CalculateActivationRangeQuantized(/* TfLiteContext* context, */
                                               int activation,
                                               const Tensor* output,
                                               int32_t* act_min,
                                               int32_t* act_max);

inline int SizeOfDimension(const Tensor* t, int dim) {
  return t->dims[dim];
}

#endif