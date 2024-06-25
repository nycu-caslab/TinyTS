#include "gen_lib/include/kernel_util.h"

#include <algorithm>
#include <limits>
#include <memory>

#include "gen_lib/include/builtin_op_data.h"
#include "gen_lib/include/common.h"
#include "gen_lib/include/quantization_util.h"
#include "gen_lib/tensorflow/cppmath.h"
// #include "gen_lib/tensorflow/round.h"

extern const int quant_min[];
extern const int quant_max[];
extern const int quant_scale[];
extern const int quant_zeropoint[];

// Per-axis & per-tensor
TfLiteStatus PopulateConvolutionQuantizationParams(
    /* TfLiteContext* context, */ const Tensor* input,
    const Tensor* filter, const Tensor* bias, Tensor* output,
    const int& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int* per_channel_shift, int num_channels) {
//   TF_LITE_ENSURE_EQ(context, input->quantization.type,
//                     kTfLiteAffineQuantization);
//   TF_LITE_ENSURE_EQ(context, filter->quantization.type,
//                     kTfLiteAffineQuantization);
  // TODO(jianlijianli): Enable bias type check and bias scale == input scale
  // * filter scale for each channel in affine quantization once bias
  // quantization is properly populated.
  // TF_LITE_ENSURE_EQ(context, bias->quantization.type,
  // kTfLiteAffineQuantization);

  // Check data type.
//   const auto* affine_quantization =
//       reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
//   TF_LITE_ENSURE(context, affine_quantization);
//   TF_LITE_ENSURE(context, affine_quantization->scale);
/*   const bool is_per_channel = affine_quantization->scale->size > 1; */
//   if (is_per_channel) {
    //  Currently only Int8/Int16 is supported for per channel quantization.
//     TF_LITE_ENSURE(context,
//                    input->type == kINT8 || input->type == kINT16);
//     TF_LITE_ENSURE_EQ(context, filter->type, kINT8);
//     TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size, num_channels);
//     TF_LITE_ENSURE_EQ(
//         context, num_channels,
//         filter->dims->data[affine_quantization->quantized_dimension]);
//   }

  // Populate multiplier and shift using affine quantization.
  const float input_scale = reinterpret_cast<const float*>(&quant_scale[input->quant_offset])[0];
  const float output_scale = reinterpret_cast<const float*>(&quant_scale[output->quant_offset])[0];
  const float* filter_scales = reinterpret_cast<const float*>(&quant_scale[filter->quant_offset]);
  for (int i = 0; i < num_channels; ++i) {
    // If per-tensor quantization parameter is specified, broadcast it along the
    // quantization dimension (channels_out).
    const float scale = /* is_per_channel ?  */filter_scales[i]/*  : filter_scales[0] */;
    const double filter_scale = static_cast<double>(scale);
    const double effective_output_scale = static_cast<double>(input_scale) *
                                          filter_scale /
                                          static_cast<double>(output_scale);
    int32_t significand;
    int channel_shift;
    QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
    per_channel_multiplier[i] = significand;
    per_channel_shift[i] = channel_shift;
  }

  // Populate scalar quantization parameters.
  // This check on legacy quantization parameters is kept only for backward
  // compatibility.
//   if (input->type == kUINT8) {
//     // Check bias scale == input scale * filter scale.
//     double real_multiplier = 0.0;
//     TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
//         /* context, */ input, filter, bias, output, &real_multiplier));
//     int exponent;

//     // Populate quantization parameters with multiplier and shift.
//     QuantizeMultiplier(real_multiplier, multiplier, &exponent);
//     *shift = -exponent;
//   }
//   if (input->type == kINT8 || /* input->type == kUINT8 || */
//       input->type == kINT16) {
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        /* context, */ activation, output, output_activation_min,
        output_activation_max));
//   }
  return kTfLiteOk;
}

TfLiteStatus GetQuantizedConvolutionMultipler(/* TfLiteContext* context, */
                                              const Tensor* input,
                                              const Tensor* filter,
                                              const Tensor* bias,
                                              const Tensor* output,
                                              double* multiplier) {
  const double input_product_scale = static_cast<double>(*reinterpret_cast<const float*>(&quant_scale[input->quant_offset])) *
                                     static_cast<double>(*reinterpret_cast<const float*>(&quant_scale[filter->quant_offset]));
  // TODO(ahentz): The following conditions must be guaranteed by the training
  // pipeline.
  if (bias) {
    const double bias_scale = static_cast<double>(*reinterpret_cast<const float*>(&quant_scale[bias->quant_offset]));
    // Here we're making sure the input_product_scale & bias_scale are about the
    // same. Since we have:
    // (output - output_zp) * output_scale =
    // input_product_scale * input_product + bias * bias_scale ---- (0)
    //
    // (0) equals:
    // (input_product + bias) * input_product_scale ----- (1)
    //           +
    // bias * (bias_scale - input_product_scale)   ------ (2)
    //
    // For the real kernel computation, we're doing (1), so we really need to
    // make sure (2) has minimum impact on the output, so:
    // bias * (bias_scale - input_product_scale) / output_scale should be
    // a small number for an integer.
    // Since normally bias should be within a small range.
    // We should expect (bias_scale - input_product_scale) / output_scale to
    // be a small number like 0.02.
    const double scale_diff = std::abs(input_product_scale - bias_scale);
    const double output_scale = static_cast<double>(*reinterpret_cast<const float*>(&quant_scale[output->quant_offset]));

    /* TF_LITE_ENSURE(context, scale_diff / output_scale <= 0.02); */
  }
  return GetQuantizedConvolutionMultipler(/* context,  */input, filter, output,
                                          multiplier);
}

TfLiteStatus GetQuantizedConvolutionMultipler(/* TfLiteContext* context, */
                                              const Tensor* input,
                                              const Tensor* filter,
                                              const Tensor* output,
                                              double* multiplier) {
  const double input_product_scale =
      static_cast<double>(*reinterpret_cast<const float*>(&quant_scale[input->quant_offset]) * *reinterpret_cast<const float*>(&quant_scale[filter->quant_offset]));
  /* TF_LITE_ENSURE(context, input_product_scale >= 0); */
  *multiplier = input_product_scale / static_cast<double>(*reinterpret_cast<const float*>(&quant_scale[output->quant_offset]));

  return kTfLiteOk;
}

namespace {
void CalculateActivationRangeQuantizedImpl(int activation,
                                           int32_t qmin, int32_t qmax,
                                           const Tensor* output,
                                           int32_t* act_min, int32_t* act_max) {
  const auto scale = *reinterpret_cast<const float*>(&quant_scale[output->quant_offset]);
  const auto zero_point = quant_zeropoint[output->quant_offset];

  auto quantize = [scale, zero_point](float f) {
    return zero_point + static_cast<int32_t>(tflite::TfLiteRound(f / scale));
  };

  if (activation == 1 /* kTfLiteActRelu */) {
    *act_min = std::max(qmin, quantize(0.0));
    *act_max = qmax;
  } else if (activation == 3 /* kTfLiteActRelu6 */) {
    *act_min = std::max(qmin, quantize(0.0));
    *act_max = std::min(qmax, quantize(6.0));
  } else if (activation == 2 /* kTfLiteActReluN1To1 */) {
    *act_min = std::max(qmin, quantize(-1.0));
    *act_max = std::min(qmax, quantize(1.0));
  } else {
    *act_min = qmin;
    *act_max = qmax;
  }
}
}  // namespace

TfLiteStatus CalculateActivationRangeQuantized(/* TfLiteContext* context, */
                                               int activation,
                                               const Tensor* output,
                                               int32_t* act_min,
                                               int32_t* act_max) {
  int32_t qmin = 0;
  int32_t qmax = 0;
//   if (output->type == kUINT8) {
//     qmin = std::numeric_limits<uint8_t>::min();
//     qmax = std::numeric_limits<uint8_t>::max();
//   } else if (output->type == kINT8) {
    qmin = std::numeric_limits<int8_t>::min();
    qmax = std::numeric_limits<int8_t>::max();
//   } else if (output->type == kINT16) {
//     qmin = std::numeric_limits<int16_t>::min();
//     qmax = std::numeric_limits<int16_t>::max();
//   } else {
//     /* TF_LITE_ENSURE(context, false); */
//   }

  CalculateActivationRangeQuantizedImpl(activation, qmin, qmax, output, act_min,
                                        act_max);
  return kTfLiteOk;
}