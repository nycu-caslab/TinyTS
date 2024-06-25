
#ifndef _TINY_DWCONV_H_
#define _TINY_DWCONV_H_

void arm_depthwise_conv_s8_tiny_kernel3x3_stride1(
                                     const int8_t* const *input,
                                     const uint16_t input_x,
                                     const uint16_t input_y,
                                     const uint16_t input_ch,
                                     const int8_t *kernel,
                                     const uint16_t output_ch,
                                     const uint16_t kernel_x,
                                     const uint16_t kernel_y,
                                     const uint16_t pad_x,
                                     const uint16_t pad_yt,
                                     const uint16_t pad_yb,
                                     const uint16_t stride_x,
                                     const uint16_t stride_y,
                                     const int32_t *bias,
                                     const int32_t *biasOffset,
                                     int8_t *output,
                                     const int32_t *output_shift,
                                     const int32_t *output_mult,
                                     const uint16_t output_x,
                                     const uint16_t output_y,
                                     const int32_t output_offset,
                                     const int32_t input_offset,
                                     const int32_t output_activation_min,
                                     const int32_t output_activation_max,
                                     const uint16_t dilation_x,
                                     const uint16_t dilation_y,
                                     int16_t *buffer_a);
                
#endif
