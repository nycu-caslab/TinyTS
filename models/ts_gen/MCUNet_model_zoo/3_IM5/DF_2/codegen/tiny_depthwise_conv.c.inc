void TINY_ENGINE_FORWARD_FN(const int8_t* const *input,
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
                                     const int32_t *offsetBias,
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
                                     int16_t *buffer_a) {

  // int8_t transformed[kernel_x * kernel_y * output_ch];
  // int8_t *tr = transformed;

  // //Transform to CHW
  // for (int c = 0; c < output_ch; ++c)  {
  //   for (int h = 0; h < kernel_y; ++h) {
  //     for (int w = 0; w < kernel_x; ++w) {
  //       int src_idx = h * kernel_x * output_ch + w * output_ch + c;
  //       *tr++ = kernel[src_idx];
  //     }
  //   }
  // }

  TINY_ENGINE_IMPL_FN(
          input, input_x, input_y, input_ch, /* transformed */kernel, bias, offsetBias, output_shift, output_mult, output_offset, input_offset,
          output_activation_min, output_activation_max, output, output_x, output_y, output_ch, buffer_a, -input_offset, pad_yt, pad_x, pad_yb);

  // int8_t *tensor_data = output;
  // int tensor_size = output_x * output_y * input_ch;
  // // printf("Tensor Size: %d\n", tensor_size);
  // static int tid = 0;
  // printf("Tensor %2d\n\t", tid);
  // for (int i = 0; i < tensor_size; i++) {
  //     printf("%4d ",tensor_data[i]);
  //     if(i%16==15 ^ i == tensor_size-1)
  //         printf("\n\t");
  // }
  // printf("\nEnd of Tensor %2d\n", tid);
  // tid++;
}


