#include "arm_nnfunctions.h"
#include "genNN.h"
#include "tinyengine_function.h"


void arm_convolve_3x3_ich3_st2_padw1_flexiblepadh_s8_tiny(const int8_t * const *input, const uint16_t input_x,
		const uint16_t input_y, const uint16_t input_ch, const int8_t *kernel,
		const int32_t *bias, const int32_t *output_shift,
		const int32_t *output_mult, const int32_t out_offset,
		const int32_t input_offset, const int32_t out_activation_min,
		const int32_t out_activation_max, int8_t *output, const uint16_t output_x,
		const uint16_t output_y, const uint16_t output_ch, int16_t *runtime_buf/* , int16_t *kbuf */, uint16_t pad_yt, uint16_t pad_yb) {
    // TODO: kbuf_size = 27*o_ch*sizeof(16)
    convolve_s8_kernel3_inputch3_stride2_padw1_flexiblepadh(input, input_x, input_y, input_ch, kernel,
      bias, output_shift, output_mult, out_offset, input_offset,
      out_activation_min, out_activation_max, output,
      output_x, output_y, output_ch, runtime_buf, runtime_buf + 2 * 27, -input_offset, pad_yt, pad_yb);

  // int8_t *tensor_data = output;
  // int tensor_size = output_x * output_y * output_ch;
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