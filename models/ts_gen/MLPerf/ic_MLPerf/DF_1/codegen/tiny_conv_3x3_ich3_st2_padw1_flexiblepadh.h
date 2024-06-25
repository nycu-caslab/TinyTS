#ifndef _TINY_CONV_3X3_ICH3_ST2_PAD1_H
#define _TINY_CONV_3X3_ICH3_ST2_PAD1_H_

void arm_convolve_3x3_ich3_st2_padw1_flexiblepadh_s8_tiny(const int8_t * const *input, const uint16_t input_x,
		const uint16_t input_y, const uint16_t input_ch, const int8_t *kernel,
		const int32_t *bias, const int32_t *output_shift,
		const int32_t *output_mult, const int32_t out_offset,
		const int32_t input_offset, const int32_t out_activation_min,
		const int32_t out_activation_max, int8_t *output, const uint16_t output_x,
		const uint16_t output_y, const uint16_t output_ch, int16_t *runtime_buf, uint16_t pad_yt, uint16_t pad_yb);

#endif