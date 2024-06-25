/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   convolve_1x1_s8_ch16.c
 *
 * Reference papers:
 *  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
 *  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
 *  - MCUNetV3: On-Device Training Under 256KB Memory, arXiv:2206.15472
 * Contact authors:
 *  - Wei-Ming Chen, wmchen@mit.edu
 *  - Wei-Chen Wang, wweichen@mit.edu
 *  - Ji Lin, jilin@mit.edu
 *  - Ligeng Zhu, ligeng@mit.edu
 *  - Song Han, songhan@mit.edu
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "img2col_element.h"
#include "tinyengine_function.h"

#define DIM_KER_X (1U)
#define DIM_KER_Y (1U)

tinyengine_status convolve_1x1_s8_ch16(const int8_t *input, const uint16_t input_x,
		const uint16_t input_y, const uint16_t input_ch, const int8_t *kernel,
		const int32_t *bias, const int32_t *output_shift,
		const int32_t *output_mult, const int32_t out_offset,
		const int32_t input_offset, const int32_t out_activation_min,
		const int32_t out_activation_max, int8_t *output, const uint16_t output_x,
		const uint16_t output_y, const uint16_t output_ch, int16_t *runtime_buf) {
	int32_t i_element;
	(void) input_x;
	(void) input_y;

	/* Partial(two columns) im2col buffer */
	int16_t *two_column_buffer = runtime_buf;
	int8_t *out = output;
	const int32_t num_elements = output_x * output_y;
	const int channel_div4 = (input_ch >> 2);

	const int16_t inoff16 = input_offset;
	int32_t offset_q15x2 = PKHBT(inoff16, inoff16, 16);

	for (i_element = 0; i_element < num_elements / 2; i_element++) {
		/* Fill buffer for partial im2col - two columns at a time */
		int8_t *src = &input[i_element * input_ch * 2];
		int16_t *dst = two_column_buffer;

		//use variables
		int32_t in_q7x4;
		int32_t in_q15x2_1;
		int32_t in_q15x2_2;
		int32_t out_q15x2_1;
		int32_t out_q15x2_2;

		int cnt = channel_div4;	//two columns
		while (cnt > 0) {
			q7_q15_offset_reordered_ele(src, dst)
			q7_q15_offset_reordered_ele(src, dst)
			cnt--;
		}

		out = mat_mult_kernel_s8_s16_reordered_ch16(kernel,
				two_column_buffer, output_ch, output_shift, output_mult,
				(int8_t) out_offset, out_activation_min,
				out_activation_max, input_ch * DIM_KER_Y * DIM_KER_X,
				bias, out);
	}

	/* check if there is an odd column left-over for computation */
	if (num_elements & 0x1) {
		int32_t i_ch_out;
		const int8_t *ker_a = kernel;
		int8_t *src = &input[(num_elements - 1) * input_ch];
		int16_t *dst = two_column_buffer;

		//use variables
		int32_t in_q7x4;
		int32_t in_q15x2_1;
		int32_t in_q15x2_2;
		int32_t out_q15x2_1;
		int32_t out_q15x2_2;

		int cnt = channel_div4;	//two * numof2col columns
		while (cnt > 0) {
			q7_q15_offset_reordered_ele(src, dst)
			cnt--;
		}

		for (i_ch_out = 0; i_ch_out < output_ch; i_ch_out++) {
			int32_t sum = bias[i_ch_out];

			/* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
			const int16_t *ip_as_col = runtime_buf;
			uint16_t col_count = (input_ch * DIM_KER_X * DIM_KER_Y) >> 2;

			while (col_count) {
				int32_t ker_a1, ker_a2;
				int32_t in_b1, in_b2;
				ker_a = read_and_pad_reordered(ker_a, &ker_a1, &ker_a2);

				in_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = SMLAD(ker_a1, in_b1, sum);
				in_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = SMLAD(ker_a2, in_b2, sum);

				col_count--;
			}

			sum = arm_nn_requantize(sum, output_mult[i_ch_out],
					output_shift[i_ch_out]);
			sum += out_offset;
			sum = MAX(sum, out_activation_min);
			sum = MIN(sum, out_activation_max);
			*out++ = (int8_t) sum;
		}
	}

	/* Return to application */
	return STATE_SUCCESS;
}
