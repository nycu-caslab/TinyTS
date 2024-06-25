/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   convolve_s8_kernel3_inputch3_stride2_pad1.c
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
#include "arm_nnsupportfunctions.h"
#include "img2col_element.h"
#include "tinyengine_function.h"


tinyengine_status convolve_s8_kernel3_inputch3_stride2_padw1_flexiblepadh(const int8_t *const *input, const uint16_t input_x, const uint16_t input_y,
		const uint16_t input_ch, const int8_t *kernel, const int32_t *bias,
		const int32_t *output_shift, const int32_t *output_mult,
		const int32_t output_offset, const int32_t input_offset,
		const int32_t output_activation_min,
		const int32_t output_activation_max, int8_t *output,
		const uint16_t output_x, const uint16_t output_y,
		const uint16_t output_ch, int16_t *runtime_buf, int16_t *kbuf, int8_t pad_value, uint16_t pad_yt, uint16_t pad_yb) {
	const int kernel_y = 3;
	const int kernel_x = 3;

	int16_t i_out_y, i_out_x, i_ker_y, i_ker_x;

	/* Generate two columns from the input tensor a GEMM computation */
	int16_t *two_column_buf = runtime_buf;
	int8_t *out = output;

	int16_t pad16 = pad_value;
	const int16_t inoff16 = input_offset;
	int16_t pad_out = pad16 + inoff16;
	int32_t pad_out_q15x2 = __PKHBT(pad_out, pad_out, 16);
	int32_t offset_q15x2 = __PKHBT(inoff16, inoff16, 16);
	// printf("pad_value %d, inoff16 %d, pad_out %d\n", pad_value, inoff16, pad_out);

	const int8_t *ip_a0 = kernel;

	// printf("output_ch %d, runtime_buf %d, kbuf%d\n", output_ch, runtime_buf, kbuf);
	for (int i = 0; i < output_ch; i += 2) {
		int16_t *dst1 = &kbuf[i * 27]; //each int32_t store 2 elements
		int16_t *dst2 = dst1 + 27;

		const int8_t *ip_a1 = ip_a0 + 27;

		//27 for each output_ch
		int32_t *dst1_31 = dst1;
		int32_t *dst2_31 = dst2;
		ip_a0 = read_and_pad(ip_a0, &dst1_31[0], &dst1_31[1]);
		ip_a1 = read_and_pad(ip_a1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		ip_a0 = read_and_pad(ip_a0, &dst1_31[0], &dst1_31[1]);
		ip_a1 = read_and_pad(ip_a1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		ip_a0 = read_and_pad(ip_a0, &dst1_31[0], &dst1_31[1]);
		ip_a1 = read_and_pad(ip_a1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		ip_a0 = read_and_pad(ip_a0, &dst1_31[0], &dst1_31[1]);
		ip_a1 = read_and_pad(ip_a1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		ip_a0 = read_and_pad(ip_a0, &dst1_31[0], &dst1_31[1]);
		ip_a1 = read_and_pad(ip_a1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		ip_a0 = read_and_pad(ip_a0, &dst1_31[0], &dst1_31[1]);
		ip_a1 = read_and_pad(ip_a1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;
		//25, 26, 27
		dst1 = dst1_31;
		dst2 = dst2_31;
		dst1[0] = *ip_a0++;
		dst1[1] = *ip_a0++;
		dst1[2] = *ip_a0++;
		dst2[0] = *ip_a1++;
		dst2[1] = *ip_a1++;
		dst2[2] = *ip_a1++;

		/* skip row */
		ip_a0 += 27;
	}

	for (i_out_y = 0; i_out_y < output_y; i_out_y++) {
		for (i_out_x = 0; i_out_x < output_x; i_out_x++) {
			/* This part implements the im2col function */
			const int16_t base_idx_y = (i_out_y * 2) - pad_yt;
			const int16_t base_idx_x = (i_out_x * 2) - 1;
			const int16_t *col_buffer = two_column_buf;

			//use variables
			int32_t in_q7x4;
			int32_t in_q15x2_1;
			int32_t in_q15x2_2;
			int32_t out_q15x2_1;
			int32_t out_q15x2_2;

			/* load address:8bit */
			int8_t *src;
			int8_t *src2;
			int8_t *src3;

			/* buffer for load:16bit */
			int16_t *dst;
			int16_t *dst2;
			int16_t *dst3;

			int input_row_offset = 3 * input_x;
			dst = col_buffer;
			dst2 = dst + 9;
			dst3 = dst2 + 9;
			if (base_idx_y != -1) {
                //center
				if (base_idx_x != -1) { //load all for now and unroll all
					//3x3 = 9 elements
					src  = input[base_idx_y] + (base_idx_x) * 3;
					src2 = input[base_idx_y+1] + (base_idx_x) * 3;
					src3 = input[base_idx_y+2] + (base_idx_x) * 3;

					//4 * 2 = 8
					q7_q15_offset_ele(src, dst)
					q7_q15_offset_ele(src, dst)
					*dst++ = *src++ + input_offset;
					//
					q7_q15_offset_ele(src2, dst2)
					q7_q15_offset_ele(src2, dst2)
					*dst2++ = *src2++ + input_offset;
					//
					q7_q15_offset_ele(src3, dst3)
					q7_q15_offset_ele(src3, dst3)
					*dst3++ = *src3++ + input_offset;
				} 
                //left
                else {						//first element is pad
												//3x3 = 9 elements
					src  = input[base_idx_y];
					src2 = input[base_idx_y+1];
					src3 = input[base_idx_y+2];

					//pad the first one: 1x3 = 3
					*dst++ = pad_out;
					*dst++ = pad_out;
					*dst++ = pad_out;
					*dst2++ = pad_out;
					*dst2++ = pad_out;
					*dst2++ = pad_out;
					*dst3++ = pad_out;
					*dst3++ = pad_out;
					*dst3++ = pad_out;
					//load 6 elements
					//4 * 1 = 6
					q7_q15_offset_ele(src, dst)
					*dst++ = *src++ + input_offset;
					*dst++ = *src++ + input_offset;
					//
					q7_q15_offset_ele(src2, dst2)
					*dst2++ = *src2++ + input_offset;
					*dst2++ = *src2++ + input_offset;
					//
					q7_q15_offset_ele(src3, dst3)
					*dst3++ = *src3++ + input_offset;
					*dst3++ = *src3++ + input_offset;
				}
			}
            //top part
            else {						// first row is padded
											//3x3 = 9 elements
				*dst++ = pad_out;
				int32_t *dst_31 = dst;
				*dst_31++ = pad_out_q15x2;
				*dst_31++ = pad_out_q15x2;
				*dst_31++ = pad_out_q15x2;
				*dst_31++ = pad_out_q15x2;
                //top center
				if (base_idx_x != -1) {	//load all for now and unroll all
					//3x3 = 9 elements
					src2 = input[0] + (base_idx_x) * 3;
					src3 = input[1] + (base_idx_x) * 3;

					//4 * 2 = 8
					q7_q15_offset_ele(src2, dst2)
					q7_q15_offset_ele(src2, dst2)
					*dst2++ = *src2++ + input_offset;
					//
					q7_q15_offset_ele(src3, dst3)
					q7_q15_offset_ele(src3, dst3)
					*dst3++ = *src3++ + input_offset;
				} 
                //top left
                else {						//first element is pad
												//3x3 = 9 elements
					src2 = input[0];
					src3 = input[1];

					//pad the first one: 1x3 = 3
					*dst2++ = pad_out;
					*dst2++ = pad_out;
					*dst2++ = pad_out;
					*dst3++ = pad_out;
					*dst3++ = pad_out;
					*dst3++ = pad_out;
					//load 6 elements
					q7_q15_offset_ele(src2, dst2)
					*dst2++ = *src2++ + input_offset;
					*dst2++ = *src2++ + input_offset;
					//
					q7_q15_offset_ele(src3, dst3)
					*dst3++ = *src3++ + input_offset;
					*dst3++ = *src3++ + input_offset;
				}
			}

			two_column_buf += 27;
			/* Computation is filed for every 2 columns */
			if (two_column_buf == runtime_buf + 2 * 27) {
                // printf("data in buf:\n");
				// for(int x=0; x<54;x++){
				// 	printf("%d ", runtime_buf[x]);
				// 	if (x==26) printf("\n");
				// }
				// printf("\n\n");
				out = arm_nn_mat_mult_kernel3_input3_s8_s16(kernel,
						runtime_buf, output_ch, output_shift, output_mult,
						output_offset, output_activation_min, output_activation_max,
						3 * 3 * 3, bias, out, kbuf);
                // printf("data in out:\n");
                // for(int x=0; x<16;x++){
				// 	printf("%d ", output[x]);
				// }printf("\n");
                // int a;
                // scanf("%d", &a);

				/* counter reset */
				two_column_buf = runtime_buf;
			}
		}
	}

	/* left-over because odd number of output pixels */
	if (two_column_buf != runtime_buf) {
		const int8_t *ker_a = kernel;
		int i;

		for (i = 0; i < output_ch; i++) {
			/* Load the accumulator with bias first */
			int32_t sum = bias[i];

			/* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
			const int16_t *ip_as_col = runtime_buf;

			/* 4 multiply and accumulates are done in one loop. */
			uint16_t col_count = (3 * 3 * 3) >> 2;

			while (col_count) {
				int32_t ker_a1, ker_a2;
				int32_t ip_b1, ip_b2;

				ker_a = read_and_pad(ker_a, &ker_a1, &ker_a2);

				ip_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = __SMLAD(ker_a1, ip_b1, sum);
				ip_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = __SMLAD(ker_a2, ip_b2, sum);

				col_count--;
			}
			/* Handle left over mac */
			col_count = 3 * 3 * 3 & 0x3;
			while (col_count) {
				int8_t ker_a1 = *ker_a++;
				int16_t ip_b1 = *ip_as_col++;
				sum += ker_a1 * ip_b1;
				col_count--;
			}

			sum = arm_nn_requantize(sum, output_mult[i], output_shift[i]);
			sum += output_offset;
			sum = MAX(sum, output_activation_min);
			sum = MIN(sum, output_activation_max);
			*out++ = (int8_t) sum;
		}
	}

	/* Return to application */
	return STATE_SUCCESS;
}
