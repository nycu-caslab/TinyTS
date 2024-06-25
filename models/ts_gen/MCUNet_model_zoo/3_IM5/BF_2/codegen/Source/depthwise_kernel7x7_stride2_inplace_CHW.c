/* This file is automatically generated */
/* ----------------------------------------------------------------------
 * Project:      TinyEngine
 * Description:  for sparse in-place 7x7 depth-wise convolution (HWC->CHW->HWC)
 * Target ISA:  ARMv7E-M
 * Author: wmchen@mit.edu
 * -------------------------------------------------------------------- */
#include "arm_nnsupportfunctions.h" //TODO: remove this in the future for self-contained
#include "tinyengine_function.h"
static void depthwise_kernel7x7_stride2_inplace_kernel_CHW(
        const uint16_t output_y, const uint16_t output_x,
        const int32_t *bias, const int32_t *offsetBias, const int8_t *ksrc, const int32_t *multiplier,
        const int32_t *shift, int8_t *output, const int32_t output_offset,
        const int32_t activation_min, const int32_t activation_max,
        int8_t *cols_8b_iterptr, const uint16_t column_x, int channel_offset, int pad_h, int pad_w);
tinyengine_status depthwise_kernel7x7_stride2_inplace_CHW(int8_t * const *input, const uint16_t input_x, const uint16_t input_y,
                const uint16_t input_ch, const int8_t *kernel, const int32_t *bias, const int32_t *offsetBias,
                const int32_t *output_shift, const int32_t *output_mult,
                const int32_t output_offset, const int32_t input_offset,
                const int32_t output_activation_min,
                const int32_t output_activation_max, int8_t *output,
                const uint16_t output_x, const uint16_t output_y,
                const uint16_t output_ch, int16_t *runtime_buf, int8_t pad_value, int pad_h, int pad_w, uint16_t pad_yb)
{

    // runtime padding on dim y
    int input_y_runtime_padded = input_y;
    int runtime_pad_yb = 0;
    int receptive_field_y = (output_y-1)*2/* stride_y */ + 3/* filter_y */;
    if(pad_w==0 && receptive_field_y > input_y){
        runtime_pad_yb = receptive_field_y - input_y;
        input_y_runtime_padded = receptive_field_y;
    }
    // runtime padding on dim x
    int input_x_runtime_padded = input_x;
    int runtime_pad_xr = 0;
    int receptive_field_x = (output_x-1)*2/* stride_x */ + 3/* filter_x */;
    if(pad_w==0 && receptive_field_x > input_x){
        runtime_pad_xr = receptive_field_x - input_x;
        input_x_runtime_padded = receptive_field_x;
    }

    uint16_t c,i,j;
    int8_t *cols_8b_start = (int8_t *)runtime_buf;
    int8_t* cols_8b = (int8_t* )cols_8b_start;

    //Set padding value
    int8_t PAD8 = pad_value;
    /* setup the padding regions for Im2col buffers */
    //top region: 8bit x (input_x + pad_w * 2) x pad_h: unroll by pad value
    for(i = 0; i < (input_x_runtime_padded + pad_w * 2) * pad_h; ++i){
        *cols_8b++ = PAD8;
    }

    //middle regions: left and right regions
    for(i = 0; i < input_y; i++){
        for (int p = 0; p < pad_w; ++p)
            *cols_8b++ = PAD8;//left
        cols_8b += input_x; //skip middle
        for (int p = 0; p < runtime_pad_xr; ++p)
            *cols_8b++ = PAD8;     //runtime padding right
        for (int p = 0; p < pad_w; ++p)
            *cols_8b++ = PAD8;  //right
    }

    //bottom region: 8bit x (input_x + pad_w * 2) x pad_h: unroll by pad value
    for(i = 0; i < (input_x_runtime_padded + pad_w * 2) * pad_yb; ++i){
        *cols_8b++ = PAD8;
    }

    //runtime padding
    for(i = 0; i < (input_x_runtime_padded + pad_w * 2) * runtime_pad_yb; ++i){
        *cols_8b++ = PAD8;
    }

    const int8_t *src;
    const int8_t *ksrc = kernel;

    for (c = 0; c < input_ch; c++){
        cols_8b = (int8_t*)(cols_8b_start +  pad_h  * (input_x_runtime_padded) +  pad_h * pad_w * 2); //skip  pad_h  rows
        for(i = 0; i < input_y; i++){
            src = input[i] + c;
            cols_8b +=  pad_w ;//skip front
            for(j = 0; j < input_x; j++){
                *cols_8b++ = *src;// + input_offset;
                src += input_ch;
            }
            cols_8b += pad_w + runtime_pad_xr;//skip end
        }
        int8_t *inplace_out = output + c;
        depthwise_kernel7x7_stride2_inplace_kernel_CHW(output_y, output_x, bias++, offsetBias++, ksrc, output_mult++, output_shift++, inplace_out,output_offset,output_activation_min, output_activation_max,cols_8b_start, input_x_runtime_padded, input_ch, pad_h, pad_w);
        ksrc += 49;

    }

}
static void depthwise_kernel7x7_stride2_inplace_kernel_CHW(
        const uint16_t output_y, const uint16_t output_x,
        const int32_t *bias, const int32_t *offsetBias, const int8_t *ksrc, const int32_t *multiplier,
        const int32_t *shift, int8_t *output, const int32_t output_offset,
        const int32_t activation_min, const int32_t activation_max,
        int8_t *cols_8b_iterptr, const uint16_t column_x, int channel_offset, int pad_h, int pad_w)
{
    #define STRIDE 2
    int8_t *cols_8b_reset = cols_8b_iterptr;
    int i, j;
    /* MACs for each output */
    for (i = 0; i < output_y; i++) {
        for (j = 0; j < output_x / 2; j++) {
            int8_t *cols_8b = cols_8b_iterptr;

            int32_t sum0 = bias[0] + offsetBias[0];
            int32_t sum1 = bias[0] + offsetBias[0];
            
            /* computation */
            sum0 += cols_8b[0]*ksrc[0];
            sum1 += cols_8b[2]*ksrc[0];
            sum0 += cols_8b[1]*ksrc[1];
            sum1 += cols_8b[3]*ksrc[1];
            sum0 += cols_8b[2]*ksrc[2];
            sum1 += cols_8b[4]*ksrc[2];
            sum0 += cols_8b[3]*ksrc[3];
            sum1 += cols_8b[5]*ksrc[3];
            sum0 += cols_8b[4]*ksrc[4];
            sum1 += cols_8b[6]*ksrc[4];
            sum0 += cols_8b[5]*ksrc[5];
            sum1 += cols_8b[7]*ksrc[5];
            sum0 += cols_8b[6]*ksrc[6];
            sum1 += cols_8b[8]*ksrc[6];
            cols_8b += column_x + pad_w * 2;
            sum0 += cols_8b[0]*ksrc[7];
            sum1 += cols_8b[2]*ksrc[7];
            sum0 += cols_8b[1]*ksrc[8];
            sum1 += cols_8b[3]*ksrc[8];
            sum0 += cols_8b[2]*ksrc[9];
            sum1 += cols_8b[4]*ksrc[9];
            sum0 += cols_8b[3]*ksrc[10];
            sum1 += cols_8b[5]*ksrc[10];
            sum0 += cols_8b[4]*ksrc[11];
            sum1 += cols_8b[6]*ksrc[11];
            sum0 += cols_8b[5]*ksrc[12];
            sum1 += cols_8b[7]*ksrc[12];
            sum0 += cols_8b[6]*ksrc[13];
            sum1 += cols_8b[8]*ksrc[13];
            cols_8b += column_x + pad_w * 2;
            sum0 += cols_8b[0]*ksrc[14];
            sum1 += cols_8b[2]*ksrc[14];
            sum0 += cols_8b[1]*ksrc[15];
            sum1 += cols_8b[3]*ksrc[15];
            sum0 += cols_8b[2]*ksrc[16];
            sum1 += cols_8b[4]*ksrc[16];
            sum0 += cols_8b[3]*ksrc[17];
            sum1 += cols_8b[5]*ksrc[17];
            sum0 += cols_8b[4]*ksrc[18];
            sum1 += cols_8b[6]*ksrc[18];
            sum0 += cols_8b[5]*ksrc[19];
            sum1 += cols_8b[7]*ksrc[19];
            sum0 += cols_8b[6]*ksrc[20];
            sum1 += cols_8b[8]*ksrc[20];
            cols_8b += column_x + pad_w * 2;
            sum0 += cols_8b[0]*ksrc[21];
            sum1 += cols_8b[2]*ksrc[21];
            sum0 += cols_8b[1]*ksrc[22];
            sum1 += cols_8b[3]*ksrc[22];
            sum0 += cols_8b[2]*ksrc[23];
            sum1 += cols_8b[4]*ksrc[23];
            sum0 += cols_8b[3]*ksrc[24];
            sum1 += cols_8b[5]*ksrc[24];
            sum0 += cols_8b[4]*ksrc[25];
            sum1 += cols_8b[6]*ksrc[25];
            sum0 += cols_8b[5]*ksrc[26];
            sum1 += cols_8b[7]*ksrc[26];
            sum0 += cols_8b[6]*ksrc[27];
            sum1 += cols_8b[8]*ksrc[27];
            cols_8b += column_x + pad_w * 2;
            sum0 += cols_8b[0]*ksrc[28];
            sum1 += cols_8b[2]*ksrc[28];
            sum0 += cols_8b[1]*ksrc[29];
            sum1 += cols_8b[3]*ksrc[29];
            sum0 += cols_8b[2]*ksrc[30];
            sum1 += cols_8b[4]*ksrc[30];
            sum0 += cols_8b[3]*ksrc[31];
            sum1 += cols_8b[5]*ksrc[31];
            sum0 += cols_8b[4]*ksrc[32];
            sum1 += cols_8b[6]*ksrc[32];
            sum0 += cols_8b[5]*ksrc[33];
            sum1 += cols_8b[7]*ksrc[33];
            sum0 += cols_8b[6]*ksrc[34];
            sum1 += cols_8b[8]*ksrc[34];
            cols_8b += column_x + pad_w * 2;
            sum0 += cols_8b[0]*ksrc[35];
            sum1 += cols_8b[2]*ksrc[35];
            sum0 += cols_8b[1]*ksrc[36];
            sum1 += cols_8b[3]*ksrc[36];
            sum0 += cols_8b[2]*ksrc[37];
            sum1 += cols_8b[4]*ksrc[37];
            sum0 += cols_8b[3]*ksrc[38];
            sum1 += cols_8b[5]*ksrc[38];
            sum0 += cols_8b[4]*ksrc[39];
            sum1 += cols_8b[6]*ksrc[39];
            sum0 += cols_8b[5]*ksrc[40];
            sum1 += cols_8b[7]*ksrc[40];
            sum0 += cols_8b[6]*ksrc[41];
            sum1 += cols_8b[8]*ksrc[41];
            cols_8b += column_x + pad_w * 2;
            sum0 += cols_8b[0]*ksrc[42];
            sum1 += cols_8b[2]*ksrc[42];
            sum0 += cols_8b[1]*ksrc[43];
            sum1 += cols_8b[3]*ksrc[43];
            sum0 += cols_8b[2]*ksrc[44];
            sum1 += cols_8b[4]*ksrc[44];
            sum0 += cols_8b[3]*ksrc[45];
            sum1 += cols_8b[5]*ksrc[45];
            sum0 += cols_8b[4]*ksrc[46];
            sum1 += cols_8b[6]*ksrc[46];
            sum0 += cols_8b[5]*ksrc[47];
            sum1 += cols_8b[7]*ksrc[47];
            sum0 += cols_8b[6]*ksrc[48];
            sum1 += cols_8b[8]*ksrc[48];

            /* requantize */
            sum0 = arm_nn_requantize(sum0, *multiplier, *shift);
            sum0 += output_offset;
            sum0 = MAX(sum0, activation_min);
            sum0 = MIN(sum0, activation_max);
            output[(i * output_x + j * 2) * channel_offset] = sum0;

            sum1 = arm_nn_requantize(sum1, *multiplier, *shift);
            sum1 += output_offset;
            sum1 = MAX(sum1, activation_min);
            sum1 = MIN(sum1, activation_max);
            output[(i * output_x + (j * 2 + 1)) * channel_offset] = sum1;

            cols_8b_iterptr += STRIDE * 2;
        }
        if (output_x & 1) {
            int8_t * cols_8b = cols_8b_iterptr;
            int32_t sum = bias[0] + offsetBias[0];
            sum += cols_8b[0]*ksrc[0];
            sum += cols_8b[1]*ksrc[1];
            sum += cols_8b[2]*ksrc[2];
            sum += cols_8b[3]*ksrc[3];
            sum += cols_8b[4]*ksrc[4];
            sum += cols_8b[5]*ksrc[5];
            sum += cols_8b[6]*ksrc[6];
            cols_8b += column_x + pad_w * 2;
            sum += cols_8b[0]*ksrc[7];
            sum += cols_8b[1]*ksrc[8];
            sum += cols_8b[2]*ksrc[9];
            sum += cols_8b[3]*ksrc[10];
            sum += cols_8b[4]*ksrc[11];
            sum += cols_8b[5]*ksrc[12];
            sum += cols_8b[6]*ksrc[13];
            cols_8b += column_x + pad_w * 2;
            sum += cols_8b[0]*ksrc[14];
            sum += cols_8b[1]*ksrc[15];
            sum += cols_8b[2]*ksrc[16];
            sum += cols_8b[3]*ksrc[17];
            sum += cols_8b[4]*ksrc[18];
            sum += cols_8b[5]*ksrc[19];
            sum += cols_8b[6]*ksrc[20];
            cols_8b += column_x + pad_w * 2;
            sum += cols_8b[0]*ksrc[21];
            sum += cols_8b[1]*ksrc[22];
            sum += cols_8b[2]*ksrc[23];
            sum += cols_8b[3]*ksrc[24];
            sum += cols_8b[4]*ksrc[25];
            sum += cols_8b[5]*ksrc[26];
            sum += cols_8b[6]*ksrc[27];
            cols_8b += column_x + pad_w * 2;
            sum += cols_8b[0]*ksrc[28];
            sum += cols_8b[1]*ksrc[29];
            sum += cols_8b[2]*ksrc[30];
            sum += cols_8b[3]*ksrc[31];
            sum += cols_8b[4]*ksrc[32];
            sum += cols_8b[5]*ksrc[33];
            sum += cols_8b[6]*ksrc[34];
            cols_8b += column_x + pad_w * 2;
            sum += cols_8b[0]*ksrc[35];
            sum += cols_8b[1]*ksrc[36];
            sum += cols_8b[2]*ksrc[37];
            sum += cols_8b[3]*ksrc[38];
            sum += cols_8b[4]*ksrc[39];
            sum += cols_8b[5]*ksrc[40];
            sum += cols_8b[6]*ksrc[41];
            cols_8b += column_x + pad_w * 2;
            sum += cols_8b[0]*ksrc[42];
            sum += cols_8b[1]*ksrc[43];
            sum += cols_8b[2]*ksrc[44];
            sum += cols_8b[3]*ksrc[45];
            sum += cols_8b[4]*ksrc[46];
            sum += cols_8b[5]*ksrc[47];
            sum += cols_8b[6]*ksrc[48];

            sum = arm_nn_requantize(sum, *multiplier, *shift);
            sum += output_offset;
            sum = MAX(sum, activation_min);
            sum = MIN(sum, activation_max);
            output[(i * output_x + output_x - 1) * channel_offset] = sum;

            cols_8b_iterptr += STRIDE;
        }
        cols_8b_iterptr = cols_8b_reset + (column_x + pad_w * 2)*STRIDE;
        cols_8b_reset = cols_8b_iterptr;
    }
}
