/*
 * SPDX-FileCopyrightText: Copyright 2010-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_convolve_wrapper_s8.c
 * Description:  s8 convolution layer wrapper function with the main purpose to call the optimal kernel available in
 * cmsis-nn to perform the convolution.
 *
 * $Date:        8 March 2023
 * $Revision:    V.2.4.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

/*
 * Convolution layer
 *
 * Refer header file for details.
 *
 */

arm_cmsis_nn_status arm_convolve_wrapper_s8(const cmsis_nn_context *ctx,
                                            const cmsis_nn_conv_params *conv_params,
                                            const cmsis_nn_per_channel_quant_params *quant_params,
                                            const cmsis_nn_dims *input_dims,
                                            const int8_t *const *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const int8_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const int32_t *bias_data,
                                            const cmsis_nn_dims *output_dims,
                                            int8_t *output_data,
                                            const int32_t *contribs)
{
    if ((conv_params->dilation.w == 1 && conv_params->dilation.h == 1) &&
        (conv_params->padding.w == 0) &&
        (conv_params->padding.h == 0) && 
        (filter_dims->w == 1) && 
        (filter_dims->h == 1))
    {
        if ((conv_params->stride.w == 1) && (conv_params->stride.h == 1))
        {
            switch (input_dims->c ){
                case 8:
                    convolve_1x1_s8_ch8(
                            input_data[0], input_dims->w, input_dims->h, input_dims->c,
                            filter_data, bias_data, quant_params->shift, quant_params->multiplier, conv_params->output_offset,
                            conv_params->input_offset, conv_params->activation.min, conv_params->activation.max,
                            output_data, output_dims->w, output_dims->h, output_dims->c, ctx->buf);
                    break;
                case 16:
                    convolve_1x1_s8_ch16(
                            input_data[0], input_dims->w, input_dims->h, input_dims->c,
                            filter_data, bias_data, quant_params->shift, quant_params->multiplier, conv_params->output_offset,
                            conv_params->input_offset, conv_params->activation.min, conv_params->activation.max,
                            output_data, output_dims->w, output_dims->h, output_dims->c, ctx->buf);
                        break;
                case 24:
                    convolve_1x1_s8_ch24(
                            input_data[0], input_dims->w, input_dims->h, input_dims->c,
                            filter_data, bias_data, quant_params->shift, quant_params->multiplier, conv_params->output_offset,
                            conv_params->input_offset, conv_params->activation.min, conv_params->activation.max,
                            output_data, output_dims->w, output_dims->h, output_dims->c, ctx->buf);
                        break;
                case 48:
                    convolve_1x1_s8_ch48(
                            input_data[0], input_dims->w, input_dims->h, input_dims->c,
                            filter_data, bias_data, quant_params->shift, quant_params->multiplier, conv_params->output_offset,
                            conv_params->input_offset, conv_params->activation.min, conv_params->activation.max,
                            output_data, output_dims->w, output_dims->h, output_dims->c, ctx->buf);
                        break;
                default:
                    // convolve_1x1_s8(
                    //         input_data[0], input_dims->w, input_dims->h, input_dims->c,
                    //         filter_data, bias_data, quant_params->shift, quant_params->multiplier, conv_params->output_offset,
                    //         conv_params->input_offset, conv_params->activation.min, conv_params->activation.max,
                    //         output_data, output_dims->w, output_dims->h, output_dims->c, ctx->buf);
                    arm_convolve_1x1_s8_fast_contribs(ctx,
                                                conv_params,
                                                quant_params,
                                                input_dims,
                                                input_data[0],
                                                filter_dims,
                                                filter_data,
                                                bias_dims,
                                                bias_data,
                                                output_dims,
                                                output_data,
                                                contribs
                                                );
            }
            return ARM_CMSIS_NN_SUCCESS;
        }
        else
        {
            return arm_convolve_1x1_s8_contribs(ctx,
                                        conv_params,
                                        quant_params,
                                        input_dims,
                                        input_data,
                                        filter_dims,
                                        filter_data,
                                        bias_dims,
                                        bias_data,
                                        output_dims,
                                        output_data,
                                        contribs
                                        );
        }
    }
    else if ((input_dims->h == 1) && 
             (conv_params->dilation.w == 1) && 
             (filter_dims->h == 1) &&
             ((conv_params->stride.w * input_dims->c) % 4 == 0))
    {
        return arm_convolve_1_x_n_s8(ctx,
                                     conv_params,
                                     quant_params,
                                     input_dims,
                                     input_data,
                                     filter_dims,
                                     filter_data,
                                     bias_dims,
                                     bias_data,
                                     output_dims,
                                     output_data);
    }
    else
    {
        return arm_convolve_s8(ctx,
                               conv_params,
                               quant_params,
                               input_dims,
                               input_data,
                               filter_dims,
                               filter_data,
                               bias_dims,
                               bias_data,
                               output_dims,
                               output_data);
    }
}

arm_cmsis_nn_status arm_convolve_wrapper_s8_ori(const cmsis_nn_context *ctx,
                                            const cmsis_nn_conv_params *conv_params,
                                            const cmsis_nn_per_channel_quant_params *quant_params,
                                            const cmsis_nn_dims *input_dims,
                                            const int8_t *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const int8_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const int32_t *bias_data,
                                            const cmsis_nn_dims *output_dims,
                                            int8_t *output_data)
{
    if ((conv_params->padding.w == 0) && (conv_params->padding.h == 0) && (filter_dims->w == 1) &&
        (filter_dims->h == 1) && (conv_params->dilation.w == 1 && conv_params->dilation.h == 1))
    {
        if ((conv_params->stride.w == 1) && (conv_params->stride.h == 1))
        {
            return arm_convolve_1x1_s8_fast(ctx,
                                            conv_params,
                                            quant_params,
                                            input_dims,
                                            input_data,
                                            filter_dims,
                                            filter_data,
                                            bias_dims,
                                            bias_data,
                                            output_dims,
                                            output_data);
        }
        else
        {
            return arm_convolve_1x1_s8(ctx,
                                       conv_params,
                                       quant_params,
                                       input_dims,
                                       input_data,
                                       filter_dims,
                                       filter_data,
                                       bias_dims,
                                       bias_data,
                                       output_dims,
                                       output_data);
        }
    }
    else if ((input_dims->h == 1) && conv_params->dilation.w == 1 && (filter_dims->h == 1) &&
             ((conv_params->stride.w * input_dims->c) % 4 == 0))
    {
        return arm_convolve_1_x_n_s8(ctx,
                                     conv_params,
                                     quant_params,
                                     input_dims,
                                     input_data,
                                     filter_dims,
                                     filter_data,
                                     bias_dims,
                                     bias_data,
                                     output_dims,
                                     output_data);
    }
    else
    {
        return arm_convolve_s8(ctx,
                               conv_params,
                               quant_params,
                               input_dims,
                               input_data,
                               filter_dims,
                               filter_data,
                               bias_dims,
                               bias_data,
                               output_dims,
                               output_data);
    }
}

/**
 * @} end of NNConv group
 */
