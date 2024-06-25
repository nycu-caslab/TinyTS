/*
 * Copyright (C) 2020 Arm Limited or its affiliates. All rights reserved.
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
 * Title:        arm_nn_mat_mult_s8_nt_t_s8
 * Description:  Matrix multiplication support function with the right-hand-side (rhs) matrix transposed
 *
 * $Date:        March 17 2020
 * $Revision:    V.1.0.1
 *
 * Target Processor:  Cortex-M
 *
 * -------------------------------------------------------------------- */

#include "cmsis/CMSIS/NN/Include/arm_nnsupportfunctions.h"


/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup NNBasicMath
 * @{
 */

/*
   * s8 matrix multiplication with the right-hand-side matrix transposed
   *
   * Refer header file for details.
   *
   */
arm_cmsis_nn_status arm_nn_mat_mult_nt_t_s8_contribs_ich16(const int8_t *lhs,
                                            const int8_t *rhs,
                                            const int32_t *bias,
                                            int8_t *dst,
                                            const int32_t *dst_multipliers,
                                            const int32_t *dst_shifts,
                                            const int32_t *contribs,
                                            const int32_t lhs_rows,
                                            const int32_t rhs_rows,
                                            const int32_t rhs_cols,
                                            const int32_t lhs_offset,
                                            const int32_t dst_offset,
                                            const int32_t activation_min,
                                            const int32_t activation_max,
                                            const int32_t lhs_cols_offset)
{
    const int32_t off0 = rhs_cols - 4;

    int32_t rhs_rows_idx = 0;
    for (; rhs_rows_idx <= (rhs_rows - 2); rhs_rows_idx += 2)
    {
        const int8_t *lhs_ptr = &lhs[0];
        int8_t       *dst_ptr = &dst[rhs_rows_idx];

        // get the precomputed contributions
        int32_t lhs_offset_contribution0 = contribs[rhs_rows_idx] + bias[rhs_rows_idx];
        int32_t lhs_offset_contribution1 = contribs[rhs_rows_idx + 1] + bias[rhs_rows_idx + 1];

        int32_t lhs_rows_idx = lhs_rows >> 1;

        while (lhs_rows_idx)
        {
            const int8_t *rhs_ptr = &rhs[0];

            int32_t res00 = lhs_offset_contribution0;
            int32_t res01 = lhs_offset_contribution1;
            int32_t res10 = lhs_offset_contribution0;
            int32_t res11 = lhs_offset_contribution1;

            int32_t rhs_cols_idx = 0;

            int32_t val0, val1, val2, val3, val4, val5;

            // mac start (16ch)
            val1 = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
            val2 = SXTB16(val1);
            val0 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
            val3 = SXTB16(val0);
            val4 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[off0]);
            val1 = SXTB16_RORn(val1, 8);
            val0 = SXTB16_RORn(val0, 8);

            // 4 x MAC res00, res01
            res00 = SMLAD(val3, val2, res00);
            val5  = SXTB16(val4);
            res00 = SMLAD(val0, val1, res00);
            val4  = SXTB16_RORn(val4, 8);
            res01 = SMLAD(val3, val5, res01);
            res01 = SMLAD(val0, val4, res01);

            // 4 x MAC res10, res11
            val0  = arm_nn_read_s8x4((const int8_t *)&lhs_ptr[off0]);
            val3  = SXTB16(val0);
            val0  = SXTB16_RORn(val0, 8);
            res10 = SMLAD(val3, val2, res10);
            res11 = SMLAD(val3, val5, res11);
            res10 = SMLAD(val0, val1, res10);
            val1  = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
            res11 = SMLAD(val0, val4, res11);

            val4 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[off0]);
            val2 = SXTB16(val1);
            val0 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
            val3 = SXTB16(val0);
            val1 = SXTB16_RORn(val1, 8);
            val0 = SXTB16_RORn(val0, 8);

            // 4 x MAC res00, res01
            res00 = SMLAD(val3, val2, res00);
            val5  = SXTB16(val4);
            res00 = SMLAD(val0, val1, res00);
            val4  = SXTB16_RORn(val4, 8);
            res01 = SMLAD(val3, val5, res01);
            res01 = SMLAD(val0, val4, res01);

            // 4 x MAC res10, res11
            val0  = arm_nn_read_s8x4((const int8_t *)&lhs_ptr[off0]);
            val3  = SXTB16(val0);
            val0  = SXTB16_RORn(val0, 8);
            res10 = SMLAD(val3, val2, res10);
            res11 = SMLAD(val3, val5, res11);
            res10 = SMLAD(val0, val1, res10);
            val1  = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
            res11 = SMLAD(val0, val4, res11);

            val4 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[off0]);
            val2 = SXTB16(val1);
            val0 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
            val3 = SXTB16(val0);
            val1 = SXTB16_RORn(val1, 8);
            val0 = SXTB16_RORn(val0, 8);

            // 4 x MAC res00, res01
            res00 = SMLAD(val3, val2, res00);
            val5  = SXTB16(val4);
            res00 = SMLAD(val0, val1, res00);
            val4  = SXTB16_RORn(val4, 8);
            res01 = SMLAD(val3, val5, res01);
            res01 = SMLAD(val0, val4, res01);

            // 4 x MAC res10, res11
            val0  = arm_nn_read_s8x4((const int8_t *)&lhs_ptr[off0]);
            val3  = SXTB16(val0);
            val0  = SXTB16_RORn(val0, 8);
            res10 = SMLAD(val3, val2, res10);
            res11 = SMLAD(val3, val5, res11);
            res10 = SMLAD(val0, val1, res10);
            val1  = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
            res11 = SMLAD(val0, val4, res11);

            val4 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[off0]);
            val2 = SXTB16(val1);
            val0 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
            val3 = SXTB16(val0);
            val1 = SXTB16_RORn(val1, 8);
            val0 = SXTB16_RORn(val0, 8);

            // 4 x MAC res00, res01
            res00 = SMLAD(val3, val2, res00);
            val5  = SXTB16(val4);
            res00 = SMLAD(val0, val1, res00);
            val4  = SXTB16_RORn(val4, 8);
            res01 = SMLAD(val3, val5, res01);
            res01 = SMLAD(val0, val4, res01);

            // 4 x MAC res10, res11
            val0  = arm_nn_read_s8x4((const int8_t *)&lhs_ptr[off0]);
            val3  = SXTB16(val0);
            val0  = SXTB16_RORn(val0, 8);
            res10 = SMLAD(val3, val2, res10);
            res11 = SMLAD(val3, val5, res11);
            res10 = SMLAD(val0, val1, res10);
            res11 = SMLAD(val0, val4, res11);
            // mac end

            // Quantize down
            res00 = arm_nn_requantize(res00, dst_multipliers[rhs_rows_idx],     dst_shifts[rhs_rows_idx]);
            res01 = arm_nn_requantize(res01, dst_multipliers[rhs_rows_idx + 1], dst_shifts[rhs_rows_idx + 1]);
            res10 = arm_nn_requantize(res10, dst_multipliers[rhs_rows_idx],     dst_shifts[rhs_rows_idx]);
            res11 = arm_nn_requantize(res11, dst_multipliers[rhs_rows_idx + 1], dst_shifts[rhs_rows_idx + 1]);

            // Add offset
            res00 += dst_offset;
            res01 += dst_offset;
            res10 += dst_offset;
            res11 += dst_offset;

            // Clamp the result
            res00 = CLAMP(res00, activation_max, activation_min);
            res01 = CLAMP(res01, activation_max, activation_min);
            res10 = CLAMP(res10, activation_max, activation_min);
            res11 = CLAMP(res11, activation_max, activation_min);

            dst_ptr[0] = (int8_t)res00;
            dst_ptr[1] = (int8_t)res01;
            dst_ptr += rhs_rows;
            dst_ptr[0] = (int8_t)res10;
            dst_ptr[1] = (int8_t)res11;
            dst_ptr += rhs_rows;

            lhs_ptr += rhs_cols;

            lhs_rows_idx--;
        }

        // Left-over rows
        if (lhs_rows % 2)
        {
            const int8_t *rhs_ptr = &rhs[0];

            int32_t res00 = lhs_offset_contribution0;
            int32_t res01 = lhs_offset_contribution1;

            int32_t rhs_cols_idx = 0;

            int32_t val0, val1, val2, val3, val4, val5;
            for (; rhs_cols_idx <= (rhs_cols - 16); rhs_cols_idx += 16)
            {
                val0 = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
                val1 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[off0]);
                val2 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
                val3 = SXTB16(val0);
                val5 = SXTB16(val2);
                val4 = SXTB16(val1);
                val0 = SXTB16_RORn(val0, 8);
                val2 = SXTB16_RORn(val2, 8);
                val1 = SXTB16_RORn(val1, 8);

                // 4 x MAC res00, res01
                res00 = SMLAD(val5, val3, res00);
                res00 = SMLAD(val2, val0, res00);
                res01 = SMLAD(val5, val4, res01);
                res01 = SMLAD(val2, val1, res01);

                val0 = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
                val1 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[off0]);
                val2 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
                val3 = SXTB16(val0);
                val5 = SXTB16(val2);
                val4 = SXTB16(val1);
                val0 = SXTB16_RORn(val0, 8);
                val2 = SXTB16_RORn(val2, 8);
                val1 = SXTB16_RORn(val1, 8);

                // 4 x MAC res00, res01
                res00 = SMLAD(val5, val3, res00);
                res00 = SMLAD(val2, val0, res00);
                res01 = SMLAD(val5, val4, res01);
                res01 = SMLAD(val2, val1, res01);

                val0 = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
                val1 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[off0]);
                val2 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
                val3 = SXTB16(val0);
                val5 = SXTB16(val2);
                val4 = SXTB16(val1);
                val0 = SXTB16_RORn(val0, 8);
                val2 = SXTB16_RORn(val2, 8);
                val1 = SXTB16_RORn(val1, 8);

                // 4 x MAC res00, res01
                res00 = SMLAD(val5, val3, res00);
                res00 = SMLAD(val2, val0, res00);
                res01 = SMLAD(val5, val4, res01);
                res01 = SMLAD(val2, val1, res01);

                val0 = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
                val1 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[off0]);
                val2 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
                val3 = SXTB16(val0);
                val5 = SXTB16(val2);
                val4 = SXTB16(val1);
                val0 = SXTB16_RORn(val0, 8);
                val2 = SXTB16_RORn(val2, 8);
                val1 = SXTB16_RORn(val1, 8);

                // 4 x MAC res00, res01
                res00 = SMLAD(val5, val3, res00);
                res00 = SMLAD(val2, val0, res00);
                res01 = SMLAD(val5, val4, res01);
                res01 = SMLAD(val2, val1, res01);
            }

            // Left-over accumulations

            while (unlikely(rhs_cols_idx < rhs_cols)) {
                int8_t rhs_value0 = rhs_ptr[0];
                int8_t rhs_value1 = rhs_ptr[rhs_cols];
                int8_t lhs_value  = lhs_ptr[0];

                res00 += lhs_value * rhs_value0;
                res01 += lhs_value * rhs_value1;

                ++rhs_ptr;
                ++lhs_ptr;
                ++rhs_cols_idx;
            }

            // Quantize down
            res00 = arm_nn_requantize(res00, dst_multipliers[rhs_rows_idx],     dst_shifts[rhs_rows_idx]);
            res01 = arm_nn_requantize(res01, dst_multipliers[rhs_rows_idx + 1], dst_shifts[rhs_rows_idx + 1]);

            // Add offset
            res00 += dst_offset;
            res01 += dst_offset;

            // Clamp the result
            res00 = CLAMP(res00, activation_max, activation_min);
            res01 = CLAMP(res01, activation_max, activation_min);

            dst_ptr[0] = (int8_t)res00;
            dst_ptr[1] = (int8_t)res01;
        }

        rhs += 2 * rhs_cols;
    }

    if (rhs_rows % 2)
    {
        const int8_t *lhs_ptr = &lhs[0];
        int8_t *dst_ptr = &dst[rhs_rows_idx + 1];

        for (int32_t lhs_rows_idx = 0; lhs_rows_idx < lhs_rows; ++lhs_rows_idx)
        {
            const int8_t *rhs_ptr = &rhs[0];
            int32_t res00 = bias[rhs_rows - 1];

            for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
            {
                int32_t rhs_value = rhs_ptr[0];
                int32_t lhs_value = lhs_ptr[0] + lhs_offset;

                res00 += lhs_value * rhs_value;

                ++rhs_ptr;
                ++lhs_ptr;
            }

            // Quantize down
            res00 = arm_nn_requantize(res00, dst_multipliers[rhs_rows - 1], dst_shifts[rhs_rows - 1]);

            // Add offset
            res00 += dst_offset;

            // Clamp the result
            res00 = CLAMP(res00, activation_max, activation_min);

            dst_ptr[0] = (int8_t)res00;
            dst_ptr += rhs_rows;
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNBasicMath group
 */
