tinyengine_status depthwise_kernel3x3_stride1_inplace_CHW(int8_t * const *input, const uint16_t input_x, const uint16_t input_y,
                const uint16_t input_ch, const int8_t *kernel, const int32_t *bias, const int32_t *offsetBias,
                const int32_t *output_shift, const int32_t *output_mult,
                const int32_t output_offset, const int32_t input_offset,
                const int32_t output_activation_min,
                const int32_t output_activation_max, int8_t *output,
                const uint16_t output_x, const uint16_t output_y,
                const uint16_t output_ch, int16_t *runtime_buf, int8_t pad_value, int pad_h, int pad_w, uint16_t pad_yb);
tinyengine_status depthwise_kernel3x3_stride2_inplace_CHW(int8_t * const *input, const uint16_t input_x, const uint16_t input_y,
                const uint16_t input_ch, const int8_t *kernel, const int32_t *bias, const int32_t *offsetBias,
                const int32_t *output_shift, const int32_t *output_mult,
                const int32_t output_offset, const int32_t input_offset,
                const int32_t output_activation_min,
                const int32_t output_activation_max, int8_t *output,
                const uint16_t output_x, const uint16_t output_y,
                const uint16_t output_ch, int16_t *runtime_buf, int8_t pad_value, int pad_h, int pad_w, uint16_t pad_yb);
