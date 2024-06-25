void conv_2d_tiny_3x3_ich3_st2_pad1(int shared_param_idx, int output_split_id, int scratch_buffer_offset) {
  #if HIDE_CONV
    return;
  #endif
  if (output_split_id == -1) {
    // unsplitted tensor
    conv_2d(shared_param_idx, scratch_buffer_offset);
    return;
  }
  const SharedParam_Conv* params = &shared_param_conv[shared_param_idx];
  const Tensor& input = tensors[params->input];
  const Tensor& filter = tensors[params->filter];
  const Tensor& output= tensors[params->output];

  // Get shape
  const DIM_TYPE* whole_input_shape = input.dims;
  const DIM_TYPE* filter_shape = filter.dims;
  const DIM_TYPE* output_shape = output.dims;

  int input_width = whole_input_shape[2];
  int input_height = whole_input_shape[1];
  int filter_width = filter_shape[2];
  int filter_height = filter_shape[1];
  int output_width = output_shape[2];
  int output_height = (output_split_id>=(output_shape[1]/SPLIT_HEIGHT))? (output_shape[1]%SPLIT_HEIGHT):(SPLIT_HEIGHT);
  OpData data;

  #if !OPT_OFFLOAD_ENABLE
  // v0.1 method: call CalculateOpData
  // Calculate padding and quantization params
  CalculateOpData(params, input_width, input_height, filter_width,
                  filter_height, output_width, output_shape[1], &data);
  #else
  // v0.2 method: use offloaded OpData
  const int32_t *op_data_buffer = GetOpData(params->op_data_offset);
  data.padding.height = op_data_buffer[0];
  data.padding.width = op_data_buffer[1];
  data.output_activation_min = op_data_buffer[2];
  data.output_activation_max = op_data_buffer[3];
  // Method  I: directly point to data in flash
  data.per_channel_output_multiplier = const_cast<int32_t*>(&op_data_buffer[4]);
  data.per_channel_output_shift = const_cast<int32_t*>(&op_data_buffer[4+output_shape[3]]);
  data.contribs = const_cast<int32_t*>(&op_data_buffer[4+output_shape[3]*2]);
  // // Method II: copy to SRAM
  // for (int i = 0; i < output_depth; i++)
  //   data.per_channel_output_multiplier[i] = op_data_buffer[4+i];
  // for (int i = 0; i < output_depth; i++)
  //   data.per_channel_output_shift[i] = op_data_buffer[4+output_depth+i];
  #endif

  // Get data

  // calculate the receptive field base on output_split_id
  // calculate the needed input split range

  uint16_t pad_yt = 0, pad_yb = 0;
  int out_split_height = SPLIT_HEIGHT;
  if ((output_split_id+1)*SPLIT_HEIGHT>output_shape[1]){
    out_split_height = output_shape[1] % SPLIT_HEIGHT;
  }
  
  int input_row_begin = (output_split_id*SPLIT_HEIGHT) * params->stride_height - data.padding.height;
  int input_row_end = input_row_begin + (filter_height-1) + (params->stride_height*(out_split_height-1));
  pad_yt = input_row_begin >= 0 ? 0:abs(input_row_begin);
  pad_yb = input_row_end <= input_height-1 ? 0:input_row_end-input_height + 1;
  input_row_end = input_row_end < input_height? input_row_end:input_height-1;
  input_row_begin = input_row_begin > 0 ? input_row_begin: 0;
  const VirtualFp<int8_t> combined_ifmap(params->input, input, input_row_begin, input_row_end, input_width*whole_input_shape[3]);
  int combined_height = input_row_end-input_row_begin+1;

  // int input_split_id_begin = (output_split_id*SPLIT_HEIGHT) * params->stride_height - data.padding.height;
  // int input_split_id_end = input_split_id_begin + (filter_height-1) + (params->stride_height*(out_split_height-1));
  // pad_yb = input_split_id_end <= total_height-1 ? 0:input_split_id_end-total_height + 1;
  // pad_yt = input_split_id_begin >= 0 ? 0:abs(input_split_id_begin);
  // input_split_id_end = input_split_id_end < total_height? input_split_id_end:total_height-1;
  // input_split_id_begin = input_split_id_begin > 0 ? input_split_id_begin: 0;
  // const VirtualFp<int8_t> combined_ifmap(params->tid_input, input, input_split_id_begin, input_split_id_end, total_width*input.dims[3]);

  // Set input_shape to the ifmap which is going to go through conv2d calculation
  const int* input_shape = combined_ifmap.combined_dim;

  const int8* filter_data = GetOfflineTensorData(params->filter);
  const int32* bias_data = reinterpret_cast<const int32*>(GetOfflineTensorData(params->bias));
  int8* output_data = GetSplitData(params->output, output_split_id);

  // conv_2d_simd(params, &data, scratch_buffer_offset, combined_ifmap, filter_data, bias_data, output_data, output_height);
  {
    const int* input_shape = combined_ifmap.combined_dim;
    const DIM_TYPE* output_shape = tensors[params->output].dims;
    const DIM_TYPE* filter_shape = tensors[params->filter].dims;

    const int batch_size = input_shape[0]/* MatchingDim(input_shape, 0, output_shape, 0) */;
    const int input_depth = input_shape[3];/* MatchingDim(input_shape, 3, filter_shape, 3) */;
    const int output_depth = output_shape[3]/* MatchingDim(filter_shape, 0, output_shape, 3) */;

    int32_t out_offset = *GetTensorQuantZP(params->output);
    int32_t input_offset = -(*GetTensorQuantZP(params->input));

    int16_t* buf = reinterpret_cast<int16_t*>(arena+scratch_buffer_offset);

    // printf("In header input_offset %d\n", input_offset);

    arm_convolve_3x3_ich3_st2_padw1_flexiblepadh_s8_tiny(combined_ifmap.inputs_data, input_width, input_height, input_depth, filter_data,
        bias_data, data.per_channel_output_shift, data.per_channel_output_multiplier, out_offset,
        input_offset, data.output_activation_min, data.output_activation_max,
        output_data, output_width, output_height, output_depth, buf, pad_yt, pad_yb);
  }
}



