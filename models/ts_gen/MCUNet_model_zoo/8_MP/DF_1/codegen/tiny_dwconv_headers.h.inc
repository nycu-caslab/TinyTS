void DWCONV_FN_NAME(int shared_param_idx, int scratch_buffer_offset) {
  const SharedParam_Depthwise_Conv* params = &shared_param_dwconv[shared_param_idx];

  const Tensor& output = tensors[params->tid_output];
  const Tensor& input = tensors[params->tid_input];
  const Tensor& filter = tensors[params->tid_filter];
  const Tensor& bias = tensors[params->tid_bias];

  // Assune data_type is int8
  int width = input.dims[2]/* SizeOfDimension(input, 2) */;
  int height = input.dims[1];
  int filter_width = filter.dims[2]/* SizeOfDimension(filter, 2) */;
  int filter_height = filter.dims[1]/* SizeOfDimension(filter, 1) */;

  OP_utils::DepthwiseConv::OpData data;
  const DIM_TYPE* filter_shape = filter.dims;
  const DIM_TYPE* input_shape = input.dims;
  const int input_height = input_shape[1];
  const int input_width = input_shape[2];
  const int input_depth = input_shape[3];
  const DIM_TYPE* output_shape = output.dims;
  const int output_height = output_shape[1];
  const int output_width = output_shape[2];

  #if !OPT_OFFLOAD_ENABLE
  // v0.1 method: call CalculateOpData
  // Calculate padding and quantization params
  CalculateOpData(params, input_width, input_height,
                  filter_width, filter_height,
                  output_shape[2], output_shape[1], &data);
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
  #endif

  const VirtualFp<int8_t> combined_ifmap(params->tid_input, input);
  const int8* filter_data = GetOfflineTensorData(params->tid_filter);
  const int32* bias_data= reinterpret_cast<const int32*>(GetOfflineTensorData(params->tid_bias));
  int8* output_data = GetTensorData(params->tid_output);
  int16_t* buf = reinterpret_cast<int16_t*>(arena+scratch_buffer_offset);

  uint16_t pad_yb = true;

  if (params->depth_multiplier == 1) {
    TINY_ENGINE_FN(
            combined_ifmap.inputs_data, input_width, input_height,
            input_depth, filter_data, input_depth,
            filter_width, filter_height, data.padding.width,
            data.padding.height, pad_yb, params->stride_width,
            params->stride_height, bias_data, data.contribs,
            output_data, data.per_channel_output_shift,
            data.per_channel_output_multiplier, output_width, output_height,
            *GetTensorQuantZP(params->tid_output)/* op_params.output_offset */,
            -(*GetTensorQuantZP(params->tid_input)) /* op_params.input_offset */,
            std::numeric_limits<int8_t>::min()/* op_params.quantized_activation_min */,
            std::numeric_limits<int8_t>::max()/* op_params.quantized_activation_max */,
            params->dilation_width_factor, params->dilation_height_factor, buf);//,
  } else {
    depthwise_conv_simd(params, &data, scratch_buffer_offset, combined_ifmap, filter_data, bias_data, output_data, output_height);
  }
}

void DWCONV_FN_NAME(int shared_param_idx, int output_split_id, int scratch_buffer_offset) {
  #if HIDE_DWCONV
    return;
  #endif
  const SharedParam_Depthwise_Conv* params = &shared_param_dwconv[shared_param_idx];

  const Tensor& output = tensors[params->tid_output];
  const Tensor& input = tensors[params->tid_input];
  const Tensor& filter = tensors[params->tid_filter];
  // const Tensor& bias = tensors[params->tid_bias];

  // Assune data_type is int8
  // const TfLiteType data_type = input->type;
  int total_width = input.dims[2]/* SizeOfDimension(input, 2) */;
  int total_height = input.dims[1];
  int filter_width = filter.dims[2]/* SizeOfDimension(filter, 2) */;
  int filter_height = filter.dims[1]/* SizeOfDimension(filter, 1) */;

  const DIM_TYPE* output_shape = output.dims;
  const int output_width = output_shape[2];
  const int output_height = (output_split_id>=(output_shape[1]/SPLIT_HEIGHT))? (output_shape[1]%SPLIT_HEIGHT):(SPLIT_HEIGHT) ;

  OP_utils::DepthwiseConv::OpData data;

  #if !OPT_OFFLOAD_ENABLE
  // v0.1 method: call CalculateOpData
  // Calculate padding and quantization params
  CalculateOpData(params, total_width, total_height,
                  filter_width, filter_height,
                  output_shape[2], output_shape[1], &data);
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
  #endif

  uint16_t pad_yt = 0, pad_yb = 0;
  int out_split_height = SPLIT_HEIGHT;
  if ((output_split_id+1)*SPLIT_HEIGHT>output_shape[1]){
    out_split_height = output_shape[1] % SPLIT_HEIGHT;
  }

  int input_split_id_begin = (output_split_id*SPLIT_HEIGHT) * params->stride_height - data.padding.height;
  int input_split_id_end = input_split_id_begin + (filter_height-1) + (params->stride_height*(out_split_height-1));
  pad_yb = input_split_id_end <= total_height-1 ? 0:input_split_id_end-total_height + 1;
  pad_yt = input_split_id_begin >= 0 ? 0:abs(input_split_id_begin);
  input_split_id_end = input_split_id_end < total_height? input_split_id_end:total_height-1;
  input_split_id_begin = input_split_id_begin > 0 ? input_split_id_begin: 0;
  const VirtualFp<int8_t> combined_ifmap(params->tid_input, input, input_split_id_begin, input_split_id_end, total_width*input.dims[3]);
  int combined_height = input_split_id_end-input_split_id_begin+1;

  const int8* filter_data = GetOfflineTensorData(params->tid_filter);
  const int32* bias_data= reinterpret_cast<const int32*>(GetOfflineTensorData(params->tid_bias));
  int8* output_data = GetSplitData(params->tid_output, output_split_id);
  {
    const int* input_shape = combined_ifmap.combined_dim;
    const DIM_TYPE* output_shape = tensors[params->tid_output].dims;
    const DIM_TYPE* filter_shape = tensors[params->tid_filter].dims;

    const int filter_height = filter_shape[1];
    const int filter_width = filter_shape[2];

    // const int input_height = combined_height;
    const int input_height = input_shape[1];
    const int input_width = input_shape[2];
    const int input_depth = input_shape[3];

    const int output_width = output_shape[2];

    int16_t* buf = reinterpret_cast<int16_t*>(arena+scratch_buffer_offset);

    if (params->depth_multiplier == 1) {
      TINY_ENGINE_FN(
              combined_ifmap.inputs_data, input_width, input_height,
              input_depth, filter_data, input_depth,
              filter_width, filter_height, data.padding.width,
              pad_yt, pad_yb, params->stride_width,
              params->stride_height, bias_data, data.contribs,
              output_data, data.per_channel_output_shift,
              data.per_channel_output_multiplier, output_width, output_height,
              *GetTensorQuantZP(params->tid_output)/* op_params.output_offset */,
              -(*GetTensorQuantZP(params->tid_input)) /* op_params.input_offset */,
              std::numeric_limits<int8_t>::min()/* op_params.quantized_activation_min */,
              std::numeric_limits<int8_t>::max()/* op_params.quantized_activation_max */,
              params->dilation_width_factor, params->dilation_height_factor, buf);//,
    } else {
      depthwise_conv_simd(params, &data, scratch_buffer_offset, combined_ifmap, filter_data, bias_data, output_data, output_height);
    }
  }
}
