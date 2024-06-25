FLAG_USE_TE_KERNEL = False

from utils import *
class OpParam_shared:
    def __init__(self, cpp_typename, cpp_instance_name):
        self.cpp_typename = cpp_typename
        self.cpp_instance_name = cpp_instance_name
    def UpdateIfNeeded(self, op_info, tensor_infos, op_infos, buffers):
        return
    def ToCppLiteral(self):
        raise("ToCppLiteral not implemented")
    def CalculateOpData(self, tensors):
        # optional member function
        # ret value must not be None if OpData offloading is needed
        return None

class ConvParam_shared(OpParam_shared):
    class OpData:
        def __init__(self):
            self.padding = {'height': 0, 'width': 0}
            self.per_channel_output_multiplier = None # int32 array
            self.per_channel_output_shift = None # int32 array, in [1,31]
            self.output_activation_min = None # int32, in [0, 255]
            self.output_activation_max = None # int32, in [0, 255]
            self.contribs = []
        def ToIntList(self):
            return [self.padding['height'], self.padding['width'], self.output_activation_min, self.output_activation_max]  + self.per_channel_output_multiplier + self.per_channel_output_shift + self.contribs

    def __init__(self, op_info, inputs, outputs, tensor_infos, op_infos, buffers):
        super().__init__("SharedParam_Conv", "shared_param_conv")
        self.input          = inputs[0]
        self.filter         = inputs[1]
        self.bias           = inputs[2]
        self.output         = outputs[0]
        self.padding        = op_info['builtin_options'].get('padding', 'SAME') # VALID/SAME
        self.fused_ActFunc  = op_info['builtin_options'].get('fused_activation_function', 'NONE') # NONE, RELU ...
        if len(op_info['inputs']) >= 4:
            pad_config_data = buffers[tensor_infos[op_info['inputs'][3]]['buffer']]['data']
            self.overridden_paddings = ByteToIntList(pad_config_data)
        else:
            self.overridden_paddings = None
        self.stride_h       = op_info['builtin_options']['stride_h']
        self.stride_w       = op_info['builtin_options']['stride_w']
        self.dilation_h     = op_info['builtin_options'].get('dilation_h_factor', 1)
        self.dilation_w     = op_info['builtin_options'].get('dilation_w_factor', 1)
        self.op_data = ConvParam_shared.OpData()
        self.op_data_offset = -1
    def UpdateIfNeeded(self, op_info, tensor_infos, op_infos, buffers):
        if len(op_info['inputs']) >= 4:
            pad_config_data = buffers[tensor_infos[op_info['inputs'][3]]['buffer']]['data']
            self.op_data.padding['height'] = self.overridden_paddings[0] = max(self.overridden_paddings[0], pad_config_data[0])
            self.op_data.padding['width'] = self.overridden_paddings[1] = max(self.overridden_paddings[1], pad_config_data[1])
    def CalculateOpData(self, tensors):
        from utils import ComputePaddingWithOffset, CalculateActivationRangeQuantized, ComputeContributions
        import numpy as np
        tensor_in = tensors[self.input]
        tensor_filter = tensors[self.filter]
        tensor_out = tensors[self.output]

        fn = tensor_filter.dim.N
        fh = tensor_filter.dim.H
        fw = tensor_filter.dim.W
        fc = tensor_filter.dim.C

        filter_slice = [np.int8(x) for x in tensor_filter.offline_data]

        if (fh == 1 and fw == 1 and 
            self.dilation_h == 1 and self.dilation_w == 1):
            self.op_data.contribs = ComputeContributions(tensor_out.dim.C, tensor_in.dim.C, filter_slice, tensor_in.quant_param.quant_zp)
        else:
            self.op_data.contribs = []

        # padding
        if self.overridden_paddings is None:
            self.op_data.padding['height'] = ComputePaddingWithOffset(
                                                self.stride_h, self.dilation_h,
                                                tensor_in.dim.H, tensor_filter.dim.H, tensor_out.dim.H)
            self.op_data.padding['width'] = ComputePaddingWithOffset(
                                                self.stride_w, self.dilation_w,
                                                tensor_in.dim.W, tensor_filter.dim.W, tensor_out.dim.W)
        else:
            self.op_data.padding['height'] = self.overridden_paddings[0]
            self.op_data.padding['width']  = self.overridden_paddings[1]
        # per channel quant param
        import tempfile, os
        with tempfile.TemporaryDirectory(dir='.') as tmp_dir_path:
            with open(os.path.join(tmp_dir_path, 'input'), 'w') as f_in:
                import csv
                writer = csv.writer(f_in, delimiter=' ')
                writer.writerows([[tensor_out.dim.C],
                                tensor_in.quant_param.quant_scale,
                                tensor_out.quant_param.quant_scale,
                                tensor_filter.quant_param.quant_scale])
            os.system(f"utils/calc_op_data_conv {tmp_dir_path}")
            with open(os.path.join(tmp_dir_path, 'output'), 'r') as f_out:
                atoi_list = lambda x: [y for y in map(int, x)]
                self.op_data.per_channel_output_multiplier = atoi_list(f_out.readline().rstrip('\n').rstrip(' ').split(' '))
                self.op_data.per_channel_output_shift = atoi_list(f_out.readline().rstrip('\n').rstrip(' ').split(' '))
        # output activation range
        act_range = CalculateActivationRangeQuantized(self.fused_ActFunc, tensor_out.quant_param.quant_scale[0], tensor_out.quant_param.quant_zp[0])
        self.op_data.output_activation_min, self.op_data.output_activation_max = act_range

        tensors[self.bias].is_bias = True
        
        if (tensor_in.is_splitted() and
            tensor_out.is_splitted()):
            tensor_in.quant_param.scale_deprecated = True
            tensor_filter.quant_param.scale_deprecated = True
            tensor_out.quant_param.scale_deprecated = True
            tensors[self.bias].quant_param.scale_deprecated = True

        return 0
    def ToCppLiteral(self):
        return f"{{{self.input}, {self.filter}, {self.bias}, {self.output}, {0 if self.padding=='SAME' else 1}, {1 if self.fused_ActFunc=='RELU' else 0}, {self.op_data_offset}, {self.stride_w}, {self.stride_h}, {self.dilation_w}, {self.dilation_h}}}"

class DwConvParam_shared(OpParam_shared):
    class OpData:
        def __init__(self):
            self.padding = {'height': 0, 'width': 0}
            self.per_channel_output_multiplier = None # int32 array
            self.per_channel_output_shift = None # int32 array, in [1,31]
            self.output_activation_min = None # int32, in [0, 255]
            self.output_activation_max = None # int32, in [0, 255]
            self.offsetBias = []
        def ToIntList(self):
            return [self.padding['height'], self.padding['width'], self.output_activation_min, self.output_activation_max] + self.per_channel_output_multiplier + self.per_channel_output_shift + self.offsetBias

    def __init__(self, op_info, inputs, outputs, tensor_infos, op_infos, buffers):
        super().__init__('SharedParam_Depthwise_Conv', "shared_param_dwconv")
        self.input              = inputs[0]
        self.filter             = inputs[1]
        self.bias               = inputs[2]
        self.output             = outputs[0]
        self.depth_multiplier   = op_info['builtin_options'].get('depth_multiplier', 0) # VALID/SAME
        self.padding            = op_info['builtin_options'].get('padding', 'SAME') # VALID/SAME
        self.fused_ActFunc      = op_info['builtin_options'].get('fused_activation_function', 'NONE') # NONE, RELU ...
        if len(op_info['inputs']) >= 4:
            pad_config_data = buffers[tensor_infos[op_info['inputs'][3]]['buffer']]['data']
            self.overridden_paddings = ByteToIntList(pad_config_data)
        else:
            self.overridden_paddings = None
        self.stride_h           = op_info['builtin_options']['stride_h']
        self.stride_w           = op_info['builtin_options']['stride_w']
        self.dilation_h         = op_info['builtin_options'].get('dilation_h_factor', 1)
        self.dilation_w         = op_info['builtin_options'].get('dilation_w_factor', 1)
        self.op_data = DwConvParam_shared.OpData()
        self.op_data_offset = -1
    def UpdateIfNeeded(self, op_info, tensor_infos, op_infos, buffers):
        if len(op_info['inputs']) >= 4:
            pad_config_data = buffers[tensor_infos[op_info['inputs'][3]]['buffer']]['data']
            self.op_data.padding['height'] = self.overridden_paddings[0] = max(self.overridden_paddings[0], pad_config_data[0])
            self.op_data.padding['width'] = self.overridden_paddings[1] = max(self.overridden_paddings[1], pad_config_data[1])
    def CalculateOpData(self, tensors):
        from utils import ComputePaddingWithOffset, CalculateActivationRangeQuantized, ComputeOffsetBias
        tensor_in = tensors[self.input]
        tensor_filter = tensors[self.filter]
        tensor_out = tensors[self.output]
        # padding
        if self.overridden_paddings is None:
            self.op_data.padding['height'] = ComputePaddingWithOffset(
                                                self.stride_h, self.dilation_h,
                                                tensor_in.dim.H, tensor_filter.dim.H, tensor_out.dim.H)
            self.op_data.padding['width'] = ComputePaddingWithOffset(
                                                self.stride_w, self.dilation_w,
                                                tensor_in.dim.W, tensor_filter.dim.W, tensor_out.dim.W)
        else:
            self.op_data.padding['height'] = self.overridden_paddings[0]
            self.op_data.padding['width']  = self.overridden_paddings[1]

        # offload contribs
        fn = tensor_filter.dim.N
        fh = tensor_filter.dim.H
        fw = tensor_filter.dim.W
        fc = tensor_filter.dim.C

        import numpy as np
        filter_slice = [np.int8(x) for x in tensor_filter.offline_data]
        self.op_data.offsetBias = ComputeOffsetBias(tensor_in.quant_param.quant_zp, filter_slice, tensor_in.dim.C)

        # convert filter's data format to NCHW
        NCHW_np = np.array(tensor_filter.offline_data).reshape([fn, fh, fw, fc]).transpose([0,3,1,2])
        tensor_filter.offline_data = NCHW_np.flatten().tolist()

        # per channel quant param
        import tempfile, os
        with tempfile.TemporaryDirectory(dir='.') as tmp_dir_path:
            with open(os.path.join(tmp_dir_path, 'input'), 'w') as f_in:
                import csv
                writer = csv.writer(f_in, delimiter=' ')
                writer.writerows([[tensor_out.dim.C],
                                tensor_in.quant_param.quant_scale,
                                tensor_out.quant_param.quant_scale,
                                tensor_filter.quant_param.quant_scale])
            os.system(f"utils/calc_op_data_conv {tmp_dir_path}")
            with open(os.path.join(tmp_dir_path, 'output'), 'r') as f_out:
                atoi_list = lambda x: [y for y in map(int, x)]
                self.op_data.per_channel_output_multiplier = atoi_list(f_out.readline().rstrip('\n').rstrip(' ').split(' '))
                self.op_data.per_channel_output_shift = atoi_list(f_out.readline().rstrip('\n').rstrip(' ').split(' '))
        # output activation range
        act_range = CalculateActivationRangeQuantized(self.fused_ActFunc, tensor_out.quant_param.quant_scale[0], tensor_out.quant_param.quant_zp[0])
        self.op_data.output_activation_min, self.op_data.output_activation_max = act_range

        tensors[self.bias].is_bias = True

        if (tensor_in.is_splitted() and
            tensor_out.is_splitted()):
            tensor_in.quant_param.scale_deprecated = True
            tensor_filter.quant_param.scale_deprecated = True
            tensor_out.quant_param.scale_deprecated = True
            tensors[self.bias].quant_param.scale_deprecated = True
        
        return 0
    def ToCppLiteral(self):
        return f"{{{self.input}, {self.filter}, {self.bias}, {self.output}, {self.depth_multiplier}, {0 if self.padding=='SAME' else 1}, {1 if self.fused_ActFunc=='RELU' else 0}, {self.op_data_offset}, {self.stride_w}, {self.stride_h}, {self.dilation_w}, {self.dilation_h}}}"

class PadParam_shared(OpParam_shared):
    Pad_LUT = {}
    def __init__(self, op_info, inputs, outputs):
        super().__init__('SharedParam_Pad', "shared_param_pad")
        self.input = inputs[0]
        self.output = outputs[0]
        self.paddings = inputs[1]
    def ToCppLiteral(self):
        return f"{{{self.input}, {self.output}, {self.paddings}}}"

class AddParam_shared(OpParam_shared):
    class OpData:
        def __init__(self):
            self.output_activation_min = None
            self.output_activation_max = None
            self.shift = None # int32 array[3]
            self.multiplier = None # int32 array[3]
            self.offset = None # int32 array[3]
        def ToIntList(self):
            return [self.output_activation_min, self.output_activation_max] + self.shift + self.multiplier + self.offset
    def __init__(self, op_info, inputs, outputs):
        super().__init__('SharedParam_Add', "shared_param_add")
        self.input_A = inputs[0]
        self.input_B = inputs[1]
        self.output = outputs[0]
        self.fused_ActFunc  = op_info['builtin_options'].get('fused_activation_function', 'NONE') # NONE, RELU ...
        self.pot_scale_int16 = op_info['builtin_options'].get('pot_scale_int16', True) # TODO: what's this?
        self.op_data = AddParam_shared.OpData()
        self.op_data_offset = -1
    def CalculateOpData(self, tensors):
        tensor_input_A = tensors[self.input_A]
        tensor_input_B = tensors[self.input_B]
        tensor_output = tensors[self.output]
        # quant param
        self.op_data.offset = ([-tensor_input_A.quant_param.quant_zp[0],
                                -tensor_input_B.quant_param.quant_zp[0],
                                tensor_output.quant_param.quant_zp[0]])
        import tempfile, os
        with tempfile.TemporaryDirectory(dir='.') as tmp_dir_path:
            with open(os.path.join(tmp_dir_path, 'input'), 'w') as f_in:
                import csv
                writer = csv.writer(f_in, delimiter=' ')
                writer.writerows([[tensor_input_A.quant_param.quant_scale[0],
                                    tensor_input_B.quant_param.quant_scale[0],
                                    tensor_output.quant_param.quant_scale[0]]])
            os.system(f"utils/calc_op_data_add {tmp_dir_path}")
            with open(os.path.join(tmp_dir_path, 'output'), 'r') as f_out:
                atoi_list = lambda x: [y for y in map(int, x)]
                self.op_data.multiplier = atoi_list(f_out.readline().rstrip('\n').rstrip(' ').split(' '))
                self.op_data.shift = atoi_list(f_out.readline().rstrip('\n').rstrip(' ').split(' '))
        # output activation range
        act_range = CalculateActivationRangeQuantized(self.fused_ActFunc, tensor_output.quant_param.quant_scale[0], tensor_output.quant_param.quant_zp[0])
        self.op_data.output_activation_min, self.op_data.output_activation_max = act_range

        if (tensor_input_A.is_splitted() and 
            tensor_input_B.is_splitted() and
            tensor_output.is_splitted()):
            tensor_input_A.quant_param.scale_deprecated = True
            tensor_input_B.quant_param.scale_deprecated = True
            tensor_output.quant_param.scale_deprecated = True
        
        return 0
    def ToCppLiteral(self):
        return f"{{{self.input_A}, {self.input_B}, {self.output}, {self.op_data_offset}, {1 if self.fused_ActFunc=='RELU' else 0}, {'true' if self.pot_scale_int16 else 'false'}}}"

class AvgPoolParam_shared(OpParam_shared):
    def __init__(self, op_info, inputs, outputs):
        super().__init__('SharedParam_AvgPool', "shared_param_avgpool")
        self.input = inputs[0]
        self.output = outputs[0]
        self.filter_height =  op_info['builtin_options'].get('filter_height',1)
        self.filter_width =  op_info['builtin_options'].get('filter_width',1)
        self.stride_h =  op_info['builtin_options'].get('stride_h',1)
        self.stride_w =  op_info['builtin_options'].get('stride_w',1)
        self.padding =  op_info['builtin_options'].get('padding', 'SAME') # VALID/SAME
        self.fused_ActFunc  =  op_info['builtin_options'].get('fused_activation_function', 'NONE') # NONE, RELU ...
    def ToCppLiteral(self):
        return f"{{{self.input}, {self.output}, {self.filter_height}, {self.filter_width}, {self.stride_h}, {self.stride_w}, {0 if self.padding=='SAME' else 1}, {1 if self.fused_ActFunc=='RELU' else 0}}}"

class MaxPoolParam_shared(OpParam_shared):
    def __init__(self, op_info, inputs, outputs):
        super().__init__('SharedParam_MaxPool', "shared_param_maxpool")
        self.input = inputs[0]
        self.output = outputs[0]
        self.filter_height =  op_info['builtin_options'].get('filter_height',1)
        self.filter_width =  op_info['builtin_options'].get('filter_width',1)
        self.stride_h =  op_info['builtin_options'].get('stride_h',1)
        self.stride_w =  op_info['builtin_options'].get('stride_w',1)
        self.padding =  op_info['builtin_options'].get('padding', 'SAME') # VALID/SAME
        self.fused_ActFunc  =  op_info['builtin_options'].get('fused_activation_function', 'NONE') # NONE, RELU ...
    def ToCppLiteral(self):
        return f"{{{self.input}, {self.output}, {self.filter_height}, {self.filter_width}, {self.stride_h}, {self.stride_w}, {0 if self.padding=='SAME' else 1}, {1 if self.fused_ActFunc=='RELU' else 0}}}"

class ReshapeParam_shared(OpParam_shared):
    def __init__(self, op_info, inputs, outputs, target_shape):
        super().__init__('SharedParam_Reshape', "shared_param_reshape")
        self.input = inputs[0]
        self.output = outputs[0]
        self.target_shape = target_shape
    def ToCppLiteral(self):
        return f"{{{self.input}, {self.output}}}"

class FCParam_shared(OpParam_shared):
    def __init__(self, op_info, inputs, outputs):
        super().__init__('SharedParam_FC', "shared_param_fc")
        self.input = inputs[0]
        self.weights = inputs[1]
        self.bias = inputs[2]
        self.output = outputs[0]
        self.fused_ActFunc  = op_info['builtin_options'].get('fused_activation_function', 'NONE') # NONE, RELU ...
        self.weights_format = op_info['builtin_options'].get('weights_format', "DEFAULT") # TODO: what's this?
        self.asymmetric_quant_input = op_info['builtin_options'].get('asymmetric_quant_input', False) # TODO: what's this?
        self.keep_num_dims = op_info['builtin_options'].get('keep_num_dims', False) # TODO: what's this?
    def ToCppLiteral(self):
        return f"{{{self.input}, {self.weights}, {self.bias}, {self.output}, {1 if self.fused_ActFunc=='RELU' else 0}, {0 if self.weights_format=='DEFAULT' else 1}, {'ture' if self.asymmetric_quant_input else 'false'}, {'ture' if self.keep_num_dims else 'false'}}}"

class SoftmaxParam_shared(OpParam_shared):
    def __init__(self, op_info, inputs, outputs):
        super().__init__('SharedParam_Softmax', "shared_param_softmax")
        self.input = inputs[0]
        self.output = outputs[0]
        self.beta = op_info['builtin_options'].get('beta', 1.0) # TODO: what's this?
    def ToCppLiteral(self):
        return f"{{{self.input}, {self.output}, {self.beta}}}"

class SplitParam_shared(OpParam_shared):
    def __init__(self, op_info, inputs, outputs, axis):
        super().__init__('SharedParam_Split', "shared_param_split")
        self.input = inputs[1]
        self.output = outputs[0]
        self.num_splits = op_info['builtin_options'].get('num_splits', 0)
        self.axis = axis
    def ToCppLiteral(self):
        return f"{{{self.input}, {self.output}, {self.num_splits}, {self.axis}}}"
class ConcatParam_shared(OpParam_shared):
    def __init__(self, op_info, inputs, outputs):
        super().__init__('SharedParam_Concat', "shared_param_concat")
        self.input = inputs[0]
        self.output = outputs[0]
        self.axis = op_info['builtin_options'].get('axis',1)
    def ToCppLiteral(self):
        return f"{{{self.input}, {self.output}, {self.axis}}}"

class LeakyReluParam_shared(OpParam_shared):
    def __init__(self, op_info, inputs, outputs):
        super().__init__('SharedParam_LeakyRelu', "shared_param_leakyrelu")
        self.input = inputs[0]
        self.output = outputs[0]
        raise(BaseException("TODO: Leaky Relu"))
        self.alpha = None
    def ToCppLiteral(self):
        return f"{{{self.input}, {self.output}, {self.axis}}}"

class PadParam_shared(OpParam_shared):
    Pad_LUT = {}
    def __init__(self, op_info, inputs, outputs):
        super().__init__('SharedParam_Pad', "shared_param_pad")
        self.input = inputs[0]
        self.output = outputs[0]
        self.paddings = inputs[1]
    def ToCppLiteral(self):
        return f"{{{self.input}, {self.output}, {self.paddings}}}"

class OpParam_private:
    def __init__(self):
        self.scratch_buffer_offset = -1
    def CalculateScratchBufferSize(self, tensors, shared_param):
        return 0
    def RequestScratchBuffer(self, tensors, shared_param):
        size_required = self.CalculateScratchBufferSize(tensors, shared_param)
        if size_required > 0:
            return self, size_required
        else:
            return None, 0

class ConvParam_private(OpParam_private):
    def __init__(self, inputs_sid, outputs_sid):
        super().__init__()
        if len(inputs_sid) >= 4:
            self.input_split_id = inputs_sid[0]
            self.out_split_id   = outputs_sid[0]
        else:
            self.input_split_id = -1
            self.out_split_id   = -1
    def CalculateScratchBufferSize(self, tensors, shared_param:ConvParam_shared):
        dim_input = tensors[shared_param.input].dim
        dim_output = tensors[shared_param.output].dim
        dim_filter = tensors[shared_param.filter].dim
        pad_H = shared_param.op_data.padding['height']
        pad_W = shared_param.op_data.padding['width']
        # 1x1
        if (pad_H == 0 and pad_W == 0 and
            shared_param.stride_h == 1 and shared_param.stride_h == 1 and
            dim_input.C % 4 == 0 and dim_filter.H == 1 and dim_filter.W == 1) :
            if (dim_input.C in [8, 16, 24, 48]):
                # TE impl needs 2 cols buffer: (2 * input_dims->c * filter_dims->w * filter_dims->h) * sizeof(int16_t)
                return (2 * dim_input.C * 1 * 1) * 2
            else:
                # CMSIS-NN arm_convolve_1x1_s8_fast doesn't need a scratch buffer
                return 0
        # 1xn
        elif (dim_output.H == 1 and dim_input.H == 1 and dim_filter.H == 1 and
            dim_output.W % 4 == 0 and dim_input.N == 1):
            # return (2 * input_dims->c * filter_dims->w * filter_dims->h) * sizeof(int16_t);
            return (2 * dim_input.C * dim_filter.W * dim_filter.H) * 2
        # 3x3 ich3 st2 pad1
        elif (dim_input.C == 3 and dim_filter.H ==3 and dim_filter.W == 3 and
                shared_param.stride_h == 2 and shared_param.stride_w == 2 and pad_H == 1 and pad_W ==1):
            # runtime_buf= <1 channel of 3x3x3 filter data size> * <2cols> * <sizeof(int16)>
            # k_buf = <filter_data_size> * <sizeof(int16)>
            return 27*2*2 + dim_filter.N*27*2
        # normal
        else:
            # return (2 * input_dims->c * filter_dims->w * filter_dims->h) * sizeof(int16_t);
            return (2 * dim_input.C * dim_filter.W * dim_filter.H) * 2

class DwConvParam_private(OpParam_private):
    def __init__(self, inputs_sid, outputs_sid):
        super().__init__()
        if len(inputs_sid) >= 4:
            self.out_split_id   = outputs_sid[0]
        else:
            self.out_split_id   = -1
    def CalculateScratchBufferSize(self, tensors, shared_param:DwConvParam_shared):
        input_tensor = tensors[shared_param.input]
        output_tensor = tensors[shared_param.output]
        filter_tensor = tensors[shared_param.filter]
        dim_input = input_tensor.dim
        dim_output = output_tensor.dim
        dim_filter = filter_tensor.dim

        pad_H_ori = shared_param.op_data.padding['height']
        pad_W_ori = shared_param.op_data.padding['width']

        # TE need size of one channel of input feature map with paddings surrounded
        # size = 1 ch * (in_H+pad_H*2) * (in_W+pad_W*2)
        if self.out_split_id == -1:
            return (dim_input.W+pad_W_ori*2) * (dim_input.H+pad_H_ori*2)
        else:
            from utils import GetSplitHeightBySid
            out_split_height = GetSplitHeightBySid(output_tensor, self.out_split_id)
            in_row_head = (self.out_split_id*input_tensor.split_height) * shared_param.stride_h - pad_H_ori
            in_row_tail = in_row_head + (dim_filter.H-1) + (shared_param.stride_h*(out_split_height-1))
            pad_HT = 0 if in_row_head>0 else -in_row_head # paddings on height dim, top region
            pad_HB = 0 if in_row_tail<dim_input.H else in_row_tail-dim_input.H + 1 # paddings on height dim, bottom region
            in_row_head = 0 if in_row_head<0 else in_row_head
            in_row_tail = dim_input.H-1 if in_row_tail > dim_input.H else in_row_tail
            compile_time_padded_x = (dim_input.W+pad_W_ori*2)
            compile_time_padded_y = ((in_row_tail-in_row_head+1)+pad_HT+pad_HB)
            receptive_field_x = (dim_output.W-1)*shared_param.stride_w + dim_filter.W
            receptive_field_y = (out_split_height-1)*shared_param.stride_h + dim_filter.H
            runtime_time_padded_x = max(compile_time_padded_x, receptive_field_x)
            runtime_time_padded_y = max(compile_time_padded_y, receptive_field_y)
            return runtime_time_padded_x * runtime_time_padded_y
        
        # CMSIS-NN arm_depthwise_conv_s8 # deprecated
        # (input_ch * kernel_x * kernel_y) * sizeof(int16_t);
        # return (dim_input.C * (dim_filter.W+pad_W*2) * (dim_filter.H+pad_H*2)) * 2

class PadParam_private(OpParam_private):
    def __init__(self, inputs_sid, outputs_sid):
        super().__init__()
        self.input_sid = inputs_sid[0]
        self.padding_sid = inputs_sid[1]
        self.output_sid = outputs_sid[0]

class AddParam_private(OpParam_private):
    def __init__(self, inputs_sid, outputs_sid):
        super().__init__()
        # TODO: all of this should be the same. Double check is needed. If it is, then only one is needed
        self.input_A_sid = inputs_sid[0]
        self.input_B_sid = inputs_sid[1]
        self.output_sid = outputs_sid[0]

class FCParam_private(OpParam_private):
    def __init__(self):
        super().__init__()
    def CalculateScratchBufferSize(self, tensors, shared_param:FCParam_shared):
        return 0
class AvgPoolParam_private(OpParam_private):
    def __init__(self):
        super().__init__()
    def CalculateScratchBufferSize(self, tensors, shared_param:AvgPoolParam_shared):
        dim_input = tensors[shared_param.input].dim
        # return (ch_src * sizeof(int16_t));
        return dim_input.C * 2
class MaxPoolParam_private(OpParam_private):
    def __init__(self):
        super().__init__()
    def CalculateScratchBufferSize(self, tensors, shared_param:MaxPoolParam_shared):
        dim_input = tensors[shared_param.input].dim
        # return (ch_src * sizeof(int16_t));
        return dim_input.C * 2
