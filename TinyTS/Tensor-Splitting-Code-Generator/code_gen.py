import copy
import shutil
from distutils.dir_util import copy_tree

from requests import options
from MemPlanner import *
from OpType import *
from GenType import *
from MemProfiler import MemoryProfiler


from depthwiseTemplate import depthwiseInplace
import os

class includeFile:
    def __init__(self, path):
        self.path = path
        self.defstring = ""

    def addDefine(self, defstr):
        self.defstring += defstr + ";\n"

    def writeFile(self):
        import os

        outpath = os.path.join(self.path, "genInclude.h")
        outf = open(outpath, "w")
        outf.write(self.defstring)
        outf.close()

class DwconvOpFile:
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename
        self.defstring = ""

    def addDefine(self, defstr):
        self.defstring += defstr

    def writeFile(self):
        import os

        outpath = os.path.join(self.path, self.filename)
        outf = open(outpath, "w")
        outf.write(self.defstring)
        outf.close()

class DepthWiseConvOp:
    def __init__(self, kernel_h, kernel_w, pad_h, pad_w, stride_h, input_c):
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.stride_h = stride_h
        self.input_c = input_c
        # pad_h = (kernel_h - 1) // 2 # from tinyengine
        # pad_w = (kernel_w - 1) // 2 # from tinyengine

    def __eq__(self, other):
        if isinstance(other, DepthWiseConvOp):
            if (
                self.kernel_h == other.kernel_h
                and self.kernel_w == other.kernel_w
                and self.stride_h == other.stride_h
            ):
                return True
            else:
                return False
        return NotImplemented

class CodeGenerator:
    def __init__(self, model, model_name, alignment, split_height, plot_result, enable_te, 
                output_root_dir='.', enabled_mem_plan_addons=[], 
                exp_esti=False, evict_in=False, inplace_add=False, inplace_dw_conv=False):
        self.model = copy.deepcopy(model)
        self.model_name = model_name
        self.alignment = alignment
        self.split_height = split_height
        self.enable_te = enable_te
        self.plot_result = plot_result
        FLAG_USE_TE_KERNEL = enable_te
        self.output_root_dir = output_root_dir
        self.TE_Codegen_root = f"{self.output_root_dir}/codegen/"
        self.include_path = self.TE_Codegen_root + "Include/"
        self.source_path = self.TE_Codegen_root + "Source/"
        os.makedirs(f'{self.output_root_dir}/gen_model/include/', exist_ok=True)
        os.makedirs(f'{self.output_root_dir}/gen_model/src/', exist_ok=True)
        copy_tree(os.path.join(os.path.dirname(__file__), 'TE_codgen_template'), self.TE_Codegen_root)
        self.enabled_mem_plan_addons = enabled_mem_plan_addons
        
        # Flags for experimental features
        self.experimental_estimation = exp_esti
        self.evict_input = evict_in
        self.inplace_add = inplace_add
        self.inplace_dw_conv = inplace_dw_conv
        self.mem_profiler = MemoryProfiler(self.model_name)
        self.import_split = []

        # build tensor LUT first
        # these LUTs is used to translate original id(normal and splits) to new id(normal and combined splits)
        self.TensorLUT, self.TensorLUT_r, self.OperatorLUT, self.OperatorLUT_r = self.buildLUT()
        self.tensors, self.operators, self.op_shared_params = self.extract_properties()
        self.op_data = self.OffloadCalculateOpData()
        self.split_offset, self.arena_size = self.CalculateOnlineTensorOffset()

    def buildLUT(self):
        subgraph = self.model['subgraphs'][0]
        tensors = self.model['subgraphs'][0]['tensors']
        operators = self.model['subgraphs'][0]['operators']

        # tensors
        TensorLUT = [ -1 for i in range(len(tensors))]
        TensorLUT_r = []
        TensorLUT_name = {}
        for tid_ori, t in enumerate(tensors):
            t['id'] = tid_ori
            # split name format: ${name}_split_${split_id}
            tokens = t['name'].split('_split_')
            name = tokens[0]
            if len(tokens) == 2:
                if tokens[1].isnumeric():
                    t['sid'] = sid = int(tokens[1])
                    name+='_s'
                # split op's axis tensor
                else:
                    name = t['name']
            elif len(tokens)>2:
                print(tokens)
                raise('ERROR')
            tid_new = TensorLUT_name.get(name, -1)
            if tid_new == -1:
                TensorLUT_name[name] = tid_new = len(TensorLUT_name)
                TensorLUT_r.append([])
            t['id_new'] = TensorLUT[tid_ori] = tid_new
            TensorLUT_r[tid_new].append(tid_ori)
        # for split_oriTIDs in TensorLUT_r:
        #     if len(split_oriTIDs) == 1:
        #         tensors[split_oriTIDs[0]]['sid'] = -1


        # ops
        OperatorLUT = [ -1 for i in range(len(operators))]
        OperatorLUT_r = []
        OperatorLUT_O = {}
        deprecated_builtin_code_LUT = {
            0: 'ADD',
            1: 'AVERAGE_POOL_2D',
            2: 'CONCATENATION',
            3: 'CONV_2D',
            4: 'DEPTHWISE_CONV_2D',
            9: 'FULLY_CONNECTED',
            17: 'MAX_POOL_2D',
            22: 'RESHAPE',
            25: 'SOFTMAX',
            34: 'PAD',
            49: 'SPLIT',
            98: 'LEAKY_RELU'
        }
        for opid_ori, op in enumerate(operators):
            opcode_index = op.get('opcode_index', 0)
            builtin_code = self.model['operator_codes'][opcode_index].get('builtin_code', 'UNKNOWN')
            deprecated_builtin_code = self.model['operator_codes'][opcode_index].get('deprecated_builtin_code', -1)
            if builtin_code != 'UNKNOWN':
                op['type'] = builtin_code
            elif deprecated_builtin_code != -1:
                op['type'] = deprecated_builtin_code_LUT[deprecated_builtin_code]
            else:
                op['type'] = 'ADD'
            op['id'] = opid_ori
            if opcode_index != 0:
                del op['opcode_index']
            if len(op['outputs'])>1 and op['type'] != 'SPLIT':
                print(op)
            out_tid = op['outputs'][0]
            out_tid_new = TensorLUT[out_tid]
            opid_shared = OperatorLUT_O.get(out_tid_new, -1)
            if opid_shared == -1:
                OperatorLUT_O[out_tid_new] = opid_shared = len(OperatorLUT_O)
                OperatorLUT_r.append([])
            op['id_shared'] = OperatorLUT[opid_ori] = opid_shared
            OperatorLUT_r[opid_shared].append(opid_ori)
        return TensorLUT, TensorLUT_r, OperatorLUT, OperatorLUT_r

    def extract_properties(self):
        from utils import get_aligned_data_list
        tensor_infos = self.model['subgraphs'][0]['tensors']
        op_infos = self.model['subgraphs'][0]['operators']
        buffers = self.model['buffers']
        self.input = self.model['subgraphs'][0]['inputs'][0]
        self.input = self.TensorLUT[self.input]
        self.output = self.model['subgraphs'][0]['outputs'][0]
        self.output = self.TensorLUT[self.output]

        #tensor
        tensors = [Tensor() for x in range(len(self.TensorLUT_r))]
        # tensor_data = [] # Deprecated, postponed to PrepareOfflineTensorData()
        split_cnt = 0
        for t in tensor_infos:
            tid_new = t['id_new']
            if tensors[tid_new].visited == False:
                dim = TensorDim(t['shape'])
                type = t['type']
                # quant data
                quant_param = None
                quant_offset = -1
                if t.get('quantization', None) is not None:
                    quant_param = QuantParam()
                    quant_param.quant_min   = t['quantization'].get('min', [])
                    quant_param.quant_max   = t['quantization'].get('max', [])
                    quant_param.quant_scale = t['quantization'].get('scale', [])
                    quant_param.quant_zp    = t['quantization'].get('zero_point', [])
                # tensor data
                data_offset = -1
                data = buffers[t['buffer']].get('data', [])
                offline_data = []
                if len(data) > 0:
                    data_offset = 0
                    offline_data += data
                # split
                splits_offset = -1
                # if len(self.TensorLUT_r[tid_new]) > 1:
                if t.get('sid',-1) != -1:
                    data_offset = -len(self.TensorLUT_r[tid_new])
                    splits_offset = split_cnt
                    split_cnt += len(self.TensorLUT_r[tid_new])
                    tensors[tid_new].splitted = True
                # fill
                tensors[tid_new].fill(
                    dim, type, offline_data, quant_offset, data_offset, splits_offset, 
                    len(self.TensorLUT_r[tid_new]), self.split_height, quant_param)
            elif tensors[tid_new].visited == True:
                dim = TensorDim(t['shape'])
                tensors[tid_new].dim.concat(dim, 1)

        # op
        depthwise_conv_ops = []

        operators = [Op() for _ in range(len(op_infos))]
        shared_param = [None for _ in range(len(self.OperatorLUT_r))]
        for opid, op_info in enumerate(op_infos):
            op = operators[opid]
            opid_shared = op_info['id_shared']
            op.id = opid
            op.type = op_info['type']
            op.shared_param_id = opid_shared
            inputs = [self.TensorLUT[tid] for tid in op_info['inputs']]
            outputs = [self.TensorLUT[tid] for tid in op_info['outputs']]
            inputs_sid = [tensor_infos[tid_ori].get('sid',-1) for tid_ori in op_info['inputs']]
            outputs_sid = [tensor_infos[tid_ori].get('sid',-1) for tid_ori in op_info['outputs']]
            # type switch, need to move to util and modulize this part
            if   op.type == "CONV_2D":
                if shared_param[opid_shared] == None:
                    shared_param[opid_shared] = ConvParam_shared(op_info, inputs, outputs, tensor_infos, op_infos, buffers)
                else:
                    shared_param[opid_shared].UpdateIfNeeded(op_info, tensor_infos, op_infos, buffers)
                op.private_param = ConvParam_private(inputs_sid, outputs_sid)
            elif op.type == "DEPTHWISE_CONV_2D":
                if shared_param[opid_shared] == None:
                    shared_param[opid_shared] = DwConvParam_shared(op_info, inputs, outputs, tensor_infos, op_infos, buffers)
                else:
                    shared_param[opid_shared].UpdateIfNeeded(op_info, tensor_infos, op_infos, buffers)
                op.private_param = DwConvParam_private(inputs_sid, outputs_sid)

                # TODO do this part after CalcOpData would be better
                from utils import ComputePaddingWithOffset
                shared = shared_param[opid_shared]

                tensor_in = tensors[shared.input]
                tensor_filter = tensors[shared.filter]
                tensor_out = tensors[shared.output]

                kernel_h = tensor_filter.dim.H
                kernel_w = tensor_filter.dim.W

                pad_h = 0
                pad_w = 0
                if shared.overridden_paddings is None:
                    pad_h = ComputePaddingWithOffset(
                                shared.stride_h, shared.dilation_h,
                                tensor_in.dim.H, tensor_filter.dim.H, tensor_out.dim.H)
                    pad_w = ComputePaddingWithOffset(
                                shared.stride_w, shared.dilation_w,
                                tensor_in.dim.W, tensor_filter.dim.W, tensor_out.dim.W)
                else:
                    pad_h = shared.overridden_paddings[0]
                    pad_w = shared.overridden_paddings[1]

                stride_h = shared.stride_h
                input_c = tensor_in.dim.C

                dwcop = DepthWiseConvOp(kernel_h, kernel_w, pad_h, pad_w, stride_h, input_c)
                if dwcop not in depthwise_conv_ops:
                    depthwise_conv_ops.append(dwcop)

            elif op.type == "ADD":
                if shared_param[opid_shared] == None:
                    shared_param[opid_shared] = AddParam_shared(op_info, inputs, outputs)
                op.private_param = AddParam_private(inputs_sid, outputs_sid)
            elif op.type == "AVERAGE_POOL_2D":
                if shared_param[opid_shared] == None:
                    shared_param[opid_shared] = AvgPoolParam_shared(op_info, inputs, outputs)
                op.private_param = AvgPoolParam_private()
            elif op.type == "MAX_POOL_2D":
                if shared_param[opid_shared] == None:
                    shared_param[opid_shared] = MaxPoolParam_shared(op_info, inputs, outputs)
                op.private_param = MaxPoolParam_private()
            elif op.type == "RESHAPE":
                target_shape_tid = op_info['inputs'][1]
                target_shape_bid = tensor_infos[target_shape_tid]['buffer']
                target_shape_data = self.model['buffers'][target_shape_bid]['data']
                target_shape = ByteToIntList(target_shape_data)
                shared_param[opid_shared] = ReshapeParam_shared(op_info, inputs, outputs, target_shape)
            elif op.type == "FULLY_CONNECTED":
                shared_param[opid_shared] = FCParam_shared(op_info, inputs, outputs)
                op.private_param = FCParam_private()
            elif op.type == "SOFTMAX":
                shared_param[opid_shared] = SoftmaxParam_shared(op_info, inputs, outputs)
            elif op.type == "SPLIT":
                axis_tid = op_info['inputs'][0]
                axis_bid = tensor_infos[axis_tid]['buffer']
                axis_data = self.model['buffers'][axis_bid]['data']
                axis = int.from_bytes(axis_data, 'little', signed=True)
                shared_param[opid_shared] = SplitParam_shared(op_info, inputs, outputs, axis)
            elif op.type == "CONCATENATION":
                shared_param[opid_shared] = ConcatParam_shared(op_info, inputs, outputs)
            elif op.type == "LEAKY_RELU":
                shared_param[opid_shared] = LeakyReluParam_shared(op_info, inputs, outputs)
            elif op.type == "PAD":
                shared_param[opid_shared] = PadParam_shared(op_info, inputs, outputs)
                op.private_param = PadParam_private(inputs_sid, outputs_sid)
            else:
                raise("Op not supported")


        for op in operators:
            op.shared_param = shared_param[op.shared_param_id]

        if not os.path.exists(self.include_path):
            os.makedirs(self.include_path)
        if not os.path.exists(self.source_path):
            os.makedirs(self.source_path)
        incfile = includeFile(self.include_path)

        dwconv_incfile = DwconvOpFile(self.TE_Codegen_root, "tiny_dwconv_headers.h")
        dwconv_forward_incfile = DwconvOpFile(self.TE_Codegen_root, "tiny_depthwise_conv.h")
        dwconv_forward_srcfile = DwconvOpFile(self.TE_Codegen_root, "tiny_depthwise_conv.c")

        dwconv_forward_incfile.addDefine(
"""
#ifndef _TINY_DWCONV_H_
#define _TINY_DWCONV_H_
"""
        )

        dwconv_forward_srcfile.addDefine(
            """
#include "arm_nnfunctions.h"
#include "genNN.h"
#include "tinyengine_function.h"
            """
        )
        for op in depthwise_conv_ops:
            depthwise_template = depthwiseInplace(
                        op.kernel_h,
                        op.kernel_w,
                        op.pad_h,
                        op.pad_w,
                        op.stride_h,
                        "CHW",
                        False
                    )

            h = str(op.kernel_h) + 'x' + str(op.kernel_w) + '_stride' + str(op.stride_h)

            dwconv_incfile.addDefine(
                """
#define DWCONV_FN_NAME depthwise_conv_2d_tiny_kernel""" + h +

                """
#define TINY_ENGINE_FN arm_depthwise_conv_s8_tiny_kernel""" + h +
                """
#include "tiny_dwconv_headers.h.inc"
#undef DWCONV_FN_NAME
#undef TINY_ENGINE_FN
                """
            )

            dwconv_forward_srcfile.addDefine(
"""
#define TINY_ENGINE_FORWARD_FN arm_depthwise_conv_s8_tiny_kernel""" + h +
"""
#define TINY_ENGINE_IMPL_FN """ + depthwise_template.genFuncName() +
"""
#include "tiny_depthwise_conv.c.inc"
#undef TINY_ENGINE_FORWARD_FN
#undef TINY_ENGINE_IMPL_FN

"""
            )

            dwconv_forward_incfile.addDefine(
                """
void arm_depthwise_conv_s8_tiny_kernel""" + h + """(
                                     const int8_t* const *input,
                                     const uint16_t input_x,
                                     const uint16_t input_y,
                                     const uint16_t input_ch,
                                     const int8_t *kernel,
                                     const uint16_t output_ch,
                                     const uint16_t kernel_x,
                                     const uint16_t kernel_y,
                                     const uint16_t pad_x,
                                     const uint16_t pad_yt,
                                     const uint16_t pad_yb,
                                     const uint16_t stride_x,
                                     const uint16_t stride_y,
                                     const int32_t *bias,
                                     const int32_t *biasOffset,
                                     int8_t *output,
                                     const int32_t *output_shift,
                                     const int32_t *output_mult,
                                     const uint16_t output_x,
                                     const uint16_t output_y,
                                     const int32_t output_offset,
                                     const int32_t input_offset,
                                     const int32_t output_activation_min,
                                     const int32_t output_activation_max,
                                     const uint16_t dilation_x,
                                     const uint16_t dilation_y,
                                     int16_t *buffer_a);
                """
            )
            depthwise_template.genFile(self.source_path)
            incfile.addDefine(depthwise_template.genFuncDefine())

        dwconv_forward_incfile.addDefine(
"""
#endif
"""
        )

        incfile.writeFile()
        dwconv_incfile.writeFile()
        dwconv_forward_incfile.writeFile()
        dwconv_forward_srcfile.writeFile()
        return tensors, operators, shared_param

    def CalculateOnlineTensorOffset(self):
        from utils import get_aligned_size
        # Step 1: Find activte time of all tensor
        from collections import defaultdict
        requirements = []
        tensor_infos = self.model['subgraphs'][0]['tensors']
        active_time = defaultdict(dict)
        for opid, op_info in enumerate(self.model['subgraphs'][0]['operators']):
            for t_id in op_info['inputs'] + op_info['outputs']:
                t_id_new = self.TensorLUT[t_id]
                t_sid = int(tensor_infos[t_id].get('sid', -1))
                if opid < active_time[(t_id_new, t_sid)].get('first_time_used', len(self.operators)):
                    active_time[(t_id_new, t_sid)]['first_time_used'] = opid
                if opid > active_time[(t_id_new, t_sid)].get('last_time_used', -1):
                    active_time[(t_id_new, t_sid)]['last_time_used'] = opid
            if self.experimental_estimation or self.evict_input:
                if op_info['type'] == 'SPLIT' and self.evict_input:
                    in_tid = op_info['inputs'][1]
                    if in_tid != self.input:
                        import sys
                        print("Split-OP's input is not model input. Turning off evict_in.",
                              file=sys.stderr)
                        self.evict_input = False
                    in_tid_new = self.TensorLUT[in_tid]
                    in_t_sid = int(tensor_infos[in_tid].get('sid', -1))
                    active_time[(in_tid_new, in_t_sid)]['first_time_used'] = -1
                    for out_tid in op_info['outputs']:
                        out_tid_new = self.TensorLUT[out_tid]
                        out_t_sid = int(tensor_infos[out_tid].get('sid', -1))
                        active_time[(out_tid_new, out_t_sid)]['first_time_used'] = -1
                    continue
                elif op_info['type'] == 'ADD' and self.inplace_add:
                    out_tid = op_info['outputs'][0]
                    out_tid_new = self.TensorLUT[out_tid]
                    out_t_sid = int(tensor_infos[out_tid].get('sid', -1))
                    active_time[(out_tid_new, out_t_sid)]['first_time_used'] = -1
                    continue
                elif op_info['type'] == 'DEPTHWISE_CONV_2D' and self.inplace_dw_conv:
                    out_tid = op_info['outputs'][0]
                    out_tid_new = self.TensorLUT[out_tid]
                    out_t_sid = int(tensor_infos[out_tid].get('sid', -1))
                    active_time[(out_tid_new, out_t_sid)]['first_time_used'] = -1
                    continue

        # Step 2: Prepare requirements
        # Step 2-1: generate requirements from tensors
        split_cnt = 0
        for tid, t in enumerate(self.tensors):
            # Offline tensor -> skip
            if t.is_offline():
                continue
            # Online tensor
            else:
                # Online normal tensor
                size = t.dim.N*t.dim.H*t.dim.W*t.dim.C
                # if t.is_online_normal_planned():
                #     print('offline offset for online tensor is currently not supported.')
                #     raise(BaseException('offline offset for online tensor is currently not supported.'))
                if t.is_online_normal_unplanned():
                    sid = -1
                    if self.experimental_estimation and active_time[(tid,sid)]['first_time_used']==-1:
                        active_time[(tid,sid)]['first_time_used'] = 0
                        active_time[(tid,sid)]['last_time_used'] = 0
                        size=0
                    req = Requirements((tid, sid), get_aligned_size(size),
                                        active_time[(tid,sid)]['first_time_used'],
                                        active_time[(tid,sid)]['last_time_used'])
                    requirements.append(req)
                # Online splitted tensor
                elif t.is_splitted():
                    split_cnt += t.split_cnt
                    for sid in range(t.split_cnt):
                        if self.experimental_estimation and active_time[(tid,sid)]['first_time_used']==-1:
                            active_time[(tid,sid)]['first_time_used'] = 0
                            active_time[(tid,sid)]['last_time_used'] = 0
                            size=0
                            req = Requirements((tid, sid), get_aligned_size(size),
                                                active_time[(tid,sid)]['first_time_used'],
                                                active_time[(tid,sid)]['last_time_used'])
                        else:
                            req = Requirements((tid, sid), get_aligned_size(t.get_split_size(sid)),
                                                active_time[(tid,sid)]['first_time_used'],
                                                active_time[(tid,sid)]['last_time_used'])
                        requirements.append(req)
                else:
                    print(t)
                    print('Error: Unknown tensor type.')
                    raise(BaseException('Unknown tensor type.'))
        # Step 2-2: generate requirements from operators(scratch buffer)
        for i, op in enumerate(self.operators):
            if op.private_param is not None:
                handle, buffer_size = op.private_param.RequestScratchBuffer(self.tensors,
                                                    self.op_shared_params[op.shared_param_id])
                if buffer_size > 0:
                    req = Requirements(handle, get_aligned_size(buffer_size), i, i)
                    req.priority = -1
                    requirements.append(req)

        # Step 3: Instantiate memory planner and attach addons, then create plan
        planner = TFLM_Greedy_Planner()
        if len(self.enabled_mem_plan_addons) > 0:
            from MemPlannerAddonAPI import AttachAddonsToPlan
            AttachAddonsToPlan(requirements, self.operators, self.op_shared_params, self.tensors,
                            *self.enabled_mem_plan_addons)
        planner.CreatePlan(requirements)

        # Step 4: Get calculated offset
        split_offset = [-1 for _ in range(split_cnt)]
        requirements, calculated_offset = planner.GetPlannedResult()

        self.mem_profiler.AddMemPlanResult(planner)

        # # Validate result
        # from MemPlannerAddonUtils import isReqOverlapInSpace, isReqOverlapInTime
        # for req_i, offset_i in zip(requirements, calculated_offset):
        #     # Online tensors
        #     if isinstance(req_i.handle, tuple):
        #         tid = req_i.handle[0]
        #         sid = req_i.handle[1]
        #         for req_j, offset_j in zip(requirements, calculated_offset):
        #             if isinstance(req_j.handle, tuple):
        #                 if tid == req_j.handle[0] and sid == req_j.handle[1]:
        #                     continue
        #             if (isReqOverlapInTime(req_i, req_j) and
        #                 isReqOverlapInSpace(req_i, offset_i, req_j, offset_j)):
        #                 print(f'{req_i.handle}: {offset_i}\n{req_j.handle}: {offset_j}\n')
        #                 raise(BaseException("Result has conflict."))

        for i, (req, offset) in enumerate(zip(requirements, calculated_offset)):
            # Online tensors
            if isinstance(req.handle, tuple):
                tid = req.handle[0]
                sid = req.handle[1]
                if sid == -1:
                    self.tensors[tid].splits_offset = offset
                elif sid >= 0:
                    if self.evict_input and req.size == 0:
                        split_offset[self.tensors[tid].splits_offset + sid] = -1
                    else:
                        split_offset[self.tensors[tid].splits_offset + sid] = offset
                else:
                    raise('unknown bug')
            # Op scratch buffers
            elif  issubclass(type(req.handle), OpParam_private):
                req.handle.scratch_buffer_offset = offset
            else:
                raise('unknown bug')

        if self.plot_result:
            planner.PlotResult(self.model_name)
        # print(f'Activation requirement of {self.model_name}: {planner.GetMaximumMemorySize()} B')

        return split_offset, planner.GetMaximumMemorySize()

    def OffloadCalculateOpData(self):
        from utils import get_aligned_data_list
        op_data = []
        for op_param in self.op_shared_params:
            ret = op_param.CalculateOpData(self.tensors)
            if ret is not None:
                op_param.op_data_offset = len(op_data)
                # If the op data contains arrays,
                # ToIntList() impl should also has awareness of alignment.
                op_data += get_aligned_data_list(op_param.op_data.ToIntList(), element_size=1, alignment=self.alignment)
        return op_data

    def PrepareQuantData(self, minimize_flash = True, keep_deprecated_quant_param=False, for_profile=False):
        # find ID of tensors whose zeropoint[] is full of 0.
        normal_tids = []
        all_0_zp_tids = []
        only_all_0_zp_tids = []
        only_zp_tids = []
        for tid, t in enumerate(self.tensors):
            if t.is_bias:
                continue
            elif t.quant_param is not None:
                if (t.quant_param.scale_deprecated and not keep_deprecated_quant_param) and all(v == 0 for v in t.quant_param.quant_zp) and minimize_flash:
                    only_all_0_zp_tids.append(tid)
                elif (t.quant_param.scale_deprecated and not keep_deprecated_quant_param) and minimize_flash:
                    only_zp_tids.append(tid)
                elif all(v == 0 for v in t.quant_param.quant_zp) and minimize_flash:
                    all_0_zp_tids.append(tid)       
                else:
                    normal_tids.append(tid) 
        # Start serialization
        quant_data = {'scale':[], 'zp':[]}
        quant_offset = 0
        # First, not-all-0-zp tensors
        for tid in normal_tids:
            t = self.tensors[tid]
            t.quant_offset = t.quant_offset if for_profile else len(quant_data['scale'])
            quant_data['scale'] += t.quant_param.quant_scale
            quant_data['zp'] += t.quant_param.quant_zp
        # Second, all-0-zp tensors
        all_0_zp_cursor = len(quant_data['zp'])
        for tid in all_0_zp_tids:
            t = self.tensors[tid]
            t.quant_offset = t.quant_offset if for_profile else len(quant_data['scale'])
            quant_data['scale'] += t.quant_param.quant_scale
        # Append sufficient 0s to quant_zp buffer.
        max_0_cnt = max([len(self.tensors[tid].quant_param.quant_zp) for tid in (all_0_zp_tids + only_all_0_zp_tids)]+[0])
        quant_data['zp'] += [0 for _ in range(max_0_cnt)]
        # Third, only-all-0-zp tensors
        for tid in only_all_0_zp_tids:
            t = self.tensors[tid]
            t.quant_offset = t.quant_offset if for_profile else all_0_zp_cursor
        # Fourth, only_zp tensors
        only_zp_cursor = len(quant_data['scale'])
        only_zp_start = len(quant_data['zp'])
        quant_offset = only_zp_cursor
        for tid in only_zp_tids:
            t = self.tensors[tid]
            t.quant_offset = t.quant_offset if for_profile else quant_offset
            quant_offset += len(t.quant_param.quant_zp)
            quant_data['zp'] += t.quant_param.quant_zp

        return quant_data, all_0_zp_cursor, only_zp_cursor, only_zp_start

    def PrepareOfflineTensorData(self):
        from utils import get_aligned_data_list
        tensor_data = []
        for t in self.tensors:
            if len(t.offline_data)>0:
                t.data_offset = len(tensor_data)
                tensor_data += get_aligned_data_list(t.offline_data, element_size=1, alignment=self.alignment)
        return tensor_data

    def GenCxtData(self):
        self.tensor_data = self.PrepareOfflineTensorData()

        quant_data, all_0_zp_cursor, only_zp_cursor, only_zp_start = self.PrepareQuantData()
        self.mem_profiler.SetElementNum('quant_scale_simplified', len(quant_data['scale']))
        self.mem_profiler.SetElementNum('quant_zeropoint_simplified', len(quant_data['zp']))
        quant_data_ori, _, _, _ = self.PrepareQuantData(keep_deprecated_quant_param=True, for_profile=True)
        self.mem_profiler.SetElementNum('quant_scale', len(quant_data_ori['scale']))
        self.mem_profiler.SetElementNum('quant_zeropoint', len(quant_data_ori['zp']))

        with open(f'{self.output_root_dir}/gen_model/include/ctx.h', 'w') as f:
            f.write('#ifndef _CTX_H_\n')
            f.write('#define _CTX_H_\n')
            f.write('#include "gen_lib/include/types.h"\n')
            f.write('#include <stdint.h>\n')
            f.write('\n')
            f.write(f"#define MODEL_NAME \"{self.model_name}\"\n")
            f.write('\n')
            f.write('#ifndef SPLIT_HEIGHT\n')
            f.write(f"#define SPLIT_HEIGHT {self.split_height}\n")
            f.write(f"#endif\n")
            if self.evict_input:
                f.write(f"#define EVICT_IN\n")
            f.write('\n')
            f.write(f'extern int8_t arena[{self.arena_size}];\n')
            f.write('extern const int32_t arena_size;\n')
            f.write('extern const int32_t input_tid;\n')
            f.write('extern const int32_t output_tid;\n')
            if self.evict_input:
                f.write('extern int8_t *model_input_data;\n')
            f.write(f'extern const Tensor tensors[{len(self.tensors)}];\n')
            f.write(f'extern const int all_0_zp_cursor;\n')
            f.write(f'extern const int only_zp_cursor;\n')
            f.write(f'extern const int only_zp_start;\n')
            # f.write(f'extern const int32_t quant_min[{len(quant_data["min"])}];\n')
            # f.write(f'extern const int32_t quant_max[{len(quant_data["max"])}];\n')
            f.write(f'extern const int32_t quant_scale[{len(quant_data["scale"])}];\n')
            f.write(f'extern const int8_t quant_zeropoint[{len(quant_data["zp"])}];\n')
            f.write(f'extern const int32_t split_offset[{len(self.split_offset)}];\n')
            f.write(f'extern const uint8_t offline_tensor_data[{len(self.tensor_data)}];\n')
            f.write('int CtxSummary();\n')
            f.write('#endif\n')
            self.mem_profiler.SetElementNum('Tensor', len(self.tensors))
            self.mem_profiler.SetElementNum('split_offset', len(self.split_offset))
            self.mem_profiler.SetElementNum('offline_tensor_data', len(self.tensor_data))

        with open(f'{self.output_root_dir}/gen_model/src/ctx.cc', 'w') as f:
            required_arena_size = self.arena_size
            out_buf = f"#include \"gen_model/include/ctx.h\"\n#include <cstdio>\n\n"
            out_buf += f"alignas({self.alignment}) int8_t arena[{required_arena_size}];\n"
            out_buf += f"const int32_t arena_size = {required_arena_size};\n"
            out_buf += f"const int32_t input_tid = {self.input};\n"
            out_buf += f"const int32_t output_tid = {self.output};\n"
            if self.evict_input:
                out_buf += f"int8_t *model_input_data;\n"
            f.write(out_buf)
            # Tensor
            out_buf = f'const Tensor tensors[{len(self.tensors)}] = {{\n    '
            for i, t in enumerate(self.tensors):
                out_buf += t.ToCppLiteral()
                if i==len(self.tensors)-1:
                    out_buf += '\n'
                else:
                    out_buf += ', \n    '
            out_buf += "};\n"
            f.write(out_buf)

            from utils import gen_data
            f.write(f'const int all_0_zp_cursor = {all_0_zp_cursor};\n')
            f.write(f'const int only_zp_cursor = {only_zp_cursor};\n')
            f.write(f'const int only_zp_start = {only_zp_start};\n')
            # # QuantData_Min
            # out_buf = f'const int32_t quant_min[{len(quant_data["min"])}] = {{\n    '
            # out_buf += gen_data(quant_data["min"])
            # out_buf += "};\n"
            # f.write(out_buf)
            # # QuantData_Max
            # out_buf = f'const int32_t quant_max[{len(quant_data["max"])}] = {{\n    '
            # out_buf += gen_data(quant_data["max"])
            # out_buf += "};\n"
            # f.write(out_buf)
            # QuantData_Scale
            out_buf = f'const int32_t quant_scale[{len(quant_data["scale"])}] = {{\n    '
            out_buf += gen_data(quant_data["scale"])
            out_buf += "};\n"
            f.write(out_buf)
            # QuantData_ZeroPoint
            out_buf = f'const int8_t quant_zeropoint[{len(quant_data["zp"])}] = {{\n    '
            out_buf += gen_data(quant_data["zp"])
            out_buf += "};\n"
            f.write(out_buf)
            # SplitOffsets
            out_buf = f'const int32_t split_offset[{len(self.split_offset)}] = {{\n    '
            out_buf += gen_data(self.split_offset)
            out_buf += "};\n"
            f.write(out_buf)
            # OfflineTensorData
            out_buf = f'alignas({self.alignment}) const uint8_t offline_tensor_data[{len(self.tensor_data)}] = {{\n    '
            out_buf += gen_data(self.tensor_data)
            out_buf += "};\n"
            f.write(out_buf)
            f.write('\n')
            f.write(r'int CtxSummary(){''\n')
            f.write(r'    printf("Arena Size: %d\n", arena_size);''\n')
            f.write(r'    printf("Tensor Metadata Summary:\n");''\n')
            f.write(r'    printf("\ttensors: %d\n",sizeof(tensors));''\n')
            # f.write(r'    printf("\tquant_min: %d\n",sizeof(quant_min));''\n')
            # f.write(r'    printf("\tquant_max: %d\n",sizeof(quant_max));''\n')
            f.write(f'    printf("\\tquant_scale: %d\\n",{self.mem_profiler.GetDataSize("quant_scale")});\n')
            f.write(f'    printf("\\tquant_zeropoint: %d\\n",{self.mem_profiler.GetDataSize("quant_zeropoint")});\n')
            f.write(r'    printf("\tsplit_offset: %d\n",sizeof(split_offset));''\n')
            f.write(r'    printf("\toffline_tensor_data: %d\n",sizeof(offline_tensor_data));''\n')
            f.write('\n')
            f.write(r'    int byte_tensor =   sizeof(tensors) +''\n')
            # f.write(r'                        sizeof(quant_min) + sizeof(quant_max) +''\n') 
            f.write(f'                        {self.mem_profiler.GetDataSize("quant_scale")} + {self.mem_profiler.GetDataSize("quant_zeropoint")} +\n')
            f.write(r'                        sizeof(split_offset) + sizeof(offline_tensor_data);''\n')
            f.write(r'    return byte_tensor;''\n}\n')

    def GenOpParam(self):
        # OpParam
        # Find used OP type
        op_collection = {}
        for op in self.operators:
            op_collection[op.type]=None
        # classify shared param according to the OP type
        from collections import defaultdict
        shared_param_classified = defaultdict(list)
        ParamID2Classified = [-1 for _ in range(len(self.op_shared_params))]
        for i, param in enumerate(self.op_shared_params):
            key = str(type(param)).lower()
            shared_param_classified[key].append(param)
            ParamID2Classified[i] = len(shared_param_classified[key])-1
        # header
        with open(f'{self.output_root_dir}/gen_model/include/op_param.h', 'w') as f:
            f.write(r'#ifndef _OP_PARAM_H_''\n')
            f.write(r'#define _OP_PARAM_H_''\n')
            f.write(r'#include "gen_lib/include/types.h"''\n')
            f.write('\n')
            f.write('#ifndef SPLIT_HEIGHT\n')
            f.write(f"#define SPLIT_HEIGHT {self.split_height}\n")
            f.write(f"#endif\n")
            # shared params
            for key in sorted(shared_param_classified.keys()):
                f.write(r'extern const 'f'{shared_param_classified[key][0].cpp_typename} {shared_param_classified[key][0].cpp_instance_name}[{len(shared_param_classified[key])}];\n')
                self.mem_profiler.SetElementNum(shared_param_classified[key][0].cpp_typename, len(shared_param_classified[key]))
            # op data
            f.write(f'extern const int32_t op_data[{len(self.op_data)}];\n')
            self.mem_profiler.SetElementNum('OpData', len(self.op_data))
            # summary function
            f.write(r'int OpParamSummary();''\n')
            f.write('#endif\n')
        # source
        with open(f'{self.output_root_dir}/gen_model/src/op_param.cc', 'w') as f:
            # shared params
            f.write('#include "gen_model/include/op_param.h"\n\n')
            for key in sorted(shared_param_classified.keys()):
                param_list = shared_param_classified[key]
                out_buf = r'const 'f'{shared_param_classified[key][0].cpp_typename} {shared_param_classified[key][0].cpp_instance_name}[{len(shared_param_classified[key])}] = {{\n'
                for param in param_list:
                    out_buf += '    ' + param.ToCppLiteral() + ',\n'
                out_buf = out_buf.rstrip(',\n')+'\n};\n'
                f.write(out_buf)
            f.write('\n')

            # op data
            from utils import gen_data
            out_buf = f'alignas({self.alignment}) const int32_t op_data[{len(self.op_data)}] = {{\n    '
            out_buf += gen_data(self.op_data)
            out_buf += "};\n"
            f.write(out_buf)
            f.write('\n')

            # summary function
            f.write('int OpParamSummary(){\n')
            f.write('    printf("Operator parameter Summary:\\n");\n')
            for key in sorted(shared_param_classified.keys()):
                cpp_instance_name = shared_param_classified[key][0].cpp_instance_name
                f.write(f'    printf("\\t{cpp_instance_name}: %d\\n",sizeof({cpp_instance_name}));\n')
            f.write(f'    printf("\\top_data: %d\\n", sizeof(op_data));\n')
            out_buf = '    return '
            indentation = '           '
            for i, key in enumerate(sorted(shared_param_classified.keys())):
                instance_name = shared_param_classified[key][0].cpp_instance_name
                out_buf += f'sizeof({instance_name}) + '
                if i%2 == 1:
                    out_buf += '\n' + indentation
            out_buf = out_buf.rstrip('\n'+indentation).rstrip(' + ') + f' +\n{indentation}sizeof(op_data);\n'
            f.write(out_buf)

            f.write('}')

    def GenEval(self):
        indentation = '    '
        ImplDir = 'gen_lib/OpImpl/'
        # Find used OP
        op_collection = {}
        for op in self.operators:
            op_collection[op.type]=None
        # classify shared param according to the OP type
        from collections import defaultdict
        shared_param_classified = defaultdict(list)
        ParamID2Classified = [-1 for _ in range(len(self.op_shared_params))]
        for i, param in enumerate(self.op_shared_params):
            shared_param_classified[type(param)].append(param)
            ParamID2Classified[i] = len(shared_param_classified[type(param)])-1
        with open(f'{self.output_root_dir}/gen_model/include/eval.h', 'w') as f:
            f.write('\n'.join([
                    "#ifndef _EVAL_H_",
                    "#define _EVAL_H_",
                    "#include <stdint.h>",
                    "void eval(int8_t *input_data);",
                    "#endif"
                ])
            )                 

        with open(f'{self.output_root_dir}/gen_model/src/eval.cc', 'w') as f:
            f.write('#include "gen_model/include/eval.h"\n')
            f.write('#include "gen_model/include/ctx.h"\n')
            f.write('#include "gen_lib/include/ctx_util.h"\n')
            out_buf = ''
            for op_name in sorted(op_collection.keys()):
                out_buf += '#include "'+ImplDir+op_name+'.h"\n'
            f.write(out_buf + '\n')

            f.write('extern "C" {\n')
            f.write('#include "arm_nnfunctions.h"\n')
            f.write('#include "genNN.h"\n')
            f.write('#include "tinyengine_function.h"\n')
            f.write('}\n')

            # Generate eval()
            if self.evict_input:
                f.write("void eval(int8_t *input_data){\n")
                f.write("    model_input_data = input_data;\n")
            else:
                f.write("void eval(int8_t *input_data){\n")
            # f.write("    int8_t local_arena[arena_size];\n") # deprecated
            # f.write("    arena = local_arena;\n") # deprecated
            out_buf = ""
            # TODO modulize this part
            for op in self.operators:
                fn_name = op.GetFnName(self.enable_te, self.tensors)

                out_buf += indentation + fn_name

                if op.type == 'ADD':
                    if op.private_param.output_sid != -1:
                        out_buf += f'({ParamID2Classified[op.shared_param_id]}, {op.private_param.output_sid});\n'
                    else:
                        out_buf += f'({ParamID2Classified[op.shared_param_id]});\n'
                elif op.type == 'AVERAGE_POOL_2D':
                    out_buf += f'({ParamID2Classified[op.shared_param_id]}, { op.private_param.scratch_buffer_offset});\n'
                elif op.type == 'MAX_POOL_2D':
                    out_buf += f'({ParamID2Classified[op.shared_param_id]}, { op.private_param.scratch_buffer_offset});\n'
                elif op.type == 'CONCATENATION':
                    out_buf += f'({ParamID2Classified[op.shared_param_id]});\n'
                elif op.type == 'CONV_2D':
                    if op.private_param.out_split_id != -1:
                        out_buf += f'({ParamID2Classified[op.shared_param_id]}, {op.private_param.out_split_id}, { op.private_param.scratch_buffer_offset});\n'
                    else:
                        out_buf += f'({ParamID2Classified[op.shared_param_id]}, { op.private_param.scratch_buffer_offset});\n'
                elif op.type == 'DEPTHWISE_CONV_2D':
                    if op.private_param.out_split_id != -1:
                        out_buf += f'({ParamID2Classified[op.shared_param_id]}, {op.private_param.out_split_id}, { op.private_param.scratch_buffer_offset});\n'
                    else:
                        out_buf += f'({ParamID2Classified[op.shared_param_id]}, { op.private_param.scratch_buffer_offset});\n'
                elif op.type == 'FULLY_CONNECTED':
                    out_buf += f'({ParamID2Classified[op.shared_param_id]});\n'
                elif op.type == 'RESHAPE':
                    out_buf += f'({ParamID2Classified[op.shared_param_id]});\n'
                elif op.type == 'SOFTMAX':
                    out_buf += f'({ParamID2Classified[op.shared_param_id]});\n'
                elif op.type == 'SPLIT':
                    out_buf += f'({ParamID2Classified[op.shared_param_id]});\n'
                elif op.type == 'PAD':
                    if op.private_param.output_sid != -1:
                        out_buf += f'({ParamID2Classified[op.shared_param_id]}, {op.private_param.output_sid});\n'
                    else:
                        out_buf += f'({ParamID2Classified[op.shared_param_id]});\n'
                else:
                    print('unknown op type')
                    raise('unknown op type')
            f.write(out_buf)
            f.write("}\n")
