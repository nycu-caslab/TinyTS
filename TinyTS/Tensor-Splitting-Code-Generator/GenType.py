class Op:
    def __init__(self):
        self.id = -1
        # self.inputs = [] # deprecated: available in share param
        # self.outputs = [] # deprecated: available in share param
        self.type = ""
        self.shared_param_id = -1
        self.shared_param = None
        self.private_param = None
        return
    
    def GetFnName(self, enable_te, tensors):
        fn_name = self.type.lower()

        if enable_te:
            shared_param = self.shared_param
            if self.type == 'DEPTHWISE_CONV_2D':
                dim_filter = tensors[shared_param.filter].dim
                fn_name += '_tiny_kernel' + str(dim_filter.H) + 'x' + str(dim_filter.W) + '_stride' + str(shared_param.stride_h)
            if self.type == 'CONV_2D':
                dim_input = tensors[shared_param.input].dim
                dim_filter = tensors[shared_param.filter].dim
                pad_H = shared_param.op_data.padding['height']
                pad_W = shared_param.op_data.padding['width']
                if(dim_input.C == 3 and dim_filter.H ==3 and dim_filter.W == 3 and
                    shared_param.stride_h == 2 and shared_param.stride_w == 2 and pad_H == 1 and pad_W ==1):
                    fn_name += '_tiny_3x3_ich3_st2_pad1'
        return fn_name
class TensorDim:
    def __init__(self, list_in=[]):
        self.data = list_in
        if len(list_in) == 4:
            self.N = list_in[0]
            self.H = list_in[1]
            self.W = list_in[2]
            self.C = list_in[3]
        elif len(list_in) == 1:
            self.N = self.W = self.C = 1
            self.H = list_in[0]
        elif len(list_in) == 0:
            self.N = self.H = self.W = self.C = 1
        else:
            while len(list_in) != 4:
                list_in.append(1)
            self.N = list_in[0]
            self.H = list_in[1]
            self.W = list_in[2]
            self.C = list_in[3]
    def concat(self, other, dim):
        if ((self.N != other.N and dim!=0) or
            (self.H != other.H and dim!=1) or
            (self.W != other.W and dim!=2) or
            (self.C != other.C and dim!=3)):
            print(self, other)
            raise("DIM error!")
        if dim == 0:
            self.N += other.N
        elif dim == 1:
            self.H += other.H
        elif dim == 2:
            self.W += other.W
        elif dim == 3:
            self.C += other.C
        else:
            print("input dim:", dim)
            raise("concat dim number error")
    def __str__(self) -> str:
        return f'{{N: {self.N}, H: {self.H}, W: {self.W}, C: {self.C}}}'

class QuantParam:
    def __init__(self, quant_min=None, quant_max=None, quant_scale=None, quant_zp=None):
        self.quant_min   = quant_min if quant_min is not None else []
        self.quant_max   = quant_max if quant_max is not None else []
        self.quant_scale = quant_scale if quant_scale is not None else []
        self.quant_zp    = quant_zp if quant_zp is not None else []
        self.scale_deprecated = False

class Tensor():
    def __init__(self):
        self.dim = TensorDim()
        self.type = ""
        self.quant_offset = -1
        # 0: a splitted tensor
        self.data_offset = -1
        self.splits_offset = []
        self.visited = False
        self.quant_param = QuantParam()
        self.splitted = False
        self.is_bias = False
    def fill(self,  dim:TensorDim, type:str, offline_data:list, 
                    quant_offset:int, data_offset:int, splits_offset:int, 
                    split_cnt:int, split_height:int, quant_param:QuantParam):
        self.dim = dim
        self.type = type
        self.quant_offset = quant_offset
        self.data_offset = data_offset
        self.offline_data = offline_data
        self.splits_offset = splits_offset
        self.split_cnt = split_cnt
        self.split_height = split_height
        self.quant_param = quant_param
        self.visited = True
    def is_offline(self):
        return self.data_offset >= 0 and self.splits_offset == -1
    def is_online(self):
        return self.data_offset < 0
    def is_online_normal_unplanned(self):
        return self.data_offset == -1 and self.splits_offset == -1
    def is_splitted(self):
        return self.splitted
    def get_split_size(self, sid):
        row_size = self.dim.W * self.dim.C
        if self.split_cnt*self.split_height == self.dim.H or sid < self.split_cnt-1:
            return row_size*self.split_height
        else:
            return row_size * (self.dim.H%self.split_height)
    def ToCppLiteral(self):
        return f"{{{{{self.dim.N}, {self.dim.H}, {self.dim.W}, {self.dim.C}}}, {self.data_offset}, {self.splits_offset}, {self.quant_offset}}}"
    def __str__(self) -> str:
        return f'{{dim: {self.dim}, type: {self.type}, quant_offset: {self.quant_offset}, data_offset: {self.data_offset}, splits_offset: {self.splits_offset}, split_cnt: {self.split_cnt}, split_height: {self.split_height}, quant_param: {self.quant_param}, visited: {self.visited}'
