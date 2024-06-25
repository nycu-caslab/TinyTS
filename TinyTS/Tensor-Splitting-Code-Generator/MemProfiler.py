from MemPlanner import TFLM_Greedy_Planner
import copy

ALL_OP_PARAM = {
    'SharedParam_Add': 24,
    'SharedParam_AvgPool': 32,
    'SharedParam_Concat': 12,
    'SharedParam_Conv': 36,
    'SharedParam_Depthwise_Conv': 40,
    'SharedParam_FC': 28,
    'SharedParam_LeakyRelu': -1,
    'SharedParam_MaxPool': 48,
    'SharedParam_Pad': 12,
    'SharedParam_Reshape': 8,
    'SharedParam_Split': 16,
    'SharedParam_Softmax': 12
}

MODEL_SIZE_RELATED_DATA = {
    **{#'key_for_element_num': 'key_for_element_sizeof'
        'Tensor': 'Tensor',
        'quant_scale': 'quant_scale',
        'quant_zeropoint': 'quant_zeropoint',
        'split_offset': 'split_offset',
        'offline_tensor_data': 'offline_tensor_data'}, 
    **{key:key for key in ALL_OP_PARAM.keys()}
}

FLASH_SIZE_RELATED_DATA = {
    **{#'key_for_element_num': 'key_for_element_sizeof'
        'Tensor' : 'Tensor',
        'quant_scale_simplified' : 'quant_scale',
        'quant_zeropoint_simplified' : 'quant_zeropoint',
        'split_offset' : 'split_offset',
        'offline_tensor_data' : 'offline_tensor_data'},
    **{key:key for key in ALL_OP_PARAM.keys()},
    'OpData': 'OpData'
}

DEFAULT_ELEMENT_SIZE_OF = {
    **{'Tensor': 16,
        'quant_scale': 4,
        'quant_zeropoint': 1,
        'split_offset': 4,
        'offline_tensor_data': 1},
    **dict().fromkeys(ALL_OP_PARAM.keys(), ALL_OP_PARAM.values()),
    **{key:val for key, val in ALL_OP_PARAM.items()},
    'OpData': 4
}

DEFAULT_INIT_ELEMENT_NUM = {
    **{ 'Tensor': 0,
        'quant_scale': 0,
        'quant_zeropoint': 0,
        'quant_scale_simplified': 0,
        'quant_zeropoint_simplified': 0,
        'split_offset': 0},
    **dict().fromkeys(ALL_OP_PARAM.keys(), 0),
    'OpData': 0
}

class MemoryProfiler():
    def __init__(self, model_name, 
                element_sizeof = DEFAULT_ELEMENT_SIZE_OF,
                init_element_num = DEFAULT_INIT_ELEMENT_NUM,
                estimated_lib_size = 50*1024):
        self.model_name = model_name
        self.element_sizeof = {**element_sizeof}
        self.element_num = {**init_element_num}
        self.sram = -1
        self.flash = -1
        self.model_size = -1
        self.estimated_lib_size = estimated_lib_size
    
    def SetElementNum(self, key, num):
        self.element_num[key] = num
        self.flash = -1
        if key in MODEL_SIZE_RELATED_DATA:
            self.model_size = -1

    def GetElementNum(self, key):
        return self.element_num[key]
    
    def GetDataSize(self, key):
        if self.element_num[key]>0 and self.element_sizeof[key]==-1:
            raise(BaseException(f'Undefined sizeof for {key}'))
        return self.element_num[key]*self.element_sizeof[key]
    

    def AddMemPlanResult(self, planner: TFLM_Greedy_Planner):
        self.planner = planner
        self.sram = planner.GetMaximumMemorySize()

    def CalculateFlashReq(self):
        if self.flash != -1:
            return
        size = 0
        for key_element_num, key_element_sizeof in FLASH_SIZE_RELATED_DATA.items():
            if self.element_num[key_element_num]>0 and self.element_sizeof[key_element_sizeof]==-1:
                raise(BaseException(f'Undefined sizeof for {key_element_num}'))
            size += self.element_num[key_element_num] * self.element_sizeof[key_element_sizeof]
        self.flash = size + self.estimated_lib_size

    def CalculateModelSize(self):
        if self.model_size != -1:
            return
        size = 0
        for key_element_num, key_element_sizeof in MODEL_SIZE_RELATED_DATA.items():
            if self.element_num[key_element_num]>0 and self.element_sizeof[key_element_sizeof]==-1:
                raise(BaseException(f'Undefined sizeof for {key_element_num}'))
            size += self.element_num[key_element_num] * self.element_sizeof[key_element_sizeof]
        self.model_size = size

    def GetSRAM(self):
        if self.sram == -1:
            raise(BaseException("AddMemPlanResult first!"))
        return self.sram

    def GetFlash(self):
        if self.sram == -1:
            self.CalculateFlashReq()
        return self.sram

    def GetModelSize(self):
        if self.model_size == -1:
            self.CalculateModelSize()
        return self.model_size

    def ReportSRAM(self):
        print(f'SRAM requirement for {self.model_name}: {self.sram:,} B')
    
    def ReportFLASH(self):
        if self.flash == -1:
            self.CalculateFlashReq()
        print(f'RO Data Size of {self.model_name}: {self.flash-self.estimated_lib_size:,} B')
        print(f'Estimated FLASH requirement for {self.model_name}: {self.flash:,} B')

    def ReportModelSize(self):
        if self.model_size == -1:
            self.CalculateModelSize()
        print(f'Model size of {self.model_name}: {self.model_size:,} B')
    
    def Summary(self):
        self.ReportSRAM()
        self.ReportFLASH()
        self.ReportModelSize()
    
    def Summary_verbose(self):
        self.ReportSRAM()
        self.ReportFLASH()
        self.ReportModelSize()
        tmp_dict = {}
        # TODO 印出表格，標示每個ro data的大小與佔比
        for key_element_num, key_element_sizeof in FLASH_SIZE_RELATED_DATA.items():
            if self.element_num[key_element_num]>0 and self.element_sizeof[key_element_sizeof]==-1:
                raise(BaseException(f'Undefined sizeof for {key_element_num}'))
            tmp_dict[key_element_num] = self.element_num[key_element_num] * self.element_sizeof[key_element_sizeof]
        for key, val in tmp_dict.items():
            print(f"{key}: {val}")


