import tflite

class MAC_calculator():
    def __init__(self, model:tflite.Model, silent: bool = False) -> None:
        self.silent = silent
        self.model = model
        self.graph = self.model.Subgraphs(0)
        self.conv_id = 0
        self.dwconv_id = 0
        self.fc_id = 0
    
    def CalcMAC(self):
        tot_mac = 0
        ops_len = self.graph.OperatorsLength()
        ops = [self.graph.Operators(i) for i in range(ops_len)]
        for op in ops:
            op_code = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            if op_code == tflite.BuiltinOperator.CONV_2D:
                mac = self.CONV_2D(op)
            elif op_code == tflite.BuiltinOperator.DEPTHWISE_CONV_2D:
                mac = self.DEPTHWISE_CONV_2D(op)
            elif op_code == tflite.BuiltinOperator.FULLY_CONNECTED:
                mac = self.FULLY_CONNECTED(op)
            else:
                mac = 0
            tot_mac += mac
        return tot_mac
        
    def CONV_2D(self, op: tflite.Operator):
        filter_shape = self.graph.Tensors(op.Inputs(1)).ShapeAsNumpy()
        output_shape = self.graph.Tensors(op.Outputs(0)).ShapeAsNumpy()

        mac = ((output_shape[1] * output_shape[2] * output_shape[3]) * 
               (filter_shape[1] * filter_shape[2] * filter_shape[3]))
        if not self.silent:
            print(f"CONV_2D{self.conv_id}: {mac}")
        self.conv_id += 1
        return mac
    
    def DEPTHWISE_CONV_2D(self, op: tflite.Operator):
        filter_shape = self.graph.Tensors(op.Inputs(1)).ShapeAsNumpy()
        output_shape = self.graph.Tensors(op.Outputs(0)).ShapeAsNumpy()

        mac = ((output_shape[1] * output_shape[2] * output_shape[3]) * 
               (filter_shape[1] * filter_shape[2]))
        if not self.silent:
            print(f"DEPTHWISE_CONV_2D{self.dwconv_id}: {mac}")
        self.dwconv_id += 1
        return mac
    
    def FULLY_CONNECTED(self, op: tflite.Operator):
        filter_shape = self.graph.Tensors(op.Inputs(1)).ShapeAsNumpy()

        mac = ((filter_shape[0] * filter_shape[1]))
        if not self.silent:
            print(f"FULLY_CONNECTED{self.fc_id}: {mac}")
        self.fc_id += 1
        return mac


