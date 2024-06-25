def int32_clip(a):
    if a < -(2**31):
        return -(2**31)
    elif a > 2**31 - 1:
        return 2**31 - 1
    return a.astype(int)

def ComputeOffsetBias(zp, weight, channel):
    offsetBias = []

    kernelsize = int(len(weight) / channel)

    # fuse the offset into bias
    for i in range(channel):
        tmpW = 0
        for j in range(kernelsize):
            tmpW += weight[j * channel + i]
        offsetBias.append(int32_clip(tmpW * (-zp[0])))

    # OffsetRBias
    # string = f"{const_str}int32_t offsetRBias" + str(Lindex) + "[" + str(len(bias)) + "] = {"
    #     fp.write(string)
    #     kernelsize = int(len(weight) / channel)
    #     for i in range(channel):
    #         tmpW = 0
    #         for j in range(kernelsize):
    #             tmpW += weight[j * channel + i]
    #         fp.write(str(bias[i] + tmpW * input_offset - self.int32_clip(bias[i] + tmpW * input_offset)) + ", ")
    #     fp.write("};\n")

    return offsetBias


def ComputeContributions(output_channel, input_channel, filter, zp):
    contribs = []

    for oc in range(0, output_channel):
        contrib = 0
        for ic in range(0, input_channel):
            contrib += filter[oc * input_channel + ic]
        contrib *= -zp[0]
        contribs.append(contrib)

    return contribs
def ByteToIntList(byte_data):
    return [int.from_bytes(byte_data[4*i:4*(i+1)], 'little', signed=True)
                                        for i in range(len(byte_data)//4)]

def ComputePaddingWithOffset(stride, dilation_rate, in_size, filter_size, out_size)->int:
    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    total_padding = ((out_size - 1) * stride + effective_filter_size - in_size)
    total_padding = total_padding if total_padding > 0 else 0
    return total_padding // 2

def CalculateActivationRangeQuantized(act_func, scale:int, zero_point:int):
    # sacle is actually float, not an integer
    # so reinterpret_cast it to a float
    from struct import pack, unpack
    scale_casted = unpack('f',pack('i', scale))[0]
    qmin = -128
    qmax = 127
    quantize = lambda f: zero_point + round(f / scale_casted, 0)
    act_min = qmin
    act_max = qmax
    if (act_func == 1 or act_func=='RELU'): # kTfLiteActRelu
        act_min = max(qmin, quantize(0.0))
        act_max = qmax
    elif (act_func == 2 or act_func=='RELU_N1_TO_1'): # kTfLiteActReluN1To1
        act_min = max(qmin, quantize(-1.0))
        act_max = min(qmax, quantize(1.0))
    elif (act_func == 3 or act_func=='RELU6'): # kTfLiteActRelu6
        act_min = max(qmin, quantize(0.0))
        act_max = min(qmax, quantize(6.0))

    return act_min, act_max

def gen_data(data_list):
    out_buf = ""
    for i, data in enumerate(data_list):
        out_buf += f'{data}'
        if i==len(data_list)-1:
            out_buf += '\n'
        else:
            out_buf += ', '
    return out_buf

def get_aligned_size(size, alignment=16, type='INT8'):
    if type == 'INT8':
        type_multiplier = 1
    elif type == 'FLOAT32':
        type_multiplier = 4
    else:
        print("Unknown type, should be one of [FLOAT32, INT8]")
        raise("Input Type Error")
    return ((size*type_multiplier-1)//alignment+1)*alignment

def get_aligned_data_list(list_in, element_size, alignment=16):
    list_len = len(list_in)
    size_ori = list_len*element_size
    size_aligned = get_aligned_size(size_ori, alignment=alignment)
    return list_in + [0 for _ in range(size_aligned - size_ori)]

def GetGeneralSplitSize(tensor, split_height):
    dim = tensor.dim
    row_size = dim.W*dim.C
    return split_height*row_size

def GetTrailingSplitSize(tensor, split_height):
    dim = tensor.dim
    row_size = dim.W*dim.C
    return split_height*row_size if (dim.H%split_height)==0 else (dim.H%split_height)*row_size

def GetSplitHeightBySid(tensor, sid):
    dim = tensor.dim
    row_size = dim.W*dim.C
    if sid == tensor.split_cnt-1:
        return GetTrailingSplitSize(tensor, tensor.split_height)
    else:
        return GetGeneralSplitSize(tensor, tensor.split_height)

def GetGeneralSplitHeight(tensor, split_height):
    dim = tensor.dim
    row_size = dim.W*dim.C
    return split_height

def GetTrailingSplitHeight(tensor, split_height):
    dim = tensor.dim
    return split_height if (dim.H%split_height)==0 else (dim.H%split_height)

def GetSplitHeightBySid(tensor, sid):
    dim = tensor.dim
    row_size = dim.W*dim.C
    if sid == tensor.split_cnt-1:
        return GetTrailingSplitHeight(tensor, tensor.split_height)
    else:
        return GetGeneralSplitHeight(tensor, tensor.split_height)
