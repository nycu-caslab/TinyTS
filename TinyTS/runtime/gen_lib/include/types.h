#ifndef GEN_INCLUDE_TYPE_H_
#define GEN_INCLUDE_TYPE_H_

#include "gen_lib/include/builtin_op_data.h"
#include "gen_lib/include/common.h"
#include "gen_lib/include/compatibility.h"


typedef struct SharedParam_Conv {
    int input;
    int filter;
    int bias;
    int output;

    int padding;// 0:same, 1:valid
    int activation;
    int op_data_offset;

    // TfLiteConvParams - in TensorNode
    int16 stride_width;
    int16 stride_height;
    int16 dilation_width_factor;
    int16 dilation_height_factor;
}SharedParam_Conv;

typedef struct SharedParam_Pad{
    int input;
    int output;
    int padding;
}SharedParam_Pad;

typedef struct SharedParam_Add{
    int input_A;
    int input_B;
    int output;
    int op_data_offset;
    int fused_ActFunc;
    bool pot_scale_int16;
}SharedParam_Add;

typedef struct SharedParam_AvgPool{
    int input;
    int output;
    int filter_height;
    int filter_width;
    int stride_h;
    int stride_w;
    int padding;
    int fused_ActFunc;

}SharedParam_AvgPool;

typedef struct SharedParam_MaxPool{
    int input;
    int output;
    int filter_height;
    int filter_width;
    int stride_h;
    int stride_w;
    int padding;
    int fused_ActFunc;
    int32 quantized_activation_min;
    int32 quantized_activation_max;
    int padding_width;
    int padding_height;

}SharedParam_MaxPool;

typedef struct SharedParam_Reshape{
    int input;
    int output;
}SharedParam_Reshape;

typedef struct SharedParam_FC{
    int input;
    int weight;
    int bias;
    int output;
    int fused_ActFunc;
    int weights_format;
    bool asymmetric_quant_input;
    bool keep_num_dims;
}SharedParam_FC;

typedef struct SharedParam_Softmax{
    int input;
    int output;
    int beta;
}SharedParam_Softmax;

typedef struct SharedParam_Split{
    int input;
    int output;
    int num_splits;
    int axis;
}SharedParam_Split;

typedef struct SharedParam_Concat{
    int input;
    int output;
    int axis;
}SharedParam_Concat;

typedef struct SharedParam_Depthwise_Conv{
    int tid_input;
    int tid_filter;
    int tid_bias;
    int tid_output;

    int depth_multiplier;
    int padding;// 0:same, 1:valid
    int activation;
    int op_data_offset;

    int16 stride_width;
    int16 stride_height;
    int16 dilation_width_factor;
    int16 dilation_height_factor;
}SharedParam_Depthwise_Conv;

#endif