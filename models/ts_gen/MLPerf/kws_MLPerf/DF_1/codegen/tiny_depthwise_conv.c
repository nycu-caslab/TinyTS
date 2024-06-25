
#include "arm_nnfunctions.h"
#include "genNN.h"
#include "tinyengine_function.h"
            
#define TINY_ENGINE_FORWARD_FN arm_depthwise_conv_s8_tiny_kernel3x3_stride1
#define TINY_ENGINE_IMPL_FN depthwise_kernel3x3_stride1_inplace_CHW
#include "tiny_depthwise_conv.c.inc"
#undef TINY_ENGINE_FORWARD_FN
#undef TINY_ENGINE_IMPL_FN

