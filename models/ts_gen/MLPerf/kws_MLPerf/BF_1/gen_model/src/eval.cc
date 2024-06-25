#include "gen_model/include/eval.h"
#include "gen_model/include/ctx.h"
#include "gen_lib/include/ctx_util.h"
#include "gen_lib/OpImpl/AVERAGE_POOL_2D.h"
#include "gen_lib/OpImpl/CONCATENATION.h"
#include "gen_lib/OpImpl/CONV_2D.h"
#include "gen_lib/OpImpl/DEPTHWISE_CONV_2D.h"
#include "gen_lib/OpImpl/FULLY_CONNECTED.h"
#include "gen_lib/OpImpl/RESHAPE.h"
#include "gen_lib/OpImpl/SOFTMAX.h"
#include "gen_lib/OpImpl/SPLIT.h"

extern "C" {
#include "arm_nnfunctions.h"
#include "genNN.h"
#include "tinyengine_function.h"
}
void eval(int8_t *input_data){
    split(0);
    conv_2d(0, 0, 320);
    conv_2d(0, 1, 640);
    conv_2d(0, 2, 992);
    conv_2d(0, 3, 1312);
    conv_2d(0, 4, 1632);
    conv_2d(0, 5, 1952);
    conv_2d(0, 6, 2272);
    conv_2d(0, 7, 2592);
    conv_2d(0, 8, 2912);
    conv_2d(0, 9, 3232);
    conv_2d(0, 10, 3552);
    conv_2d(0, 11, 3872);
    conv_2d(0, 12, 4192);
    conv_2d(0, 13, 4512);
    conv_2d(0, 14, 4832);
    conv_2d(0, 15, 5152);
    conv_2d(0, 16, 5472);
    conv_2d(0, 17, 5792);
    conv_2d(0, 18, 6112);
    conv_2d(0, 19, 6432);
    conv_2d(0, 20, 6752);
    conv_2d(0, 21, 7072);
    conv_2d(0, 22, 7392);
    conv_2d(0, 23, 7712);
    conv_2d(0, 24, 8080);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 0, 8320);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 1, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 2, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 3, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 4, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 5, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 6, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 7, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 8, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 9, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 10, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 11, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 12, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 13, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 14, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 15, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 16, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 17, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 18, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 19, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 20, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 21, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 22, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 23, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 24, 8640);
    conv_2d(1, 0, -1);
    conv_2d(1, 1, -1);
    conv_2d(1, 2, -1);
    conv_2d(1, 3, -1);
    conv_2d(1, 4, -1);
    conv_2d(1, 5, -1);
    conv_2d(1, 6, -1);
    conv_2d(1, 7, -1);
    conv_2d(1, 8, -1);
    conv_2d(1, 9, -1);
    conv_2d(1, 10, -1);
    conv_2d(1, 11, -1);
    conv_2d(1, 12, -1);
    conv_2d(1, 13, -1);
    conv_2d(1, 14, -1);
    conv_2d(1, 15, -1);
    conv_2d(1, 16, -1);
    conv_2d(1, 17, -1);
    conv_2d(1, 18, -1);
    conv_2d(1, 19, -1);
    conv_2d(1, 20, -1);
    conv_2d(1, 21, -1);
    conv_2d(1, 22, -1);
    conv_2d(1, 23, -1);
    conv_2d(1, 24, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 0, 8320);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 1, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 2, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 3, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 4, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 5, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 6, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 7, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 8, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 9, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 10, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 11, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 12, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 13, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 14, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 15, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 16, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 17, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 18, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 19, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 20, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 21, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 22, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 23, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 24, 8640);
    conv_2d(2, 0, -1);
    conv_2d(2, 1, -1);
    conv_2d(2, 2, -1);
    conv_2d(2, 3, -1);
    conv_2d(2, 4, -1);
    conv_2d(2, 5, -1);
    conv_2d(2, 6, -1);
    conv_2d(2, 7, -1);
    conv_2d(2, 8, -1);
    conv_2d(2, 9, -1);
    conv_2d(2, 10, -1);
    conv_2d(2, 11, -1);
    conv_2d(2, 12, -1);
    conv_2d(2, 13, -1);
    conv_2d(2, 14, -1);
    conv_2d(2, 15, -1);
    conv_2d(2, 16, -1);
    conv_2d(2, 17, -1);
    conv_2d(2, 18, -1);
    conv_2d(2, 19, -1);
    conv_2d(2, 20, -1);
    conv_2d(2, 21, -1);
    conv_2d(2, 22, -1);
    conv_2d(2, 23, -1);
    conv_2d(2, 24, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 8320);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 8, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 9, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 10, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 11, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 12, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 13, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 14, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 15, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 16, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 17, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 18, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 19, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 20, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 21, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 22, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 23, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 24, 8640);
    conv_2d(3, 0, -1);
    conv_2d(3, 1, -1);
    conv_2d(3, 2, -1);
    conv_2d(3, 3, -1);
    conv_2d(3, 4, -1);
    conv_2d(3, 5, -1);
    conv_2d(3, 6, -1);
    conv_2d(3, 7, -1);
    conv_2d(3, 8, -1);
    conv_2d(3, 9, -1);
    conv_2d(3, 10, -1);
    conv_2d(3, 11, -1);
    conv_2d(3, 12, -1);
    conv_2d(3, 13, -1);
    conv_2d(3, 14, -1);
    conv_2d(3, 15, -1);
    conv_2d(3, 16, -1);
    conv_2d(3, 17, -1);
    conv_2d(3, 18, -1);
    conv_2d(3, 19, -1);
    conv_2d(3, 20, -1);
    conv_2d(3, 21, -1);
    conv_2d(3, 22, -1);
    conv_2d(3, 23, -1);
    conv_2d(3, 24, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 0, 8320);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 1, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 2, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 3, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 4, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 5, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 6, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 7, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 8, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 9, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 10, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 11, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 12, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 13, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 14, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 15, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 16, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 17, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 18, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 19, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 20, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 21, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 22, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 23, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 24, 0);
    conv_2d(4, 0, -1);
    conv_2d(4, 1, -1);
    conv_2d(4, 2, -1);
    conv_2d(4, 3, -1);
    conv_2d(4, 4, -1);
    conv_2d(4, 5, -1);
    conv_2d(4, 6, -1);
    conv_2d(4, 7, -1);
    conv_2d(4, 8, -1);
    conv_2d(4, 9, -1);
    conv_2d(4, 10, -1);
    conv_2d(4, 11, -1);
    conv_2d(4, 12, -1);
    conv_2d(4, 13, -1);
    conv_2d(4, 14, -1);
    conv_2d(4, 15, -1);
    conv_2d(4, 16, -1);
    conv_2d(4, 17, -1);
    conv_2d(4, 18, -1);
    conv_2d(4, 19, -1);
    conv_2d(4, 20, -1);
    conv_2d(4, 21, -1);
    conv_2d(4, 22, -1);
    conv_2d(4, 23, -1);
    conv_2d(4, 24, -1);
    concatenation(0);
    average_pool_2d(0, 8064);
    reshape(0);
    fully_connected(0);
    softmax(0);
}
