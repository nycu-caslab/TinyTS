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
    reshape(0);
    split(0);
    conv_2d(0, 0, 5520);
    conv_2d(0, 1, 11040);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 0, 13808);
    conv_2d(1, 0, -1);
    conv_2d(0, 2, 11040);
    conv_2d(0, 3, 19328);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 1, 22096);
    conv_2d(1, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 0, 2768);
    conv_2d(2, 0, -1);
    conv_2d(0, 4, 5520);
    conv_2d(0, 5, 24864);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 2, 27632);
    conv_2d(1, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 1, 2768);
    conv_2d(2, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 2768);
    conv_2d(3, 0, -1);
    conv_2d(0, 6, 11040);
    conv_2d(0, 7, 30400);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 3, 33168);
    conv_2d(1, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 2, 2768);
    conv_2d(2, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 2768);
    conv_2d(3, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 0, 2768);
    conv_2d(4, 0, -1);
    conv_2d(0, 8, 5520);
    conv_2d(0, 9, 35936);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 4, 38704);
    conv_2d(1, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 3, 2768);
    conv_2d(2, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 2768);
    conv_2d(3, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 1, 2768);
    conv_2d(4, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 0, 2768);
    conv_2d(5, 0, -1);
    conv_2d(0, 10, 11040);
    conv_2d(0, 11, 41472);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 5, 44240);
    conv_2d(1, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 4, 5536);
    conv_2d(2, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 8304);
    conv_2d(3, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 2, 8304);
    conv_2d(4, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 1, 8304);
    conv_2d(5, 1, -1);
    conv_2d(0, 12, 8304);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 6, 9696);
    conv_2d(1, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 5, 8304);
    conv_2d(2, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 11072);
    conv_2d(3, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 3, 13840);
    conv_2d(4, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 2, 16608);
    conv_2d(5, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 6, 4160);
    conv_2d(2, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 2768);
    conv_2d(3, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 4, 5536);
    conv_2d(4, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 3, 16608);
    conv_2d(5, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 1392);
    conv_2d(3, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 5, 8304);
    conv_2d(4, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 4, 11072);
    conv_2d(5, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 6, 1392);
    conv_2d(4, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 5, 4160);
    conv_2d(5, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 6, 1392);
    conv_2d(5, 6, -1);
    concatenation(0);
    average_pool_2d(0, 18240);
    reshape(1);
    fully_connected(0);
    softmax(0);
}
