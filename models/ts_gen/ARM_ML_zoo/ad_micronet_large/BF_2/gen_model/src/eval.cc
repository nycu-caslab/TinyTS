#include "gen_model/include/eval.h"
#include "gen_model/include/ctx.h"
#include "gen_lib/include/ctx_util.h"
#include "gen_lib/OpImpl/AVERAGE_POOL_2D.h"
#include "gen_lib/OpImpl/CONCATENATION.h"
#include "gen_lib/OpImpl/CONV_2D.h"
#include "gen_lib/OpImpl/DEPTHWISE_CONV_2D.h"
#include "gen_lib/OpImpl/RESHAPE.h"
#include "gen_lib/OpImpl/SPLIT.h"

extern "C" {
#include "arm_nnfunctions.h"
#include "genNN.h"
#include "tinyengine_function.h"
}
void eval(int8_t *input_data){
    split(0);
    conv_2d(0, 0, 17664);
    conv_2d(0, 1, 35392);
    conv_2d(0, 2, 53056);
    conv_2d(0, 3, 70720);
    conv_2d(0, 4, 88384);
    conv_2d(0, 5, 106048);
    conv_2d(0, 6, 123712);
    conv_2d(0, 7, 141376);
    conv_2d(0, 8, 159040);
    conv_2d(0, 9, 176704);
    conv_2d(0, 10, 194368);
    conv_2d(0, 11, 212032);
    conv_2d(0, 12, 229696);
    conv_2d(0, 13, 247360);
    conv_2d(0, 14, 265024);
    conv_2d(0, 15, 282752);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 0, 291456);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 1, 8832);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 2, 17664);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 3, 26496);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 4, 35328);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 5, 44160);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 6, 52992);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 7, 61824);
    conv_2d(1, 0, -1);
    conv_2d(1, 1, -1);
    conv_2d(1, 2, -1);
    conv_2d(1, 3, -1);
    conv_2d(1, 4, -1);
    conv_2d(1, 5, -1);
    conv_2d(1, 6, -1);
    conv_2d(1, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 0, 55552);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 1, 55552);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 2, 63488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 3, 71424);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 4, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 5, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 6, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 7, 0);
    conv_2d(2, 0, -1);
    conv_2d(2, 1, -1);
    conv_2d(2, 2, -1);
    conv_2d(2, 3, -1);
    conv_2d(2, 4, -1);
    conv_2d(2, 5, -1);
    conv_2d(2, 6, -1);
    conv_2d(2, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 88320);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 88320);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 88320);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 88320);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 88320);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 88320);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 88320);
    conv_2d(3, 0, -1);
    conv_2d(3, 1, -1);
    conv_2d(3, 2, -1);
    conv_2d(3, 3, -1);
    conv_2d(3, 4, -1);
    conv_2d(3, 5, -1);
    conv_2d(3, 6, -1);
    conv_2d(3, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 0, 48576);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 1, 52992);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 2, 4416);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 3, 8832);
    conv_2d(4, 0, -1);
    conv_2d(4, 1, -1);
    conv_2d(4, 2, -1);
    conv_2d(4, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(4, 0, 5952);
    depthwise_conv_2d_tiny_kernel3x3_stride2(4, 1, 7936);
    conv_2d(5, 0, -1);
    conv_2d(5, 1, -1);
    concatenation(0);
    average_pool_2d(0, 4224);
    conv_2d(6, -1);
    reshape(0);
}
