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
    conv_2d(0, 0, 5520);
    conv_2d(0, 1, 11040);
    conv_2d(0, 2, 16592);
    conv_2d(0, 3, 22112);
    conv_2d(0, 4, 27632);
    conv_2d(0, 5, 33152);
    conv_2d(0, 6, 38672);
    conv_2d(0, 7, 44192);
    conv_2d(0, 8, 49712);
    conv_2d(0, 9, 55232);
    conv_2d(0, 10, 60752);
    conv_2d(0, 11, 66272);
    conv_2d(0, 12, 71792);
    conv_2d(0, 13, 77312);
    conv_2d(0, 14, 82832);
    conv_2d(0, 15, 88352);
    conv_2d(0, 16, 93872);
    conv_2d(0, 17, 99392);
    conv_2d(0, 18, 104912);
    conv_2d(0, 19, 110432);
    conv_2d(0, 20, 115952);
    conv_2d(0, 21, 121472);
    conv_2d(0, 22, 126992);
    conv_2d(0, 23, 132512);
    conv_2d(0, 24, 135328);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 0, 138016);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 1, 2768);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 2, 5536);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 3, 8304);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 4, 11072);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 5, 13840);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 6, 16608);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 7, 19376);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 8, 22144);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 9, 24912);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 10, 27680);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 11, 30448);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 12, 30448);
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
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 0, 27280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 1, 29760);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 2, 32240);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 3, 34720);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 4, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 5, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 6, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 7, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 8, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 9, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 10, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 11, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 12, 0);
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
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 35984);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 8, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 9, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 10, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 11, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 12, 27680);
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
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 0, 35984);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 1, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 2, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 3, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 4, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 5, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 6, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 7, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 8, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 9, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 10, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 11, 40144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 12, 19376);
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
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 0, 17360);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 1, 17360);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 2, 19840);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 3, 22320);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 4, 24800);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 5, 27280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 6, 29760);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 7, 32240);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 8, 32240);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 9, 32240);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 10, 32240);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 11, 32240);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 12, 9920);
    conv_2d(5, 0, -1);
    conv_2d(5, 1, -1);
    conv_2d(5, 2, -1);
    conv_2d(5, 3, -1);
    conv_2d(5, 4, -1);
    conv_2d(5, 5, -1);
    conv_2d(5, 6, -1);
    conv_2d(5, 7, -1);
    conv_2d(5, 8, -1);
    conv_2d(5, 9, -1);
    conv_2d(5, 10, -1);
    conv_2d(5, 11, -1);
    conv_2d(5, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 0, 32240);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 1, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 2, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 3, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 4, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 5, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 6, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 7, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 8, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 9, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 10, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 11, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 12, 2480);
    conv_2d(6, 0, -1);
    conv_2d(6, 1, -1);
    conv_2d(6, 2, -1);
    conv_2d(6, 3, -1);
    conv_2d(6, 4, -1);
    conv_2d(6, 5, -1);
    conv_2d(6, 6, -1);
    conv_2d(6, 7, -1);
    conv_2d(6, 8, -1);
    conv_2d(6, 9, -1);
    conv_2d(6, 10, -1);
    conv_2d(6, 11, -1);
    conv_2d(6, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(6, 0, 32240);
    depthwise_conv_2d_tiny_kernel3x3_stride1(6, 1, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(6, 2, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(6, 3, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(6, 4, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(6, 5, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(6, 6, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(6, 7, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(6, 8, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(6, 9, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(6, 10, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(6, 11, 35968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(6, 12, 28528);
    conv_2d(7, 0, -1);
    conv_2d(7, 1, -1);
    conv_2d(7, 2, -1);
    conv_2d(7, 3, -1);
    conv_2d(7, 4, -1);
    conv_2d(7, 5, -1);
    conv_2d(7, 6, -1);
    conv_2d(7, 7, -1);
    conv_2d(7, 8, -1);
    conv_2d(7, 9, -1);
    conv_2d(7, 10, -1);
    conv_2d(7, 11, -1);
    conv_2d(7, 12, -1);
    concatenation(0);
    average_pool_2d(0, 31264);
    conv_2d(8, -1);
    reshape(0);
}
