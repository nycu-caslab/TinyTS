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
    conv_2d(0, 0, 8832);
    conv_2d(0, 1, 17696);
    conv_2d(0, 2, 26528);
    conv_2d(0, 3, 35360);
    conv_2d(0, 4, 44192);
    conv_2d(0, 5, 53024);
    conv_2d(0, 6, 61856);
    conv_2d(0, 7, 70688);
    conv_2d(0, 8, 79520);
    conv_2d(0, 9, 88352);
    conv_2d(0, 10, 97184);
    conv_2d(0, 11, 106016);
    conv_2d(0, 12, 114848);
    conv_2d(0, 13, 123680);
    conv_2d(0, 14, 132512);
    conv_2d(0, 15, 141344);
    conv_2d(0, 16, 150176);
    conv_2d(0, 17, 159008);
    conv_2d(0, 18, 167840);
    conv_2d(0, 19, 176672);
    conv_2d(0, 20, 185504);
    conv_2d(0, 21, 194336);
    conv_2d(0, 22, 203168);
    conv_2d(0, 23, 212000);
    conv_2d(0, 24, 220832);
    conv_2d(0, 25, 229664);
    conv_2d(0, 26, 238496);
    conv_2d(0, 27, 247328);
    conv_2d(0, 28, 256160);
    conv_2d(0, 29, 264992);
    conv_2d(0, 30, 273824);
    conv_2d(0, 31, 282688);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 0, 287040);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 1, 4416);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 2, 8832);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 3, 13248);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 4, 17664);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 5, 22080);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 6, 26496);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 7, 30912);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 8, 35328);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 9, 39744);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 10, 44160);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 11, 48576);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 12, 52992);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 13, 57408);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 14, 61824);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 15, 66240);
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
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 0, 59520);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 1, 63488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 2, 67456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 3, 71424);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 4, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 5, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 6, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 7, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 8, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 9, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 10, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 11, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 12, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 13, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 14, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 15, 0);
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
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 75072);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 8, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 9, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 10, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 11, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 12, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 13, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 14, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 15, 79488);
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
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 0, 59616);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 1, 61824);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 2, 2208);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 3, 4416);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 4, 6624);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 5, 8832);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 6, 11040);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 7, 13248);
    conv_2d(4, 0, -1);
    conv_2d(4, 1, -1);
    conv_2d(4, 2, -1);
    conv_2d(4, 3, -1);
    conv_2d(4, 4, -1);
    conv_2d(4, 5, -1);
    conv_2d(4, 6, -1);
    conv_2d(4, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(4, 0, 10912);
    depthwise_conv_2d_tiny_kernel3x3_stride2(4, 1, 11904);
    depthwise_conv_2d_tiny_kernel3x3_stride2(4, 2, 992);
    depthwise_conv_2d_tiny_kernel3x3_stride2(4, 3, 1984);
    conv_2d(5, 0, -1);
    conv_2d(5, 1, -1);
    conv_2d(5, 2, -1);
    conv_2d(5, 3, -1);
    concatenation(0);
    average_pool_2d(0, 4224);
    conv_2d(6, -1);
    reshape(0);
}
