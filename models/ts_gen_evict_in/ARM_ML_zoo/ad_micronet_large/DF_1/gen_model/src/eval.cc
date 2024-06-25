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
    model_input_data = input_data;
    split(0);
    conv_2d(0, 0, 8832);
    conv_2d(0, 1, 17664);
    conv_2d(0, 2, 26496);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 0, 30912);
    conv_2d(1, 0, -1);
    conv_2d(0, 3, 8832);
    conv_2d(0, 4, 26496);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 1, 30912);
    conv_2d(1, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 0, 3968);
    conv_2d(2, 0, -1);
    conv_2d(0, 5, 17664);
    conv_2d(0, 6, 30912);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 2, 39296);
    conv_2d(1, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 1, 3968);
    conv_2d(2, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 4416);
    conv_2d(3, 0, -1);
    conv_2d(0, 7, 8832);
    conv_2d(0, 8, 39744);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 3, 48128);
    conv_2d(1, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 2, 3968);
    conv_2d(2, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 4416);
    conv_2d(3, 1, -1);
    conv_2d(0, 9, 17664);
    conv_2d(0, 10, 44160);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 4, 56512);
    conv_2d(1, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 3, 3968);
    conv_2d(2, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 4416);
    conv_2d(3, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 0, 2208);
    conv_2d(4, 0, -1);
    conv_2d(0, 11, 8832);
    conv_2d(0, 12, 26496);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 5, 35328);
    conv_2d(1, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 4, 3968);
    conv_2d(2, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 4416);
    conv_2d(3, 3, -1);
    conv_2d(0, 13, 17664);
    conv_2d(0, 14, 39744);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 6, 52544);
    conv_2d(1, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 5, 3968);
    conv_2d(2, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 4416);
    conv_2d(3, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 1, 2208);
    conv_2d(4, 1, -1);
    conv_2d(0, 15, 8832);
    conv_2d(0, 16, 30912);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 7, 35328);
    conv_2d(1, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 6, 3968);
    conv_2d(2, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 4416);
    conv_2d(3, 5, -1);
    conv_2d(0, 17, 17664);
    conv_2d(0, 18, 35328);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 8, 48576);
    conv_2d(1, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 7, 3968);
    conv_2d(2, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 4416);
    conv_2d(3, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 2, 2208);
    conv_2d(4, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(4, 0, 992);
    conv_2d(5, 0, -1);
    conv_2d(0, 19, 8832);
    conv_2d(0, 20, 26496);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 9, 44160);
    conv_2d(1, 9, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 8, 3968);
    conv_2d(2, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 4416);
    conv_2d(3, 7, -1);
    conv_2d(0, 21, 17664);
    conv_2d(0, 22, 44160);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 10, 56512);
    conv_2d(1, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 9, 3968);
    conv_2d(2, 9, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 8, 4416);
    conv_2d(3, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 3, 2208);
    conv_2d(4, 3, -1);
    conv_2d(0, 23, 8832);
    conv_2d(0, 24, 30912);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 11, 39744);
    conv_2d(1, 11, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 10, 3968);
    conv_2d(2, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 9, 4416);
    conv_2d(3, 9, -1);
    conv_2d(0, 25, 17664);
    conv_2d(0, 26, 39744);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 12, 52544);
    conv_2d(1, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 11, 3968);
    conv_2d(2, 11, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 10, 4416);
    conv_2d(3, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 4, 2208);
    conv_2d(4, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(4, 1, 992);
    conv_2d(5, 1, -1);
    conv_2d(0, 27, 8832);
    conv_2d(0, 28, 26496);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 13, 35328);
    conv_2d(1, 13, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 12, 3968);
    conv_2d(2, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 11, 4416);
    conv_2d(3, 11, -1);
    conv_2d(0, 29, 17664);
    conv_2d(0, 30, 35328);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 14, 48576);
    conv_2d(1, 14, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 13, 3968);
    conv_2d(2, 13, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 12, 4416);
    conv_2d(3, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 5, 2208);
    conv_2d(4, 5, -1);
    conv_2d(0, 31, 30912);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 15, 37312);
    conv_2d(1, 15, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 14, 0);
    conv_2d(2, 14, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 13, 17664);
    conv_2d(3, 13, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 15, 4416);
    conv_2d(2, 15, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 14, 26496);
    conv_2d(3, 14, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 6, 24288);
    conv_2d(4, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(4, 2, 14240);
    conv_2d(5, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 15, 20640);
    conv_2d(3, 15, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 7, 6624);
    conv_2d(4, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(4, 3, 2976);
    conv_2d(5, 3, -1);
    concatenation(0);
    average_pool_2d(0, 4224);
    conv_2d(6, -1);
    reshape(0);
}
