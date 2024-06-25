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
    conv_2d(0, 0, 2768);
    conv_2d(0, 1, 5536);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 0, 6928);
    conv_2d(1, 0, -1);
    conv_2d(0, 2, 5536);
    conv_2d(0, 3, 9696);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 1, 11088);
    conv_2d(1, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 0, 1392);
    conv_2d(2, 0, -1);
    conv_2d(0, 4, 2768);
    conv_2d(0, 5, 12480);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 2, 13872);
    conv_2d(1, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 1, 1392);
    conv_2d(2, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 1392);
    conv_2d(3, 0, -1);
    conv_2d(0, 6, 5536);
    conv_2d(0, 7, 15264);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 3, 16656);
    conv_2d(1, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 2, 1392);
    conv_2d(2, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 1392);
    conv_2d(3, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 0, 1392);
    conv_2d(4, 0, -1);
    conv_2d(0, 8, 2768);
    conv_2d(0, 9, 18048);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 4, 19440);
    conv_2d(1, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 3, 1392);
    conv_2d(2, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 1392);
    conv_2d(3, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 1, 1392);
    conv_2d(4, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 0, 1392);
    conv_2d(5, 0, -1);
    conv_2d(0, 10, 5536);
    conv_2d(0, 11, 20832);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 5, 22224);
    conv_2d(1, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 4, 1392);
    conv_2d(2, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 1392);
    conv_2d(3, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 2, 1392);
    conv_2d(4, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 1, 1392);
    conv_2d(5, 1, -1);
    conv_2d(0, 12, 2768);
    conv_2d(0, 13, 15264);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 6, 23616);
    conv_2d(1, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 5, 1392);
    conv_2d(2, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 1392);
    conv_2d(3, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 3, 1392);
    conv_2d(4, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 2, 1392);
    conv_2d(5, 2, -1);
    conv_2d(0, 14, 5536);
    conv_2d(0, 15, 12480);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 7, 25008);
    conv_2d(1, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 6, 1392);
    conv_2d(2, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 1392);
    conv_2d(3, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 4, 1392);
    conv_2d(4, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 3, 1392);
    conv_2d(5, 3, -1);
    conv_2d(0, 16, 2768);
    conv_2d(0, 17, 9696);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 8, 26400);
    conv_2d(1, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 7, 1392);
    conv_2d(2, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 1392);
    conv_2d(3, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 5, 1392);
    conv_2d(4, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 4, 1392);
    conv_2d(5, 4, -1);
    conv_2d(0, 18, 5536);
    conv_2d(0, 19, 8304);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 9, 27792);
    conv_2d(1, 9, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 8, 1392);
    conv_2d(2, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 1392);
    conv_2d(3, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 6, 1392);
    conv_2d(4, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 5, 1392);
    conv_2d(5, 5, -1);
    conv_2d(0, 20, 2768);
    conv_2d(0, 21, 11088);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 10, 29184);
    conv_2d(1, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 9, 1392);
    conv_2d(2, 9, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 8, 1392);
    conv_2d(3, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 7, 1392);
    conv_2d(4, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 6, 1392);
    conv_2d(5, 6, -1);
    conv_2d(0, 22, 5536);
    conv_2d(0, 23, 13872);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 11, 30576);
    conv_2d(1, 11, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 10, 1392);
    conv_2d(2, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 9, 1392);
    conv_2d(3, 9, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 8, 1392);
    conv_2d(4, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 7, 1392);
    conv_2d(5, 7, -1);
    conv_2d(0, 24, 4160);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 12, 4160);
    conv_2d(1, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 11, 1392);
    conv_2d(2, 11, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 10, 1392);
    conv_2d(3, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 9, 1392);
    conv_2d(4, 9, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 8, 1392);
    conv_2d(5, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 12, 1392);
    conv_2d(2, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 11, 2784);
    conv_2d(3, 11, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 10, 4176);
    conv_2d(4, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 9, 4176);
    conv_2d(5, 9, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 12, 4176);
    conv_2d(3, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 11, 4176);
    conv_2d(4, 11, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 10, 5568);
    conv_2d(5, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 12, 5568);
    conv_2d(4, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 11, 2784);
    conv_2d(5, 11, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 12, 2784);
    conv_2d(5, 12, -1);
    concatenation(0);
    average_pool_2d(0, 18240);
    reshape(1);
    fully_connected(0);
    softmax(0);
}
