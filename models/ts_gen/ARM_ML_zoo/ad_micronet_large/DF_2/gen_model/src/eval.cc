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
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 0, 62080);
    conv_2d(1, 0, -1);
    conv_2d(0, 3, 17664);
    conv_2d(0, 4, 52992);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 1, 62080);
    conv_2d(1, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 0, 7936);
    conv_2d(2, 0, -1);
    conv_2d(0, 5, 35328);
    conv_2d(0, 6, 61824);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 2, 78592);
    conv_2d(1, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 1, 7936);
    conv_2d(2, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 8832);
    conv_2d(3, 0, -1);
    conv_2d(0, 7, 17664);
    conv_2d(0, 8, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 3, 96384);
    conv_2d(1, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 2, 7936);
    conv_2d(2, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 8832);
    conv_2d(3, 1, -1);
    conv_2d(0, 9, 35328);
    conv_2d(0, 10, 88320);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 4, 113024);
    conv_2d(1, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 3, 7936);
    conv_2d(2, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 8832);
    conv_2d(3, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 0, 4416);
    conv_2d(4, 0, -1);
    conv_2d(0, 11, 17664);
    conv_2d(0, 12, 52992);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 5, 70656);
    conv_2d(1, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 4, 7936);
    conv_2d(2, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 8832);
    conv_2d(3, 3, -1);
    conv_2d(0, 13, 35328);
    conv_2d(0, 14, 79488);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 6, 105088);
    conv_2d(1, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 5, 7936);
    conv_2d(2, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 8832);
    conv_2d(3, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 1, 4416);
    conv_2d(4, 1, -1);
    conv_2d(0, 15, 61824);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 7, 74624);
    conv_2d(1, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 6, 0);
    conv_2d(2, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 35328);
    conv_2d(3, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 7, 8832);
    conv_2d(2, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 52992);
    conv_2d(3, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 2, 48576);
    conv_2d(4, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(4, 0, 28480);
    conv_2d(5, 0, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 41280);
    conv_2d(3, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 3, 13248);
    conv_2d(4, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(4, 1, 5952);
    conv_2d(5, 1, -1);
    concatenation(0);
    average_pool_2d(0, 4224);
    conv_2d(6, -1);
    reshape(0);
}
