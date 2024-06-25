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
    model_input_data = input_data;
    split(0);
    conv_2d(0, 0, 640);
    conv_2d(0, 1, 1280);
    conv_2d(0, 2, 1920);
    conv_2d(0, 3, 2560);
    conv_2d(0, 4, 3200);
    conv_2d(0, 5, 3840);
    conv_2d(0, 6, 4480);
    conv_2d(0, 7, 5120);
    conv_2d(0, 8, 5760);
    conv_2d(0, 9, 6400);
    conv_2d(0, 10, 7040);
    conv_2d(0, 11, 7680);
    conv_2d(0, 12, 7680);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 0, 8320);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 1, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 2, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 3, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 4, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 5, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 6, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 7, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 8, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 9, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 10, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 11, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 12, 6400);
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
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 0, 8320);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 1, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 2, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 3, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 4, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 5, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 6, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 7, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 8, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 9, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 10, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 11, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 12, 4480);
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
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 8320);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 8, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 9, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 10, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 11, 9280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 12, 2560);
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
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 0, 8320);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 1, 8960);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 2, 8960);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 3, 8960);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 4, 8960);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 5, 8960);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 6, 8960);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 7, 8960);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 8, 8960);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 9, 8960);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 10, 8960);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 11, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 12, 0);
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
    concatenation(0);
    average_pool_2d(0, 8064);
    reshape(0);
    fully_connected(0);
    softmax(0);
}
