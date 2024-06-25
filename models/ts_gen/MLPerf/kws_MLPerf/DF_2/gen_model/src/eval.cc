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
    conv_2d(0, 0, 640);
    conv_2d(0, 1, 1344);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 0, 1920);
    conv_2d(1, 0, -1);
    conv_2d(0, 2, 2624);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 1, 3200);
    conv_2d(1, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 0, 3200);
    conv_2d(2, 0, -1);
    conv_2d(0, 3, 3904);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 2, 4480);
    conv_2d(1, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 1, 4480);
    conv_2d(2, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 4480);
    conv_2d(3, 0, -1);
    conv_2d(0, 4, 5184);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 3, 5760);
    conv_2d(1, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 2, 5760);
    conv_2d(2, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 5120);
    conv_2d(3, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 0, 0);
    conv_2d(4, 0, -1);
    conv_2d(0, 5, 6464);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 4, 7040);
    conv_2d(1, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 3, 7040);
    conv_2d(2, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 6400);
    conv_2d(3, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 1, 640);
    conv_2d(4, 1, -1);
    conv_2d(0, 6, 6400);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 5, 7680);
    conv_2d(1, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 4, 7680);
    conv_2d(2, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 6400);
    conv_2d(3, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 2, 1280);
    conv_2d(4, 2, -1);
    conv_2d(0, 7, 6400);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 6, 8320);
    conv_2d(1, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 5, 8320);
    conv_2d(2, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 6400);
    conv_2d(3, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 3, 7040);
    conv_2d(4, 3, -1);
    conv_2d(0, 8, 2560);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 7, 8960);
    conv_2d(1, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 6, 8960);
    conv_2d(2, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 8960);
    conv_2d(3, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 4, 2560);
    conv_2d(4, 4, -1);
    conv_2d(0, 9, 8960);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 8, 9600);
    conv_2d(1, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 7, 8960);
    conv_2d(2, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 8960);
    conv_2d(3, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 5, 3200);
    conv_2d(4, 5, -1);
    conv_2d(0, 10, 8960);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 9, 10240);
    conv_2d(1, 9, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 8, 10240);
    conv_2d(2, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 8960);
    conv_2d(3, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 6, 3840);
    conv_2d(4, 6, -1);
    conv_2d(0, 11, 4480);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 10, 10880);
    conv_2d(1, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 9, 10880);
    conv_2d(2, 9, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 8, 10880);
    conv_2d(3, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 7, 4480);
    conv_2d(4, 7, -1);
    conv_2d(0, 12, 5120);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 11, 8320);
    conv_2d(1, 11, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 10, 8320);
    conv_2d(2, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 9, 8320);
    conv_2d(3, 9, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 8, 5120);
    conv_2d(4, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 12, 9280);
    conv_2d(1, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 11, 8960);
    conv_2d(2, 11, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 10, 7680);
    conv_2d(3, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 9, 5760);
    conv_2d(4, 9, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 12, 9920);
    conv_2d(2, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 11, 9600);
    conv_2d(3, 11, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 10, 6400);
    conv_2d(4, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 12, 7360);
    conv_2d(3, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 11, 7040);
    conv_2d(4, 11, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 12, 7680);
    conv_2d(4, 12, -1);
    concatenation(0);
    average_pool_2d(0, 8064);
    reshape(0);
    fully_connected(0);
    softmax(0);
}
