#include "gen_model/include/eval.h"
#include "gen_model/include/ctx.h"
#include "gen_lib/include/ctx_util.h"
#include "gen_lib/OpImpl/ADD.h"
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
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 0, 0);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 1, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 0, 1024);
    conv_2d(1, 0, 1024);
    conv_2d(2, 0, 4608);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 2, 4096);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 1, 5120);
    conv_2d(1, 1, 3072);
    conv_2d(2, 1, 7680);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 3, 7168);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 2, 8192);
    conv_2d(1, 2, 6144);
    conv_2d(2, 2, 9728);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 0, 12800);
    conv_2d(3, 0, 0);
    conv_2d(4, 0, 0);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 4, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 3, 1024);
    conv_2d(1, 3, 1024);
    conv_2d(2, 3, 9728);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 5, 9216);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 4, 10240);
    conv_2d(1, 4, 10240);
    conv_2d(2, 4, 12800);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 1, 14336);
    conv_2d(3, 1, 3072);
    conv_2d(4, 1, 3072);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 5120);
    conv_2d(5, 0, -1);
    add(0, 0);
    conv_2d(6, 0, 3584);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 6, 3072);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 5, 4096);
    conv_2d(1, 5, 4096);
    conv_2d(2, 5, 6656);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 7, 6144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 6, 7168);
    conv_2d(1, 6, 7168);
    conv_2d(2, 6, 19968);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 2, 20992);
    conv_2d(3, 2, 0);
    conv_2d(4, 2, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 2048);
    conv_2d(5, 1, -1);
    add(0, 1);
    conv_2d(6, 1, 512);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 0, 1280);
    conv_2d(7, 0, -1);
    conv_2d(8, 0, 0);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 8, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 7, 1024);
    conv_2d(1, 7, 1024);
    conv_2d(2, 7, 9728);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 9, 9216);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 8, 10240);
    conv_2d(1, 8, 10240);
    conv_2d(2, 8, 12800);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 3, 13824);
    conv_2d(3, 3, 3072);
    conv_2d(4, 3, 3072);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 5120);
    conv_2d(5, 2, -1);
    add(0, 2);
    conv_2d(6, 2, 3584);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 10, 3072);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 9, 4096);
    conv_2d(1, 9, 4096);
    conv_2d(2, 9, 6656);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 11, 6144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 10, 7168);
    conv_2d(1, 10, 7168);
    conv_2d(2, 10, 17920);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 4, 18944);
    conv_2d(3, 4, 0);
    conv_2d(4, 4, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 2048);
    conv_2d(5, 3, -1);
    add(0, 3);
    conv_2d(6, 3, 512);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 1, 1280);
    conv_2d(7, 1, -1);
    conv_2d(8, 1, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 0, 1536);
    conv_2d(9, 0, -1);
    add(1, 0);
    conv_2d(10, 0, 0);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 12, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 11, 1024);
    conv_2d(1, 11, 1024);
    conv_2d(2, 11, 9728);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 13, 9216);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 12, 10240);
    conv_2d(1, 12, 10240);
    conv_2d(2, 12, 12800);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 5, 13824);
    conv_2d(3, 5, 3072);
    conv_2d(4, 5, 3072);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 5120);
    conv_2d(5, 4, -1);
    add(0, 4);
    conv_2d(6, 4, 3584);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 14, 3072);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 13, 4096);
    conv_2d(1, 13, 4096);
    conv_2d(2, 13, 6656);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 15, 6144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 14, 7168);
    conv_2d(1, 14, 7168);
    conv_2d(2, 14, 20480);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 6, 21504);
    conv_2d(3, 6, 0);
    conv_2d(4, 6, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 2048);
    conv_2d(5, 5, -1);
    add(0, 5);
    conv_2d(6, 5, 512);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 2, 1280);
    conv_2d(7, 2, -1);
    conv_2d(8, 2, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 1, 1536);
    conv_2d(9, 1, -1);
    add(1, 1);
    conv_2d(10, 1, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 0, 1536);
    conv_2d(11, 0, -1);
    add(2, 0);
    conv_2d(12, 0, 384);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 15, 1024);
    conv_2d(1, 15, 1024);
    conv_2d(2, 15, 14592);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 7, 13696);
    conv_2d(3, 7, 0);
    conv_2d(4, 7, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 2048);
    conv_2d(5, 6, -1);
    add(0, 6);
    conv_2d(6, 6, 5120);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 7168);
    conv_2d(5, 7, -1);
    add(0, 7);
    conv_2d(6, 7, 5632);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 3, 6400);
    conv_2d(7, 3, -1);
    conv_2d(8, 3, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 2, 1536);
    conv_2d(9, 2, -1);
    add(1, 2);
    conv_2d(10, 2, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 1, 1536);
    conv_2d(11, 1, -1);
    add(2, 1);
    conv_2d(12, 1, 5760);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 0, 6336);
    conv_2d(13, 0, -1);
    conv_2d(14, 0, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 3, 7296);
    conv_2d(9, 3, -1);
    add(1, 3);
    conv_2d(10, 3, 1920);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 2, 3456);
    conv_2d(11, 2, -1);
    add(2, 2);
    conv_2d(12, 2, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 3, 10432);
    conv_2d(11, 3, -1);
    add(2, 3);
    conv_2d(12, 3, 6144);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 1, 6720);
    conv_2d(13, 1, -1);
    conv_2d(14, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 0, 2560);
    conv_2d(15, 0, -1);
    add(3, 0);
    conv_2d(16, 0, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 1, 4160);
    conv_2d(15, 1, -1);
    add(3, 1);
    conv_2d(16, 1, -1);
    depthwise_conv_2d_tiny_kernel7x7_stride1(8, 0, 2240);
    conv_2d(17, 0, -1);
    add(4, 0);
    conv_2d(18, 0, -1);
    depthwise_conv_2d_tiny_kernel7x7_stride1(8, 1, 2240);
    conv_2d(17, 1, -1);
    add(4, 1);
    conv_2d(18, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 0, 2880);
    conv_2d(19, 0, -1);
    conv_2d(20, 0, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 1, 4416);
    conv_2d(19, 1, -1);
    conv_2d(20, 1, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 0, 1536);
    conv_2d(21, 0, -1);
    add(5, 0);
    conv_2d(22, 0, 4992);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 1, 6912);
    conv_2d(21, 1, -1);
    add(5, 1);
    conv_2d(22, 1, 4224);
    depthwise_conv_2d_tiny_kernel7x7_stride2(11, 0, 4800);
    conv_2d(23, 0, -1);
    conv_2d(24, 0, -1);
    depthwise_conv_2d_tiny_kernel5x5_stride1(12, 0, 3840);
    conv_2d(25, 0, -1);
    add(6, 0);
    conv_2d(26, 0, -1);
    depthwise_conv_2d_tiny_kernel7x7_stride1(13, 0, 3072);
    conv_2d(27, 0, -1);
    concatenation(0);
    average_pool_2d(0, 160);
    conv_2d(28, -1);
    reshape(0);
}
