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
    split(0);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 0, 12288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 1, 12288);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 0, 12288);
    conv_2d(1, 0, 512);
    conv_2d(2, 0, 0);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 2, 14336);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 1, 512);
    conv_2d(1, 1, 768);
    conv_2d(2, 1, 0);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 3, 15872);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 2, 512);
    conv_2d(1, 2, 768);
    conv_2d(2, 2, 0);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 0, 768);
    conv_2d(3, 0, 768);
    conv_2d(4, 0, 0);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 4, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 3, 512);
    conv_2d(1, 3, 512);
    conv_2d(2, 3, 1536);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 5, 17920);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 4, 16896);
    conv_2d(1, 4, 2048);
    conv_2d(2, 4, 1536);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 1, 1536);
    conv_2d(3, 1, 1536);
    conv_2d(4, 1, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 1536);
    conv_2d(5, 0, -1);
    add(0, 0);
    conv_2d(6, 0, 1536);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 6, 12288);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 5, 2048);
    conv_2d(1, 5, 2304);
    conv_2d(2, 5, 1536);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 7, 13824);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 6, 2048);
    conv_2d(1, 6, 2304);
    conv_2d(2, 6, 1536);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 2, 2304);
    conv_2d(3, 2, 0);
    conv_2d(4, 2, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 1024);
    conv_2d(5, 1, -1);
    add(0, 1);
    conv_2d(6, 1, 256);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 0, 640);
    conv_2d(7, 0, -1);
    conv_2d(8, 0, 0);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 8, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 7, 512);
    conv_2d(1, 7, 512);
    conv_2d(2, 7, 1792);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 9, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 8, 2048);
    conv_2d(1, 8, 2048);
    conv_2d(2, 8, 3328);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 3, 3072);
    conv_2d(3, 3, 3072);
    conv_2d(4, 3, 3072);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 3072);
    conv_2d(5, 2, -1);
    add(0, 2);
    conv_2d(6, 2, 3072);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 10, 12288);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 9, 3584);
    conv_2d(1, 9, 3840);
    conv_2d(2, 9, 3072);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 11, 15360);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 10, 3584);
    conv_2d(1, 10, 3840);
    conv_2d(2, 10, 3072);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 4, 3840);
    conv_2d(3, 4, 0);
    conv_2d(4, 4, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 1024);
    conv_2d(5, 3, -1);
    add(0, 3);
    conv_2d(6, 3, 256);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 1, 640);
    conv_2d(7, 1, -1);
    conv_2d(8, 1, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 0, 768);
    conv_2d(9, 0, -1);
    add(1, 0);
    conv_2d(10, 0, 0);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 12, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 11, 512);
    conv_2d(1, 11, 512);
    conv_2d(2, 11, 3328);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 13, 3072);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 12, 3584);
    conv_2d(1, 12, 3584);
    conv_2d(2, 12, 4864);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 5, 4608);
    conv_2d(3, 5, 1536);
    conv_2d(4, 5, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 2560);
    conv_2d(5, 4, -1);
    add(0, 4);
    conv_2d(6, 4, 1792);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 14, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 13, 2048);
    conv_2d(1, 13, 2048);
    conv_2d(2, 13, 4864);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 15, 12288);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 14, 5120);
    conv_2d(1, 14, 5376);
    conv_2d(2, 14, 4608);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 6, 5376);
    conv_2d(3, 6, 0);
    conv_2d(4, 6, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 1024);
    conv_2d(5, 5, -1);
    add(0, 5);
    conv_2d(6, 5, 256);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 2, 640);
    conv_2d(7, 2, -1);
    conv_2d(8, 2, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 1, 768);
    conv_2d(9, 1, -1);
    add(1, 1);
    conv_2d(10, 1, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 0, 768);
    conv_2d(11, 0, -1);
    add(2, 0);
    conv_2d(12, 0, 192);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 16, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 15, 512);
    conv_2d(1, 15, 512);
    conv_2d(2, 15, 4864);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 17, 4608);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 16, 5120);
    conv_2d(1, 16, 5120);
    conv_2d(2, 16, 6400);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 7, 6144);
    conv_2d(3, 7, 1536);
    conv_2d(4, 7, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 2560);
    conv_2d(5, 6, -1);
    add(0, 6);
    conv_2d(6, 6, 1792);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 18, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 17, 2048);
    conv_2d(1, 17, 2048);
    conv_2d(2, 17, 3328);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 19, 3072);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 18, 3584);
    conv_2d(1, 18, 3584);
    conv_2d(2, 18, 6400);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 8, 6912);
    conv_2d(3, 8, 0);
    conv_2d(4, 8, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 1024);
    conv_2d(5, 7, -1);
    add(0, 7);
    conv_2d(6, 7, 256);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 3, 640);
    conv_2d(7, 3, -1);
    conv_2d(8, 3, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 2, 768);
    conv_2d(9, 2, -1);
    add(1, 2);
    conv_2d(10, 2, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 1, 768);
    conv_2d(11, 1, -1);
    add(2, 1);
    conv_2d(12, 1, 192);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 0, 480);
    conv_2d(13, 0, -1);
    conv_2d(14, 0, -1);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 20, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 19, 512);
    conv_2d(1, 19, 512);
    conv_2d(2, 19, 6400);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 21, 6144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 20, 6656);
    conv_2d(1, 20, 6656);
    conv_2d(2, 20, 7936);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 9, 7680);
    conv_2d(3, 9, 1536);
    conv_2d(4, 9, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 8, 2560);
    conv_2d(5, 8, -1);
    add(0, 8);
    conv_2d(6, 8, 1792);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 22, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 21, 2048);
    conv_2d(1, 21, 2048);
    conv_2d(2, 21, 4864);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 23, 4608);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 22, 5120);
    conv_2d(1, 22, 5120);
    conv_2d(2, 22, 7936);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 10, 8448);
    conv_2d(3, 10, 0);
    conv_2d(4, 10, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 9, 1024);
    conv_2d(5, 9, -1);
    add(0, 9);
    conv_2d(6, 9, 256);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 4, 640);
    conv_2d(7, 4, -1);
    conv_2d(8, 4, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 3, 768);
    conv_2d(9, 3, -1);
    add(1, 3);
    conv_2d(10, 3, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 2, 768);
    conv_2d(11, 2, -1);
    add(2, 2);
    conv_2d(12, 2, 192);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 24, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 23, 512);
    conv_2d(1, 23, 512);
    conv_2d(2, 23, 3328);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 25, 3072);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 24, 3584);
    conv_2d(1, 24, 3584);
    conv_2d(2, 24, 9216);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 11, 12288);
    conv_2d(3, 11, 1536);
    conv_2d(4, 11, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 10, 2560);
    conv_2d(5, 10, -1);
    add(0, 10);
    conv_2d(6, 10, 1792);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 26, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 25, 2048);
    conv_2d(1, 25, 2048);
    conv_2d(2, 25, 6400);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 27, 6144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 26, 6656);
    conv_2d(1, 26, 6656);
    conv_2d(2, 26, 9216);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 12, 9728);
    conv_2d(3, 12, 0);
    conv_2d(4, 12, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 11, 1024);
    conv_2d(5, 11, -1);
    add(0, 11);
    conv_2d(6, 11, 256);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 5, 640);
    conv_2d(7, 5, -1);
    conv_2d(8, 5, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 4, 768);
    conv_2d(9, 4, -1);
    add(1, 4);
    conv_2d(10, 4, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 3, 768);
    conv_2d(11, 3, -1);
    add(2, 3);
    conv_2d(12, 3, 192);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 1, 480);
    conv_2d(13, 1, -1);
    conv_2d(14, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 0, 640);
    conv_2d(15, 0, -1);
    add(3, 0);
    conv_2d(16, 0, -1);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 28, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 27, 512);
    conv_2d(1, 27, 512);
    conv_2d(2, 27, 4864);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 29, 4608);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 28, 5120);
    conv_2d(1, 28, 5120);
    conv_2d(2, 28, 7936);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 13, 8448);
    conv_2d(3, 13, 1536);
    conv_2d(4, 13, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 12, 2560);
    conv_2d(5, 12, -1);
    add(0, 12);
    conv_2d(6, 12, 1792);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 30, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 29, 2048);
    conv_2d(1, 29, 2048);
    conv_2d(2, 29, 3328);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 31, 3072);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 30, 3584);
    conv_2d(1, 30, 3584);
    conv_2d(2, 30, 11520);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 14, 12032);
    conv_2d(3, 14, 0);
    conv_2d(4, 14, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 13, 1024);
    conv_2d(5, 13, -1);
    add(0, 13);
    conv_2d(6, 13, 256);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 6, 640);
    conv_2d(7, 6, -1);
    conv_2d(8, 6, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 5, 768);
    conv_2d(9, 5, -1);
    add(1, 5);
    conv_2d(10, 5, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 4, 768);
    conv_2d(11, 4, -1);
    add(2, 4);
    conv_2d(12, 4, 192);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 31, 512);
    conv_2d(1, 31, 512);
    conv_2d(2, 31, 10176);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 15, 9920);
    conv_2d(3, 15, 0);
    conv_2d(4, 15, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 14, 1024);
    conv_2d(5, 14, -1);
    add(0, 14);
    conv_2d(6, 14, 2560);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 15, 3584);
    conv_2d(5, 15, -1);
    add(0, 15);
    conv_2d(6, 15, 2816);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 7, 3200);
    conv_2d(7, 7, -1);
    conv_2d(8, 7, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 6, 768);
    conv_2d(9, 6, -1);
    add(1, 6);
    conv_2d(10, 6, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 5, 768);
    conv_2d(11, 5, -1);
    add(2, 5);
    conv_2d(12, 5, 2880);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 2, 3168);
    conv_2d(13, 2, -1);
    conv_2d(14, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 1, 3328);
    conv_2d(15, 1, -1);
    add(3, 1);
    conv_2d(16, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 7, 3648);
    conv_2d(9, 7, -1);
    add(1, 7);
    conv_2d(10, 7, 960);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 6, 1728);
    conv_2d(11, 6, -1);
    add(2, 6);
    conv_2d(12, 6, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 7, 5856);
    conv_2d(11, 7, -1);
    add(2, 7);
    conv_2d(12, 7, 3072);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 3, 3360);
    conv_2d(13, 3, -1);
    conv_2d(14, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 2, 1440);
    conv_2d(15, 2, -1);
    add(3, 2);
    conv_2d(16, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 3, 1440);
    conv_2d(15, 3, -1);
    add(3, 3);
    conv_2d(16, 3, -1);
    depthwise_conv_2d_tiny_kernel7x7_stride1(8, 0, 480);
    conv_2d(17, 0, -1);
    add(4, 0);
    conv_2d(18, 0, -1);
    depthwise_conv_2d_tiny_kernel7x7_stride1(8, 1, 960);
    conv_2d(17, 1, -1);
    add(4, 1);
    conv_2d(18, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 0, 960);
    conv_2d(19, 0, -1);
    conv_2d(20, 0, 480);
    depthwise_conv_2d_tiny_kernel7x7_stride1(8, 2, 2080);
    conv_2d(17, 2, -1);
    add(4, 2);
    conv_2d(18, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 1, 1920);
    conv_2d(19, 1, -1);
    conv_2d(20, 1, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 0, 768);
    conv_2d(21, 0, -1);
    add(5, 0);
    conv_2d(22, 0, 1728);
    depthwise_conv_2d_tiny_kernel7x7_stride1(8, 3, 1728);
    conv_2d(17, 3, -1);
    add(4, 3);
    conv_2d(18, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 2, 1728);
    conv_2d(19, 2, -1);
    conv_2d(20, 2, 1728);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 1, 1728);
    conv_2d(21, 1, -1);
    add(5, 1);
    conv_2d(22, 1, 3648);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 3, 2688);
    conv_2d(19, 3, -1);
    conv_2d(20, 3, 2688);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 2, 2688);
    conv_2d(21, 2, -1);
    add(5, 2);
    conv_2d(22, 2, 4608);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 3, 5376);
    conv_2d(21, 3, -1);
    add(5, 3);
    conv_2d(22, 3, 4032);
    depthwise_conv_2d_tiny_kernel7x7_stride2(11, 0, 4320);
    conv_2d(23, 0, -1);
    conv_2d(24, 0, -1);
    depthwise_conv_2d_tiny_kernel7x7_stride2(11, 1, 5472);
    conv_2d(23, 1, -1);
    conv_2d(24, 1, -1);
    depthwise_conv_2d_tiny_kernel5x5_stride1(12, 0, 1920);
    conv_2d(25, 0, -1);
    add(6, 0);
    conv_2d(26, 0, -1);
    depthwise_conv_2d_tiny_kernel5x5_stride1(12, 1, 2880);
    conv_2d(25, 1, -1);
    add(6, 1);
    conv_2d(26, 1, -1);
    depthwise_conv_2d_tiny_kernel7x7_stride1(13, 0, 1536);
    conv_2d(27, 0, -1);
    depthwise_conv_2d_tiny_kernel7x7_stride1(13, 1, 1856);
    conv_2d(27, 1, -1);
    concatenation(0);
    average_pool_2d(0, 800);
    conv_2d(28, -1);
    reshape(0);
}