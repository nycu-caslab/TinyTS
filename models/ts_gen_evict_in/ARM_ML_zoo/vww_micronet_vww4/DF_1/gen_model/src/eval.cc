#include "gen_model/include/eval.h"
#include "gen_model/include/ctx.h"
#include "gen_lib/include/ctx_util.h"
#include "gen_lib/OpImpl/ADD.h"
#include "gen_lib/OpImpl/AVERAGE_POOL_2D.h"
#include "gen_lib/OpImpl/CONCATENATION.h"
#include "gen_lib/OpImpl/CONV_2D.h"
#include "gen_lib/OpImpl/DEPTHWISE_CONV_2D.h"
#include "gen_lib/OpImpl/PAD.h"
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
    conv_2d(0, 0, 512);
    conv_2d(1, 0, 512);
    conv_2d(0, 1, 512);
    conv_2d(1, 1, 512);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 0, 256);
    conv_2d(2, 0, -1);
    conv_2d(3, 0, -1);
    conv_2d(0, 2, 1280);
    conv_2d(1, 2, 1280);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 1, 768);
    conv_2d(2, 1, -1);
    conv_2d(3, 1, -1);
    conv_2d(0, 3, 2048);
    conv_2d(1, 3, 2304);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 2, 1280);
    conv_2d(2, 2, -1);
    conv_2d(3, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 0, 2304);
    conv_2d(4, 0, 0);
    pad(0, 0);
    conv_2d(5, 0, -1);
    conv_2d(0, 4, 1536);
    conv_2d(1, 4, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 3, 768);
    conv_2d(2, 3, -1);
    conv_2d(3, 3, -1);
    conv_2d(0, 5, 2304);
    conv_2d(1, 5, 2304);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 4, 1792);
    conv_2d(2, 4, -1);
    conv_2d(3, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 1, 2304);
    conv_2d(4, 1, 896);
    pad(0, 1);
    conv_2d(5, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 512);
    conv_2d(6, 0, 512);
    add(0, 0);
    conv_2d(7, 0, 1408);
    conv_2d(0, 6, 1408);
    conv_2d(1, 6, 1408);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 5, 2304);
    conv_2d(2, 5, -1);
    conv_2d(3, 5, -1);
    conv_2d(0, 7, 1408);
    conv_2d(1, 7, 1408);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 6, 2816);
    conv_2d(2, 6, -1);
    conv_2d(3, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 2, 3776);
    conv_2d(4, 2, 1792);
    pad(0, 2);
    conv_2d(5, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 1408);
    conv_2d(6, 1, 1408);
    add(0, 1);
    conv_2d(7, 1, 2304);
    conv_2d(0, 8, 2304);
    conv_2d(1, 8, 2304);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 7, 3072);
    conv_2d(2, 7, -1);
    conv_2d(3, 7, -1);
    conv_2d(0, 9, 2304);
    conv_2d(1, 9, 2304);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 8, 3584);
    conv_2d(2, 8, -1);
    conv_2d(3, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 3, 2304);
    conv_2d(4, 3, 2688);
    pad(0, 3);
    conv_2d(5, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 2304);
    conv_2d(6, 2, 2304);
    add(0, 2);
    conv_2d(7, 2, 3200);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 0, 3200);
    conv_2d(8, 0, -1);
    conv_2d(9, 0, -1);
    conv_2d(0, 10, 1536);
    conv_2d(1, 10, 3200);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 9, 4096);
    conv_2d(2, 9, -1);
    conv_2d(3, 9, -1);
    conv_2d(0, 11, 3200);
    conv_2d(1, 11, 3200);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 10, 4608);
    conv_2d(2, 10, -1);
    conv_2d(3, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 4, 3200);
    conv_2d(4, 4, 3584);
    pad(0, 4);
    conv_2d(5, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 3200);
    conv_2d(6, 3, 3200);
    add(0, 3);
    conv_2d(7, 3, 4096);
    conv_2d(0, 12, 5120);
    conv_2d(1, 12, 5120);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 11, 4352);
    conv_2d(2, 11, -1);
    conv_2d(3, 11, -1);
    conv_2d(0, 13, 1536);
    conv_2d(1, 13, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 12, 5120);
    conv_2d(2, 12, -1);
    conv_2d(3, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 5, 5120);
    conv_2d(4, 5, 4480);
    pad(0, 5);
    conv_2d(5, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 1536);
    conv_2d(6, 4, 1536);
    add(0, 4);
    conv_2d(7, 4, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 1, 1536);
    conv_2d(8, 1, -1);
    conv_2d(9, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 0, 1536);
    conv_2d(10, 0, -1);
    pad(1, 0);
    add(1, 0);
    conv_2d(11, 0, -1);
    conv_2d(0, 14, 2048);
    conv_2d(1, 14, 2048);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 13, 1792);
    conv_2d(2, 13, -1);
    conv_2d(3, 13, -1);
    conv_2d(0, 15, 2048);
    conv_2d(1, 15, 2048);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 14, 6400);
    conv_2d(2, 14, -1);
    conv_2d(3, 14, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 6, 8896);
    conv_2d(4, 6, 1920);
    pad(0, 6);
    conv_2d(5, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 1536);
    conv_2d(6, 5, 2048);
    add(0, 5);
    conv_2d(7, 5, 7424);
    conv_2d(0, 16, 7936);
    conv_2d(1, 16, 7936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 15, 7680);
    conv_2d(2, 15, -1);
    conv_2d(3, 15, -1);
    conv_2d(0, 17, 8896);
    conv_2d(1, 17, 8896);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 16, 8192);
    conv_2d(2, 16, -1);
    conv_2d(3, 16, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 7, 8192);
    conv_2d(4, 7, 6784);
    pad(0, 7);
    conv_2d(5, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 6400);
    conv_2d(6, 6, 7872);
    add(0, 6);
    conv_2d(7, 6, 6400);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 2, 6848);
    conv_2d(8, 2, -1);
    conv_2d(9, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 1, 1536);
    conv_2d(10, 1, -1);
    pad(1, 1);
    add(1, 1);
    conv_2d(11, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 0, 1536);
    conv_2d(12, 0, -1);
    pad(2, 0);
    add(2, 0);
    conv_2d(13, 0, -1);
    conv_2d(0, 18, 9984);
    conv_2d(1, 18, 9984);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 17, 9728);
    conv_2d(2, 17, -1);
    conv_2d(3, 17, -1);
    conv_2d(0, 19, 11520);
    conv_2d(1, 19, 11520);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 18, 10240);
    conv_2d(2, 18, -1);
    conv_2d(3, 18, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 8, 10240);
    conv_2d(4, 8, 9856);
    pad(0, 8);
    conv_2d(5, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 9472);
    conv_2d(6, 7, 9984);
    add(0, 7);
    conv_2d(7, 7, 10880);
    conv_2d(0, 20, 11392);
    conv_2d(1, 20, 11392);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 19, 11136);
    conv_2d(2, 19, -1);
    conv_2d(3, 19, -1);
    conv_2d(0, 21, 11392);
    conv_2d(1, 21, 11392);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 20, 12288);
    conv_2d(2, 20, -1);
    conv_2d(3, 20, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 9, 12288);
    conv_2d(4, 9, 11264);
    pad(0, 9);
    conv_2d(5, 9, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 8, 10880);
    conv_2d(6, 8, 11840);
    add(0, 8);
    conv_2d(7, 8, 10368);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 3, 10816);
    conv_2d(8, 3, -1);
    conv_2d(9, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 2, 9088);
    conv_2d(10, 2, -1);
    pad(1, 2);
    add(1, 2);
    conv_2d(11, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 1, 9088);
    conv_2d(12, 1, -1);
    pad(2, 1);
    add(2, 1);
    conv_2d(13, 1, -1);
    conv_2d(0, 22, 5376);
    conv_2d(1, 22, 5376);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 21, 5120);
    conv_2d(2, 21, -1);
    conv_2d(3, 21, -1);
    conv_2d(0, 23, 5376);
    conv_2d(1, 23, 5376);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 22, 13664);
    conv_2d(2, 22, -1);
    conv_2d(3, 22, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 10, 13408);
    conv_2d(4, 10, 4864);
    pad(0, 10);
    conv_2d(5, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 9, 5376);
    conv_2d(6, 9, 5376);
    add(0, 9);
    conv_2d(7, 9, 5376);
    conv_2d(0, 24, 5376);
    conv_2d(1, 24, 5376);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 23, 14560);
    conv_2d(2, 23, -1);
    conv_2d(3, 23, -1);
    conv_2d(0, 25, 5376);
    conv_2d(1, 25, 5376);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 24, 14816);
    conv_2d(2, 24, -1);
    conv_2d(3, 24, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 11, 14816);
    conv_2d(4, 11, 5376);
    pad(0, 11);
    conv_2d(5, 11, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 10, 5376);
    conv_2d(6, 10, 5376);
    add(0, 10);
    conv_2d(7, 10, 4864);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 4, 5312);
    conv_2d(8, 4, -1);
    conv_2d(9, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 3, 4864);
    conv_2d(10, 3, -1);
    pad(1, 3);
    add(1, 3);
    conv_2d(11, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 2, 6400);
    conv_2d(12, 2, -1);
    pad(2, 2);
    add(2, 2);
    conv_2d(13, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 0, 8512);
    conv_2d(14, 0, -1);
    conv_2d(15, 0, -1);
    conv_2d(0, 26, 512);
    conv_2d(1, 26, 512);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 25, 256);
    conv_2d(2, 25, -1);
    conv_2d(3, 25, -1);
    conv_2d(0, 27, 1024);
    conv_2d(1, 27, 1024);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 26, 768);
    conv_2d(2, 26, -1);
    conv_2d(3, 26, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 12, 1408);
    conv_2d(4, 12, 384);
    pad(0, 12);
    conv_2d(5, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 11, 0);
    conv_2d(6, 11, 512);
    add(0, 11);
    conv_2d(7, 11, 2688);
    conv_2d(0, 28, 2688);
    conv_2d(1, 28, 2688);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 27, 2176);
    conv_2d(2, 27, -1);
    conv_2d(3, 27, -1);
    conv_2d(0, 29, 3200);
    conv_2d(1, 29, 3200);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 28, 2688);
    conv_2d(2, 28, -1);
    conv_2d(3, 28, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 13, 2688);
    conv_2d(4, 13, 1280);
    pad(0, 13);
    conv_2d(5, 13, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 12, 896);
    conv_2d(6, 12, 2368);
    add(0, 12);
    conv_2d(7, 12, 896);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 5, 1344);
    conv_2d(8, 5, -1);
    conv_2d(9, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 4, 1536);
    conv_2d(10, 4, -1);
    pad(1, 4);
    add(1, 4);
    conv_2d(11, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 3, 1536);
    conv_2d(12, 3, -1);
    pad(2, 3);
    add(2, 3);
    conv_2d(13, 3, -1);
    conv_2d(0, 30, 10464);
    conv_2d(1, 30, 10464);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 29, 10208);
    conv_2d(2, 29, -1);
    conv_2d(3, 29, -1);
    conv_2d(0, 31, 10976);
    conv_2d(1, 31, 10976);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 30, 10720);
    conv_2d(2, 30, -1);
    conv_2d(3, 30, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 14, 11232);
    conv_2d(4, 14, 3712);
    pad(0, 14);
    conv_2d(5, 14, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 13, 3328);
    conv_2d(6, 13, 11424);
    add(0, 13);
    conv_2d(7, 13, 4224);
    conv_2d(0, 32, 4224);
    conv_2d(1, 32, 4224);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 31, 11232);
    conv_2d(2, 31, -1);
    conv_2d(3, 31, -1);
    conv_2d(0, 33, 4224);
    conv_2d(1, 33, 4224);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 32, 11744);
    conv_2d(2, 32, -1);
    conv_2d(3, 32, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 15, 11744);
    conv_2d(4, 15, 4224);
    pad(0, 15);
    conv_2d(5, 15, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 14, 4224);
    conv_2d(6, 14, 4224);
    add(0, 14);
    conv_2d(7, 14, 4224);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 6, 4224);
    conv_2d(8, 6, -1);
    conv_2d(9, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 5, 3968);
    conv_2d(10, 5, -1);
    pad(1, 5);
    add(1, 5);
    conv_2d(11, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 4, 3968);
    conv_2d(12, 4, -1);
    pad(2, 4);
    add(2, 4);
    conv_2d(13, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 1, 10656);
    conv_2d(14, 1, -1);
    conv_2d(15, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 0, 2144);
    conv_2d(16, 0, -1);
    pad(3, 0);
    add(3, 0);
    conv_2d(17, 0, -1);
    pad(4, 0);
    conv_2d(0, 34, 512);
    conv_2d(1, 34, 512);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 33, 256);
    conv_2d(2, 33, -1);
    conv_2d(3, 33, -1);
    conv_2d(0, 35, 1024);
    conv_2d(1, 35, 1024);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 34, 768);
    conv_2d(2, 34, -1);
    conv_2d(3, 34, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 16, 1408);
    conv_2d(4, 16, 384);
    pad(0, 16);
    conv_2d(5, 16, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 15, 0);
    conv_2d(6, 15, 512);
    add(0, 15);
    conv_2d(7, 15, 5120);
    conv_2d(0, 36, 5120);
    conv_2d(1, 36, 5120);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 35, 2176);
    conv_2d(2, 35, -1);
    conv_2d(3, 35, -1);
    conv_2d(0, 37, 5632);
    conv_2d(1, 37, 5632);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 36, 5120);
    conv_2d(2, 36, -1);
    conv_2d(3, 36, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 17, 5120);
    conv_2d(4, 17, 1280);
    pad(0, 17);
    conv_2d(5, 17, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 16, 896);
    conv_2d(6, 16, 2368);
    add(0, 16);
    conv_2d(7, 16, 896);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 7, 1344);
    conv_2d(8, 7, -1);
    conv_2d(9, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 6, 1536);
    conv_2d(10, 6, -1);
    pad(1, 6);
    add(1, 6);
    conv_2d(11, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 5, 1536);
    conv_2d(12, 5, -1);
    pad(2, 5);
    add(2, 5);
    conv_2d(13, 5, -1);
    conv_2d(0, 38, 15008);
    conv_2d(1, 38, 15008);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 37, 14496);
    conv_2d(2, 37, -1);
    conv_2d(3, 37, -1);
    conv_2d(0, 39, 15520);
    conv_2d(1, 39, 15520);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 38, 15008);
    conv_2d(2, 38, -1);
    conv_2d(3, 38, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 18, 15520);
    conv_2d(4, 18, 6144);
    pad(0, 18);
    conv_2d(5, 18, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 17, 5760);
    conv_2d(6, 17, 15712);
    add(0, 17);
    conv_2d(7, 17, 6656);
    conv_2d(0, 40, 6656);
    conv_2d(1, 40, 6656);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 39, 15520);
    conv_2d(2, 39, -1);
    conv_2d(3, 39, -1);
    conv_2d(0, 41, 6656);
    conv_2d(1, 41, 6656);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 40, 16032);
    conv_2d(2, 40, -1);
    conv_2d(3, 40, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 19, 16032);
    conv_2d(4, 19, 6656);
    pad(0, 19);
    conv_2d(5, 19, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 18, 6656);
    conv_2d(6, 18, 6656);
    add(0, 18);
    conv_2d(7, 18, 6656);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 8, 6656);
    conv_2d(8, 8, -1);
    conv_2d(9, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 7, 6400);
    conv_2d(10, 7, -1);
    pad(1, 7);
    add(1, 7);
    conv_2d(11, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 6, 6400);
    conv_2d(12, 6, -1);
    pad(2, 6);
    add(2, 6);
    conv_2d(13, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 2, 14944);
    conv_2d(14, 2, -1);
    conv_2d(15, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 1, 2144);
    conv_2d(16, 1, -1);
    pad(3, 1);
    add(3, 1);
    conv_2d(17, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 0, 2144);
    conv_2d(18, 0, -1);
    add(4, 0);
    conv_2d(19, 0, -1);
    pad(4, 1);
    conv_2d(0, 42, 512);
    conv_2d(1, 42, 512);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 41, 256);
    conv_2d(2, 41, -1);
    conv_2d(3, 41, -1);
    conv_2d(0, 43, 1024);
    conv_2d(1, 43, 1024);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 42, 768);
    conv_2d(2, 42, -1);
    conv_2d(3, 42, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 20, 1408);
    conv_2d(4, 20, 384);
    pad(0, 20);
    conv_2d(5, 20, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 19, 0);
    conv_2d(6, 19, 512);
    add(0, 19);
    conv_2d(7, 19, 2688);
    conv_2d(0, 44, 2688);
    conv_2d(1, 44, 2688);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 43, 2176);
    conv_2d(2, 43, -1);
    conv_2d(3, 43, -1);
    conv_2d(0, 45, 3200);
    conv_2d(1, 45, 3200);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 44, 2688);
    conv_2d(2, 44, -1);
    conv_2d(3, 44, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 21, 2688);
    conv_2d(4, 21, 1280);
    pad(0, 21);
    conv_2d(5, 21, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 20, 896);
    conv_2d(6, 20, 2368);
    add(0, 20);
    conv_2d(7, 20, 896);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 9, 1344);
    conv_2d(8, 9, -1);
    conv_2d(9, 9, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 8, 1536);
    conv_2d(10, 8, -1);
    pad(1, 8);
    add(1, 8);
    conv_2d(11, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 7, 1536);
    conv_2d(12, 7, -1);
    pad(2, 7);
    add(2, 7);
    conv_2d(13, 7, -1);
    conv_2d(0, 46, 20064);
    conv_2d(1, 46, 20064);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 45, 19808);
    conv_2d(2, 45, -1);
    conv_2d(3, 45, -1);
    conv_2d(0, 47, 20064);
    conv_2d(1, 47, 20064);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 46, 23488);
    conv_2d(2, 46, -1);
    conv_2d(3, 46, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 22, 24000);
    conv_2d(4, 22, 3712);
    pad(0, 22);
    conv_2d(5, 22, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 21, 3328);
    conv_2d(6, 21, 20064);
    add(0, 21);
    conv_2d(7, 21, 4224);
    conv_2d(0, 48, 4224);
    conv_2d(1, 48, 4224);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 47, 24000);
    conv_2d(2, 47, -1);
    conv_2d(3, 47, -1);
    conv_2d(0, 49, 4224);
    conv_2d(1, 49, 4224);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 48, 24512);
    conv_2d(2, 48, -1);
    conv_2d(3, 48, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 23, 24512);
    conv_2d(4, 23, 4224);
    pad(0, 23);
    conv_2d(5, 23, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 22, 4224);
    conv_2d(6, 22, 4224);
    add(0, 22);
    conv_2d(7, 22, 4224);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 10, 4224);
    conv_2d(8, 10, -1);
    conv_2d(9, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 9, 3968);
    conv_2d(10, 9, -1);
    pad(1, 9);
    add(1, 9);
    conv_2d(11, 9, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 8, 3968);
    conv_2d(12, 8, -1);
    pad(2, 8);
    add(2, 8);
    conv_2d(13, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 3, 19232);
    conv_2d(14, 3, -1);
    conv_2d(15, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 2, 2144);
    conv_2d(16, 2, -1);
    pad(3, 2);
    add(3, 2);
    conv_2d(17, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 1, 2144);
    conv_2d(18, 1, -1);
    add(4, 1);
    conv_2d(19, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 0, 2144);
    conv_2d(20, 0, -1);
    pad(5, 0);
    add(5, 0);
    conv_2d(21, 0, -1);
    pad(4, 2);
    conv_2d(0, 50, 512);
    conv_2d(1, 50, 512);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 49, 256);
    conv_2d(2, 49, -1);
    conv_2d(3, 49, -1);
    conv_2d(0, 51, 1024);
    conv_2d(1, 51, 1024);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 50, 768);
    conv_2d(2, 50, -1);
    conv_2d(3, 50, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 24, 1408);
    conv_2d(4, 24, 384);
    pad(0, 24);
    conv_2d(5, 24, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 23, 0);
    conv_2d(6, 23, 512);
    add(0, 23);
    conv_2d(7, 23, 5120);
    conv_2d(0, 52, 5120);
    conv_2d(1, 52, 5120);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 51, 2176);
    conv_2d(2, 51, -1);
    conv_2d(3, 51, -1);
    conv_2d(0, 53, 5632);
    conv_2d(1, 53, 5632);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 52, 5120);
    conv_2d(2, 52, -1);
    conv_2d(3, 52, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 25, 5120);
    conv_2d(4, 25, 1280);
    pad(0, 25);
    conv_2d(5, 25, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 24, 896);
    conv_2d(6, 24, 2368);
    add(0, 24);
    conv_2d(7, 24, 896);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 11, 1344);
    conv_2d(8, 11, -1);
    conv_2d(9, 11, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 10, 1536);
    conv_2d(10, 10, -1);
    pad(1, 10);
    add(1, 10);
    conv_2d(11, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 9, 1536);
    conv_2d(12, 9, -1);
    pad(2, 9);
    add(2, 9);
    conv_2d(13, 9, -1);
    conv_2d(0, 54, 21184);
    conv_2d(1, 54, 21184);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 53, 20928);
    conv_2d(2, 53, -1);
    conv_2d(3, 53, -1);
    conv_2d(0, 55, 21696);
    conv_2d(1, 55, 21696);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 54, 21440);
    conv_2d(2, 54, -1);
    conv_2d(3, 54, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 26, 21952);
    conv_2d(4, 26, 6144);
    pad(0, 26);
    conv_2d(5, 26, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 25, 5760);
    conv_2d(6, 25, 22144);
    add(0, 25);
    conv_2d(7, 25, 6656);
    conv_2d(0, 56, 6656);
    conv_2d(1, 56, 6656);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 55, 21952);
    conv_2d(2, 55, -1);
    conv_2d(3, 55, -1);
    conv_2d(0, 57, 6656);
    conv_2d(1, 57, 6656);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 56, 22464);
    conv_2d(2, 56, -1);
    conv_2d(3, 56, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 27, 22464);
    conv_2d(4, 27, 6656);
    pad(0, 27);
    conv_2d(5, 27, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 26, 6656);
    conv_2d(6, 26, 6656);
    add(0, 26);
    conv_2d(7, 26, 6656);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 12, 6656);
    conv_2d(8, 12, -1);
    conv_2d(9, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 11, 6400);
    conv_2d(10, 11, -1);
    pad(1, 11);
    add(1, 11);
    conv_2d(11, 11, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 10, 6400);
    conv_2d(12, 10, -1);
    pad(2, 10);
    add(2, 10);
    conv_2d(13, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 4, 21376);
    conv_2d(14, 4, -1);
    conv_2d(15, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 3, 2144);
    conv_2d(16, 3, -1);
    pad(3, 3);
    add(3, 3);
    conv_2d(17, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 2, 2144);
    conv_2d(18, 2, -1);
    add(4, 2);
    conv_2d(19, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 1, 2144);
    conv_2d(20, 1, -1);
    pad(5, 1);
    add(5, 1);
    conv_2d(21, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 0, 1536);
    conv_2d(22, 0, -1);
    conv_2d(23, 0, -1);
    pad(4, 3);
    conv_2d(0, 58, 512);
    conv_2d(1, 58, 512);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 57, 256);
    conv_2d(2, 57, -1);
    conv_2d(3, 57, -1);
    conv_2d(0, 59, 1024);
    conv_2d(1, 59, 1024);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 58, 768);
    conv_2d(2, 58, -1);
    conv_2d(3, 58, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 28, 1408);
    conv_2d(4, 28, 384);
    pad(0, 28);
    conv_2d(5, 28, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 27, 0);
    conv_2d(6, 27, 512);
    add(0, 27);
    conv_2d(7, 27, 2688);
    conv_2d(0, 60, 2688);
    conv_2d(1, 60, 2688);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 59, 2176);
    conv_2d(2, 59, -1);
    conv_2d(3, 59, -1);
    conv_2d(0, 61, 3200);
    conv_2d(1, 61, 3200);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 60, 2688);
    conv_2d(2, 60, -1);
    conv_2d(3, 60, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 29, 2688);
    conv_2d(4, 29, 1280);
    pad(0, 29);
    conv_2d(5, 29, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 28, 896);
    conv_2d(6, 28, 2368);
    add(0, 28);
    conv_2d(7, 28, 896);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 13, 1344);
    conv_2d(8, 13, -1);
    conv_2d(9, 13, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 12, 1536);
    conv_2d(10, 12, -1);
    pad(1, 12);
    add(1, 12);
    conv_2d(11, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 11, 1536);
    conv_2d(12, 11, -1);
    pad(2, 11);
    add(2, 11);
    conv_2d(13, 11, -1);
    conv_2d(0, 62, 17152);
    conv_2d(1, 62, 17152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 61, 16640);
    conv_2d(2, 61, -1);
    conv_2d(3, 61, -1);
    conv_2d(0, 63, 17856);
    conv_2d(1, 63, 17856);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 62, 17152);
    conv_2d(2, 62, -1);
    conv_2d(3, 62, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 30, 26080);
    conv_2d(4, 30, 3712);
    pad(0, 30);
    conv_2d(5, 30, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 29, 3328);
    conv_2d(6, 29, 17408);
    add(0, 29);
    conv_2d(7, 29, 4224);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 63, 17664);
    conv_2d(2, 63, -1);
    conv_2d(3, 63, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 31, 29376);
    conv_2d(4, 31, 4224);
    pad(0, 31);
    conv_2d(5, 31, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 30, 4224);
    conv_2d(6, 30, 4224);
    add(0, 30);
    conv_2d(7, 30, 4224);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 14, 4224);
    conv_2d(8, 14, -1);
    conv_2d(9, 14, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 13, 3968);
    conv_2d(10, 13, -1);
    pad(1, 13);
    add(1, 13);
    conv_2d(11, 13, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 12, 3968);
    conv_2d(12, 12, -1);
    pad(2, 12);
    add(2, 12);
    conv_2d(13, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 5, 17088);
    conv_2d(14, 5, -1);
    conv_2d(15, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 4, 2144);
    conv_2d(16, 4, -1);
    pad(3, 4);
    add(3, 4);
    conv_2d(17, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 3, 2144);
    conv_2d(18, 3, -1);
    add(4, 3);
    conv_2d(19, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 2, 2144);
    conv_2d(20, 2, -1);
    pad(5, 2);
    add(5, 2);
    conv_2d(21, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 1, 1536);
    conv_2d(22, 1, -1);
    conv_2d(23, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 0, 1376);
    conv_2d(24, 0, -1);
    add(6, 0);
    conv_2d(25, 0, -1);
    pad(4, 4);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 31, 0);
    conv_2d(6, 31, 960);
    add(0, 31);
    conv_2d(7, 31, 1408);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 15, 1344);
    conv_2d(8, 15, -1);
    conv_2d(9, 15, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 14, 1536);
    conv_2d(10, 14, -1);
    pad(1, 14);
    add(1, 14);
    conv_2d(11, 14, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 13, 1536);
    conv_2d(12, 13, -1);
    pad(2, 13);
    add(2, 13);
    conv_2d(13, 13, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 15, 6912);
    conv_2d(10, 15, -1);
    pad(1, 15);
    add(1, 15);
    conv_2d(11, 15, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 14, 6400);
    conv_2d(12, 14, -1);
    pad(2, 14);
    add(2, 14);
    conv_2d(13, 14, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 6, 12800);
    conv_2d(14, 6, -1);
    conv_2d(15, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 5, 2144);
    conv_2d(16, 5, -1);
    pad(3, 5);
    add(3, 5);
    conv_2d(17, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 4, 2144);
    conv_2d(18, 4, -1);
    add(4, 4);
    conv_2d(19, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 3, 2144);
    conv_2d(20, 3, -1);
    pad(5, 3);
    add(5, 3);
    conv_2d(21, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 2, 1536);
    conv_2d(22, 2, -1);
    conv_2d(23, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 1, 1376);
    conv_2d(24, 1, -1);
    add(6, 1);
    conv_2d(25, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 0, 1376);
    conv_2d(26, 0, -1);
    pad(6, 0);
    add(7, 0);
    conv_2d(27, 0, -1);
    pad(4, 5);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 15, 1536);
    conv_2d(12, 15, -1);
    pad(2, 15);
    add(2, 15);
    conv_2d(13, 15, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 7, 4736);
    conv_2d(14, 7, -1);
    conv_2d(15, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 6, 2144);
    conv_2d(16, 6, -1);
    pad(3, 6);
    add(3, 6);
    conv_2d(17, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 5, 2144);
    conv_2d(18, 5, -1);
    add(4, 5);
    conv_2d(19, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 4, 2144);
    conv_2d(20, 4, -1);
    pad(5, 4);
    add(5, 4);
    conv_2d(21, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 3, 1536);
    conv_2d(22, 3, -1);
    conv_2d(23, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 2, 1376);
    conv_2d(24, 2, -1);
    add(6, 2);
    conv_2d(25, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 1, 1376);
    conv_2d(26, 1, -1);
    pad(6, 1);
    add(7, 1);
    conv_2d(27, 1, -1);
    pad(4, 6);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 7, 2304);
    conv_2d(16, 7, -1);
    pad(3, 7);
    add(3, 7);
    conv_2d(17, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 6, 2304);
    conv_2d(18, 6, -1);
    add(4, 6);
    conv_2d(19, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 5, 2304);
    conv_2d(20, 5, -1);
    pad(5, 5);
    add(5, 5);
    conv_2d(21, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 4, 2304);
    conv_2d(22, 4, -1);
    conv_2d(23, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 3, 2304);
    conv_2d(24, 3, -1);
    add(6, 3);
    conv_2d(25, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 2, 2304);
    conv_2d(26, 2, -1);
    pad(6, 2);
    add(7, 2);
    conv_2d(27, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(13, 0, 2304);
    conv_2d(28, 0, -1);
    conv_2d(29, 0, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 7, 2144);
    conv_2d(18, 7, -1);
    pad(4, 7);
    add(4, 7);
    conv_2d(19, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 6, 2144);
    conv_2d(20, 6, -1);
    pad(5, 6);
    add(5, 6);
    conv_2d(21, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 5, 1536);
    conv_2d(22, 5, -1);
    conv_2d(23, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 4, 1376);
    conv_2d(24, 4, -1);
    add(6, 4);
    conv_2d(25, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 3, 1376);
    conv_2d(26, 3, -1);
    pad(6, 3);
    add(7, 3);
    conv_2d(27, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 7, 4448);
    conv_2d(20, 7, -1);
    pad(5, 7);
    add(5, 7);
    conv_2d(21, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 6, 3840);
    conv_2d(22, 6, -1);
    conv_2d(23, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 5, 3680);
    conv_2d(24, 5, -1);
    add(6, 5);
    conv_2d(25, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 4, 3680);
    conv_2d(26, 4, -1);
    pad(6, 4);
    add(7, 4);
    conv_2d(27, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(13, 1, 4672);
    conv_2d(28, 1, -1);
    conv_2d(29, 1, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(14, 0, 384);
    conv_2d(30, 0, -1);
    add(8, 0);
    pad(7, 0);
    conv_2d(31, 0, 64);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 7, 1536);
    conv_2d(22, 7, -1);
    conv_2d(23, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 6, 1376);
    conv_2d(24, 6, -1);
    add(6, 6);
    conv_2d(25, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 5, 1376);
    conv_2d(26, 5, -1);
    pad(6, 5);
    add(7, 5);
    conv_2d(27, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 7, 6528);
    conv_2d(24, 7, -1);
    add(6, 7);
    conv_2d(25, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 6, 5984);
    conv_2d(26, 6, -1);
    pad(6, 6);
    add(7, 6);
    conv_2d(27, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(13, 2, 11360);
    conv_2d(28, 2, -1);
    conv_2d(29, 2, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(14, 1, 384);
    conv_2d(30, 1, -1);
    add(8, 1);
    pad(7, 1);
    conv_2d(31, 1, 64);
    depthwise_conv_2d_tiny_kernel3x3_stride1(15, 0, 384);
    conv_2d(32, 0, -1);
    add(9, 0);
    conv_2d(33, 0, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 7, 1376);
    conv_2d(26, 7, -1);
    pad(6, 7);
    add(7, 7);
    conv_2d(27, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(13, 3, 6912);
    conv_2d(28, 3, -1);
    conv_2d(29, 3, 384);
    depthwise_conv_2d_tiny_kernel3x3_stride1(14, 2, 768);
    conv_2d(30, 2, -1);
    add(8, 2);
    pad(7, 2);
    conv_2d(31, 2, 768);
    depthwise_conv_2d_tiny_kernel3x3_stride1(15, 1, 1152);
    conv_2d(32, 1, -1);
    add(9, 1);
    conv_2d(33, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(16, 0, 1664);
    conv_2d(34, 0, -1);
    conv_2d(35, 0, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(14, 3, 1664);
    conv_2d(30, 3, -1);
    add(8, 3);
    pad(7, 3);
    conv_2d(31, 3, 1216);
    depthwise_conv_2d_tiny_kernel3x3_stride1(15, 2, 1664);
    conv_2d(32, 2, -1);
    add(9, 2);
    conv_2d(33, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(16, 1, 2560);
    conv_2d(34, 1, -1);
    conv_2d(35, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(15, 3, 3072);
    conv_2d(32, 3, -1);
    add(9, 3);
    conv_2d(33, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(16, 2, 1536);
    conv_2d(34, 2, -1);
    conv_2d(35, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(16, 3, 768);
    conv_2d(34, 3, -1);
    conv_2d(35, 3, -1);
    concatenation(0);
    average_pool_2d(0, 2176);
    conv_2d(36, -1);
    reshape(0);
}
