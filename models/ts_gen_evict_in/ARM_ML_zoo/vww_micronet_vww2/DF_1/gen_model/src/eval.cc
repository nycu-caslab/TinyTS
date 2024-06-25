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
    conv_2d(0, 0, 304);
    conv_2d(1, 0, -1);
    conv_2d(0, 1, 304);
    conv_2d(1, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 0, 112);
    conv_2d(2, 0, -1);
    conv_2d(3, 0, 912);
    conv_2d(0, 2, 1120);
    conv_2d(1, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 1, 816);
    conv_2d(2, 1, -1);
    conv_2d(3, 1, 1616);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 0, 2000);
    conv_2d(4, 0, -1);
    conv_2d(5, 0, 0);
    conv_2d(0, 3, 304);
    conv_2d(1, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 2, 112);
    conv_2d(2, 2, -1);
    conv_2d(3, 2, 1616);
    conv_2d(0, 4, 1712);
    conv_2d(1, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 3, 1408);
    conv_2d(2, 3, -1);
    conv_2d(3, 3, 1616);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 1, 1408);
    conv_2d(4, 1, -1);
    conv_2d(5, 1, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 368);
    conv_2d(6, 0, -1);
    pad(0, 0);
    add(0, 0);
    conv_2d(7, 0, 2688);
    conv_2d(0, 5, 2672);
    conv_2d(1, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 4, 2480);
    conv_2d(2, 4, -1);
    conv_2d(3, 4, 3648);
    conv_2d(0, 6, 3744);
    conv_2d(1, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 5, 3440);
    conv_2d(2, 5, -1);
    conv_2d(3, 5, 3648);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 2, 4128);
    conv_2d(4, 2, -1);
    conv_2d(5, 2, 1664);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 2032);
    conv_2d(6, 1, -1);
    pad(0, 1);
    add(0, 1);
    conv_2d(7, 1, 3648);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 0, 4928);
    conv_2d(8, 0, -1);
    conv_2d(9, 0, -1);
    conv_2d(0, 7, 304);
    conv_2d(1, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 6, 112);
    conv_2d(2, 6, -1);
    conv_2d(3, 6, 912);
    conv_2d(0, 8, 1008);
    conv_2d(1, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 7, 704);
    conv_2d(2, 7, -1);
    conv_2d(3, 7, 912);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 3, 1072);
    conv_2d(4, 3, -1);
    conv_2d(5, 3, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 368);
    conv_2d(6, 2, -1);
    pad(0, 2);
    add(0, 2);
    conv_2d(7, 2, 4352);
    conv_2d(0, 9, 4336);
    conv_2d(1, 9, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 8, 4144);
    conv_2d(2, 8, -1);
    conv_2d(3, 8, 4944);
    conv_2d(0, 10, 4736);
    conv_2d(1, 10, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 9, 4736);
    conv_2d(2, 9, -1);
    conv_2d(3, 9, 4944);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 4, 4736);
    conv_2d(4, 4, -1);
    conv_2d(5, 4, 3328);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 3696);
    conv_2d(6, 3, -1);
    pad(0, 3);
    add(0, 3);
    conv_2d(7, 3, 6384);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 1, 6960);
    conv_2d(8, 1, -1);
    conv_2d(9, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 0, 1072);
    conv_2d(10, 0, -1);
    pad(1, 0);
    add(1, 0);
    conv_2d(11, 0, -1);
    conv_2d(0, 11, 304);
    conv_2d(1, 11, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 10, 112);
    conv_2d(2, 10, -1);
    conv_2d(3, 10, 912);
    conv_2d(0, 12, 1008);
    conv_2d(1, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 11, 704);
    conv_2d(2, 11, -1);
    conv_2d(3, 11, 912);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 5, 1072);
    conv_2d(4, 5, -1);
    conv_2d(5, 5, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 368);
    conv_2d(6, 4, -1);
    pad(0, 4);
    add(0, 4);
    conv_2d(7, 4, 2800);
    conv_2d(0, 13, 2800);
    conv_2d(1, 13, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 12, 2480);
    conv_2d(2, 12, -1);
    conv_2d(3, 12, 3280);
    conv_2d(0, 14, 3072);
    conv_2d(1, 14, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 13, 3072);
    conv_2d(2, 13, -1);
    conv_2d(3, 13, 3280);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 6, 3072);
    conv_2d(4, 6, -1);
    conv_2d(5, 6, 1664);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 2032);
    conv_2d(6, 5, -1);
    pad(0, 5);
    add(0, 5);
    conv_2d(7, 5, 8528);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 2, 9104);
    conv_2d(8, 2, -1);
    conv_2d(9, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 1, 1072);
    conv_2d(10, 1, -1);
    pad(1, 1);
    add(1, 1);
    conv_2d(11, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 0, 1072);
    conv_2d(12, 0, -1);
    add(2, 0);
    conv_2d(13, 0, -1);
    conv_2d(0, 15, 304);
    conv_2d(1, 15, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 14, 112);
    conv_2d(2, 14, -1);
    conv_2d(3, 14, 912);
    conv_2d(0, 16, 1008);
    conv_2d(1, 16, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 15, 704);
    conv_2d(2, 15, -1);
    conv_2d(3, 15, 912);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 7, 1072);
    conv_2d(4, 7, -1);
    conv_2d(5, 7, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 368);
    conv_2d(6, 6, -1);
    pad(0, 6);
    add(0, 6);
    conv_2d(7, 6, 4464);
    conv_2d(0, 17, 4464);
    conv_2d(1, 17, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 16, 4144);
    conv_2d(2, 16, -1);
    conv_2d(3, 16, 4944);
    conv_2d(0, 18, 4736);
    conv_2d(1, 18, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 17, 4736);
    conv_2d(2, 17, -1);
    conv_2d(3, 17, 4944);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 8, 4736);
    conv_2d(4, 8, -1);
    conv_2d(5, 8, 3328);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 3696);
    conv_2d(6, 7, -1);
    pad(0, 7);
    add(0, 7);
    conv_2d(7, 7, 10672);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 3, 11248);
    conv_2d(8, 3, -1);
    conv_2d(9, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 2, 1072);
    conv_2d(10, 2, -1);
    pad(1, 2);
    add(1, 2);
    conv_2d(11, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 1, 1072);
    conv_2d(12, 1, -1);
    add(2, 1);
    conv_2d(13, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 0, 608);
    conv_2d(14, 0, -1);
    conv_2d(15, 0, 0);
    conv_2d(0, 19, 304);
    conv_2d(1, 19, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 18, 112);
    conv_2d(2, 18, -1);
    conv_2d(3, 18, 912);
    conv_2d(0, 20, 1008);
    conv_2d(1, 20, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 19, 704);
    conv_2d(2, 19, -1);
    conv_2d(3, 19, 912);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 9, 1072);
    conv_2d(4, 9, -1);
    conv_2d(5, 9, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 8, 368);
    conv_2d(6, 8, -1);
    pad(0, 8);
    add(0, 8);
    conv_2d(7, 8, 2800);
    conv_2d(0, 21, 2800);
    conv_2d(1, 21, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 20, 2480);
    conv_2d(2, 20, -1);
    conv_2d(3, 20, 3280);
    conv_2d(0, 22, 3072);
    conv_2d(1, 22, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 21, 3072);
    conv_2d(2, 21, -1);
    conv_2d(3, 21, 3280);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 10, 3072);
    conv_2d(4, 10, -1);
    conv_2d(5, 10, 1664);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 9, 2032);
    conv_2d(6, 9, -1);
    pad(0, 9);
    add(0, 9);
    conv_2d(7, 9, 9600);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 4, 10176);
    conv_2d(8, 4, -1);
    conv_2d(9, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 3, 1072);
    conv_2d(10, 3, -1);
    pad(1, 3);
    add(1, 3);
    conv_2d(11, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 2, 1072);
    conv_2d(12, 2, -1);
    add(2, 2);
    conv_2d(13, 2, -1);
    conv_2d(0, 23, 304);
    conv_2d(1, 23, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 22, 112);
    conv_2d(2, 22, -1);
    conv_2d(3, 22, 912);
    conv_2d(0, 24, 1008);
    conv_2d(1, 24, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 23, 704);
    conv_2d(2, 23, -1);
    conv_2d(3, 23, 912);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 11, 1072);
    conv_2d(4, 11, -1);
    conv_2d(5, 11, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 10, 368);
    conv_2d(6, 10, -1);
    pad(0, 10);
    add(0, 10);
    conv_2d(7, 10, 4576);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 24, 4144);
    conv_2d(2, 24, -1);
    conv_2d(3, 24, 4944);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 12, 4736);
    conv_2d(4, 12, -1);
    conv_2d(5, 12, 3328);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 11, 3696);
    conv_2d(6, 11, -1);
    pad(0, 11);
    add(0, 11);
    conv_2d(7, 11, 11744);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 5, 12768);
    conv_2d(8, 5, -1);
    conv_2d(9, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 4, 1072);
    conv_2d(10, 4, -1);
    pad(1, 4);
    add(1, 4);
    conv_2d(11, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 3, 1072);
    conv_2d(12, 3, -1);
    add(2, 3);
    conv_2d(13, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 1, 608);
    conv_2d(14, 1, -1);
    conv_2d(15, 1, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 0, 448);
    conv_2d(16, 0, -1);
    add(3, 0);
    conv_2d(17, 0, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 12, 368);
    conv_2d(6, 12, -1);
    pad(0, 12);
    add(0, 12);
    conv_2d(7, 12, 3056);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 6, 2736);
    conv_2d(8, 6, -1);
    conv_2d(9, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 5, 1280);
    conv_2d(10, 5, -1);
    pad(1, 5);
    add(1, 5);
    conv_2d(11, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 4, 1280);
    conv_2d(12, 4, -1);
    add(2, 4);
    conv_2d(13, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 6, 1280);
    conv_2d(10, 6, -1);
    pad(1, 6);
    add(1, 6);
    conv_2d(11, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 5, 2144);
    conv_2d(12, 5, -1);
    add(2, 5);
    conv_2d(13, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 2, 2144);
    conv_2d(14, 2, -1);
    conv_2d(15, 2, 2144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 1, 2592);
    conv_2d(16, 1, -1);
    add(3, 1);
    conv_2d(17, 1, 2144);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 0, 2592);
    conv_2d(18, 0, -1);
    pad(2, 0);
    add(4, 0);
    conv_2d(19, 0, 2144);
    pad(3, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 6, 2144);
    conv_2d(12, 6, -1);
    add(2, 6);
    conv_2d(13, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 3, 2752);
    conv_2d(14, 3, -1);
    conv_2d(15, 3, 448);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 2, 896);
    conv_2d(16, 2, -1);
    add(3, 2);
    conv_2d(17, 2, 896);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 1, 1344);
    conv_2d(18, 1, -1);
    pad(2, 1);
    add(4, 1);
    conv_2d(19, 1, 896);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 0, 1200);
    conv_2d(20, 0, -1);
    add(5, 0);
    conv_2d(21, 0, -1);
    pad(3, 1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 3, 2144);
    conv_2d(16, 3, -1);
    add(3, 3);
    conv_2d(17, 3, 896);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 2, 1952);
    conv_2d(18, 2, -1);
    pad(2, 2);
    add(4, 2);
    conv_2d(19, 2, 896);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 1, 1200);
    conv_2d(20, 1, -1);
    add(5, 1);
    conv_2d(21, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 0, 1200);
    conv_2d(22, 0, -1);
    conv_2d(23, 0, 896);
    pad(4, 0);
    pad(3, 2);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 3, 3120);
    conv_2d(18, 3, -1);
    pad(2, 3);
    add(4, 3);
    conv_2d(19, 3, 304);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 2, 608);
    conv_2d(20, 2, -1);
    add(5, 2);
    conv_2d(21, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 1, 912);
    conv_2d(22, 1, -1);
    conv_2d(23, 1, 608);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 0, 832);
    conv_2d(24, 0, -1);
    add(6, 0);
    conv_2d(25, 0, -1);
    pad(4, 1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 3, 1360);
    conv_2d(20, 3, -1);
    pad(3, 3);
    add(5, 3);
    conv_2d(21, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 2, 1360);
    conv_2d(22, 2, -1);
    conv_2d(23, 2, 608);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 1, 832);
    conv_2d(24, 1, -1);
    add(6, 1);
    conv_2d(25, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 0, 832);
    conv_2d(26, 0, -1);
    add(7, 0);
    conv_2d(27, 0, -1);
    pad(4, 2);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 3, 2256);
    conv_2d(22, 3, -1);
    conv_2d(23, 3, 224);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 2, 448);
    conv_2d(24, 2, -1);
    add(6, 2);
    conv_2d(25, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 1, 672);
    conv_2d(26, 1, -1);
    add(7, 1);
    conv_2d(27, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 3, 1008);
    conv_2d(24, 3, -1);
    pad(4, 3);
    add(6, 3);
    conv_2d(25, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 2, 1008);
    conv_2d(26, 2, -1);
    add(7, 2);
    conv_2d(27, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(13, 0, 1008);
    conv_2d(28, 0, -1);
    conv_2d(29, 0, 448);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 3, 1232);
    conv_2d(26, 3, -1);
    add(7, 3);
    conv_2d(27, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(13, 1, 336);
    conv_2d(28, 1, -1);
    conv_2d(29, 1, 192);
    depthwise_conv_2d_tiny_kernel3x3_stride1(14, 0, 384);
    conv_2d(30, 0, -1);
    add(8, 0);
    conv_2d(31, 0, 384);
    depthwise_conv_2d_tiny_kernel3x3_stride1(14, 1, 640);
    conv_2d(30, 1, -1);
    add(8, 1);
    conv_2d(31, 1, 384);
    depthwise_conv_2d_tiny_kernel3x3_stride1(15, 0, 576);
    conv_2d(32, 0, -1);
    add(9, 0);
    conv_2d(33, 0, 416);
    depthwise_conv_2d_tiny_kernel3x3_stride1(15, 1, 576);
    conv_2d(32, 1, -1);
    add(9, 1);
    conv_2d(33, 1, 224);
    depthwise_conv_2d_tiny_kernel3x3_stride1(16, 0, 384);
    conv_2d(34, 0, -1);
    conv_2d(35, 0, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(16, 1, 384);
    conv_2d(34, 1, -1);
    conv_2d(35, 1, -1);
    concatenation(0);
    average_pool_2d(0, 640);
    conv_2d(36, -1);
    reshape(0);
}
