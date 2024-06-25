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
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 0, 1152);
    conv_2d(1, 0, 1152);
    conv_2d(2, 0, 5184);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 2, 4608);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 1, 5760);
    conv_2d(1, 1, 3456);
    conv_2d(2, 1, 7488);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 0, 9792);
    conv_2d(3, 0, 0);
    conv_2d(4, 0, 0);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 3, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 2, 1152);
    conv_2d(1, 2, 1152);
    conv_2d(2, 2, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 4, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 3, 8064);
    conv_2d(1, 3, 8064);
    conv_2d(2, 3, 12672);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 1, 14112);
    conv_2d(3, 1, 0);
    conv_2d(4, 1, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 1728);
    conv_2d(5, 0, 2016);
    add(0, 0);
    conv_2d(6, 0, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 5, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 4, 1152);
    conv_2d(1, 4, 1152);
    conv_2d(2, 4, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 6, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 5, 4608);
    conv_2d(1, 5, 4608);
    conv_2d(2, 5, 16128);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 2, 17280);
    conv_2d(3, 2, 0);
    conv_2d(4, 2, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 1728);
    conv_2d(5, 1, 2016);
    add(0, 1);
    conv_2d(6, 1, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 0, 864);
    conv_2d(7, 0, 864);
    conv_2d(8, 0, 0);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 7, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 6, 1152);
    conv_2d(1, 6, 1152);
    conv_2d(2, 6, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 8, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 7, 8064);
    conv_2d(1, 7, 8064);
    conv_2d(2, 7, 17856);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 3, 19008);
    conv_2d(3, 3, 0);
    conv_2d(4, 3, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 1728);
    conv_2d(5, 2, 2016);
    add(0, 2);
    conv_2d(6, 2, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 9, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 8, 1152);
    conv_2d(1, 8, 1152);
    conv_2d(2, 8, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 10, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 9, 4608);
    conv_2d(1, 9, 4608);
    conv_2d(2, 9, 19584);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 4, 20736);
    conv_2d(3, 4, 0);
    conv_2d(4, 4, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 1728);
    conv_2d(5, 3, 2016);
    add(0, 3);
    conv_2d(6, 3, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 1, 864);
    conv_2d(7, 1, 864);
    conv_2d(8, 1, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 0, 1728);
    conv_2d(9, 0, -1);
    add(1, 0);
    conv_2d(10, 0, 0);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 11, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 10, 1152);
    conv_2d(1, 10, 1152);
    conv_2d(2, 10, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 12, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 11, 8064);
    conv_2d(1, 11, 8064);
    conv_2d(2, 11, 21312);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 5, 23616);
    conv_2d(3, 5, 0);
    conv_2d(4, 5, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 1728);
    conv_2d(5, 4, 2016);
    add(0, 4);
    conv_2d(6, 4, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 13, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 12, 1152);
    conv_2d(1, 12, 1152);
    conv_2d(2, 12, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 14, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 13, 4608);
    conv_2d(1, 13, 4608);
    conv_2d(2, 13, 23040);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 6, 25344);
    conv_2d(3, 6, 0);
    conv_2d(4, 6, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 1728);
    conv_2d(5, 5, 2016);
    add(0, 5);
    conv_2d(6, 5, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 2, 864);
    conv_2d(7, 2, 864);
    conv_2d(8, 2, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 1, 1728);
    conv_2d(9, 1, -1);
    add(1, 1);
    conv_2d(10, 1, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 0, 1728);
    conv_2d(11, 0, -1);
    add(2, 0);
    conv_2d(12, 0, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 15, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 14, 1152);
    conv_2d(1, 14, 1152);
    conv_2d(2, 14, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 16, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 15, 8064);
    conv_2d(1, 15, 8064);
    conv_2d(2, 15, 24768);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 7, 28512);
    conv_2d(3, 7, 0);
    conv_2d(4, 7, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 1728);
    conv_2d(5, 6, 2016);
    add(0, 6);
    conv_2d(6, 6, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 17, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 16, 1152);
    conv_2d(1, 16, 1152);
    conv_2d(2, 16, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 18, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 17, 4608);
    conv_2d(1, 17, 4608);
    conv_2d(2, 17, 26496);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 8, 27648);
    conv_2d(3, 8, 0);
    conv_2d(4, 8, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 1728);
    conv_2d(5, 7, 2016);
    add(0, 7);
    conv_2d(6, 7, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 3, 864);
    conv_2d(7, 3, 864);
    conv_2d(8, 3, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 2, 1728);
    conv_2d(9, 2, -1);
    add(1, 2);
    conv_2d(10, 2, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 1, 1728);
    conv_2d(11, 1, -1);
    add(2, 1);
    conv_2d(12, 1, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 0, 864);
    conv_2d(13, 0, -1);
    conv_2d(14, 0, 0);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 19, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 18, 1152);
    conv_2d(1, 18, 1152);
    conv_2d(2, 18, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 20, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 19, 8064);
    conv_2d(1, 19, 8064);
    conv_2d(2, 19, 17856);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 9, 20736);
    conv_2d(3, 9, 0);
    conv_2d(4, 9, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 8, 1728);
    conv_2d(5, 8, 2016);
    add(0, 8);
    conv_2d(6, 8, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 21, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 20, 1152);
    conv_2d(1, 20, 1152);
    conv_2d(2, 20, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 22, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 21, 4608);
    conv_2d(1, 21, 4608);
    conv_2d(2, 21, 24768);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 10, 27648);
    conv_2d(3, 10, 0);
    conv_2d(4, 10, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 9, 1728);
    conv_2d(5, 9, 2016);
    add(0, 9);
    conv_2d(6, 9, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 4, 864);
    conv_2d(7, 4, 864);
    conv_2d(8, 4, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 3, 1728);
    conv_2d(9, 3, -1);
    add(1, 3);
    conv_2d(10, 3, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 2, 1728);
    conv_2d(11, 2, -1);
    add(2, 2);
    conv_2d(12, 2, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 23, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 22, 1152);
    conv_2d(1, 22, 1152);
    conv_2d(2, 22, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 24, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 23, 8064);
    conv_2d(1, 23, 8064);
    conv_2d(2, 23, 23040);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 11, 28800);
    conv_2d(3, 11, 0);
    conv_2d(4, 11, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 10, 1728);
    conv_2d(5, 10, 2016);
    add(0, 10);
    conv_2d(6, 10, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 25, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 24, 1152);
    conv_2d(1, 24, 1152);
    conv_2d(2, 24, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 26, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 25, 4608);
    conv_2d(1, 25, 4608);
    conv_2d(2, 25, 28224);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 12, 30672);
    conv_2d(3, 12, 0);
    conv_2d(4, 12, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 11, 1728);
    conv_2d(5, 11, 2016);
    add(0, 11);
    conv_2d(6, 11, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 5, 864);
    conv_2d(7, 5, 864);
    conv_2d(8, 5, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 4, 1728);
    conv_2d(9, 4, -1);
    add(1, 4);
    conv_2d(10, 4, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 3, 1728);
    conv_2d(11, 3, -1);
    add(2, 3);
    conv_2d(12, 3, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 1, 864);
    conv_2d(13, 1, -1);
    conv_2d(14, 1, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 0, 1296);
    conv_2d(15, 0, -1);
    add(3, 0);
    conv_2d(16, 0, 0);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 27, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 26, 1152);
    conv_2d(1, 26, 1152);
    conv_2d(2, 26, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 28, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 27, 8064);
    conv_2d(1, 27, 8064);
    conv_2d(2, 27, 12672);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 13, 13824);
    conv_2d(3, 13, 0);
    conv_2d(4, 13, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 12, 1728);
    conv_2d(5, 12, 2016);
    add(0, 12);
    conv_2d(6, 12, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 29, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 28, 1152);
    conv_2d(1, 28, 1152);
    conv_2d(2, 28, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 30, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 29, 4608);
    conv_2d(1, 29, 4608);
    conv_2d(2, 29, 23040);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 14, 25920);
    conv_2d(3, 14, 0);
    conv_2d(4, 14, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 13, 1728);
    conv_2d(5, 13, 2016);
    add(0, 13);
    conv_2d(6, 13, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 6, 864);
    conv_2d(7, 6, 864);
    conv_2d(8, 6, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 5, 1728);
    conv_2d(9, 5, -1);
    add(1, 5);
    conv_2d(10, 5, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 4, 1728);
    conv_2d(11, 4, -1);
    add(2, 4);
    conv_2d(12, 4, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 31, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 30, 1152);
    conv_2d(1, 30, 1152);
    conv_2d(2, 30, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 32, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 31, 8064);
    conv_2d(1, 31, 8064);
    conv_2d(2, 31, 24768);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 15, 28800);
    conv_2d(3, 15, 0);
    conv_2d(4, 15, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 14, 1728);
    conv_2d(5, 14, 2016);
    add(0, 14);
    conv_2d(6, 14, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 33, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 32, 1152);
    conv_2d(1, 32, 1152);
    conv_2d(2, 32, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 34, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 33, 4608);
    conv_2d(1, 33, 4608);
    conv_2d(2, 33, 28224);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 16, 33264);
    conv_2d(3, 16, 0);
    conv_2d(4, 16, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 15, 1728);
    conv_2d(5, 15, 2016);
    add(0, 15);
    conv_2d(6, 15, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 7, 864);
    conv_2d(7, 7, 864);
    conv_2d(8, 7, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 6, 1728);
    conv_2d(9, 6, -1);
    add(1, 6);
    conv_2d(10, 6, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 5, 1728);
    conv_2d(11, 5, -1);
    add(2, 5);
    conv_2d(12, 5, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 2, 864);
    conv_2d(13, 2, -1);
    conv_2d(14, 2, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 1, 1296);
    conv_2d(15, 1, -1);
    add(3, 1);
    conv_2d(16, 1, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 0, 1296);
    conv_2d(17, 0, -1);
    add(4, 0);
    conv_2d(18, 0, 0);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 35, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 34, 1152);
    conv_2d(1, 34, 1152);
    conv_2d(2, 34, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 36, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 35, 8064);
    conv_2d(1, 35, 8064);
    conv_2d(2, 35, 10944);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 17, 15552);
    conv_2d(3, 17, 0);
    conv_2d(4, 17, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 16, 1728);
    conv_2d(5, 16, 2016);
    add(0, 16);
    conv_2d(6, 16, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 37, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 36, 1152);
    conv_2d(1, 36, 1152);
    conv_2d(2, 36, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 38, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 37, 4608);
    conv_2d(1, 37, 4608);
    conv_2d(2, 37, 24768);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 18, 25920);
    conv_2d(3, 18, 0);
    conv_2d(4, 18, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 17, 1728);
    conv_2d(5, 17, 2016);
    add(0, 17);
    conv_2d(6, 17, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 8, 864);
    conv_2d(7, 8, 864);
    conv_2d(8, 8, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 7, 1728);
    conv_2d(9, 7, -1);
    add(1, 7);
    conv_2d(10, 7, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 6, 1728);
    conv_2d(11, 6, -1);
    add(2, 6);
    conv_2d(12, 6, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 39, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 38, 1152);
    conv_2d(1, 38, 1152);
    conv_2d(2, 38, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 40, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 39, 8064);
    conv_2d(1, 39, 8064);
    conv_2d(2, 39, 23040);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 19, 28800);
    conv_2d(3, 19, 0);
    conv_2d(4, 19, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 18, 1728);
    conv_2d(5, 18, 2016);
    add(0, 18);
    conv_2d(6, 18, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 41, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 40, 1152);
    conv_2d(1, 40, 1152);
    conv_2d(2, 40, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 42, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 41, 4608);
    conv_2d(1, 41, 4608);
    conv_2d(2, 41, 28224);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 20, 35856);
    conv_2d(3, 20, 0);
    conv_2d(4, 20, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 19, 1728);
    conv_2d(5, 19, 2016);
    add(0, 19);
    conv_2d(6, 19, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 9, 864);
    conv_2d(7, 9, 864);
    conv_2d(8, 9, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 8, 1728);
    conv_2d(9, 8, -1);
    add(1, 8);
    conv_2d(10, 8, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 7, 1728);
    conv_2d(11, 7, -1);
    add(2, 7);
    conv_2d(12, 7, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 3, 864);
    conv_2d(13, 3, -1);
    conv_2d(14, 3, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 2, 1296);
    conv_2d(15, 2, -1);
    add(3, 2);
    conv_2d(16, 2, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 1, 1296);
    conv_2d(17, 1, -1);
    add(4, 1);
    conv_2d(18, 1, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 0, 1296);
    conv_2d(19, 0, -1);
    add(5, 0);
    conv_2d(20, 0, 224);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 43, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 42, 1152);
    conv_2d(1, 42, 1152);
    conv_2d(2, 42, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 44, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 43, 8064);
    conv_2d(1, 43, 8064);
    conv_2d(2, 43, 19584);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 21, 20736);
    conv_2d(3, 21, 0);
    conv_2d(4, 21, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 20, 1728);
    conv_2d(5, 20, 2016);
    add(0, 20);
    conv_2d(6, 20, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 45, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 44, 1152);
    conv_2d(1, 44, 1152);
    conv_2d(2, 44, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 46, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 45, 4608);
    conv_2d(1, 45, 4608);
    conv_2d(2, 45, 23040);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 22, 25920);
    conv_2d(3, 22, 0);
    conv_2d(4, 22, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 21, 1728);
    conv_2d(5, 21, 2016);
    add(0, 21);
    conv_2d(6, 21, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 10, 864);
    conv_2d(7, 10, 864);
    conv_2d(8, 10, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 9, 1728);
    conv_2d(9, 9, -1);
    add(1, 9);
    conv_2d(10, 9, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 8, 1728);
    conv_2d(11, 8, -1);
    add(2, 8);
    conv_2d(12, 8, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 47, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 46, 1152);
    conv_2d(1, 46, 1152);
    conv_2d(2, 46, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 48, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 47, 8064);
    conv_2d(1, 47, 8064);
    conv_2d(2, 47, 24768);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 23, 28800);
    conv_2d(3, 23, 0);
    conv_2d(4, 23, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 22, 1728);
    conv_2d(5, 22, 2016);
    add(0, 22);
    conv_2d(6, 22, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 49, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 48, 1152);
    conv_2d(1, 48, 1152);
    conv_2d(2, 48, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 50, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 49, 4608);
    conv_2d(1, 49, 4608);
    conv_2d(2, 49, 28224);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 24, 30672);
    conv_2d(3, 24, 0);
    conv_2d(4, 24, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 23, 1728);
    conv_2d(5, 23, 2016);
    add(0, 23);
    conv_2d(6, 23, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 11, 864);
    conv_2d(7, 11, 864);
    conv_2d(8, 11, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 10, 1728);
    conv_2d(9, 10, -1);
    add(1, 10);
    conv_2d(10, 10, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 9, 1728);
    conv_2d(11, 9, -1);
    add(2, 9);
    conv_2d(12, 9, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 4, 864);
    conv_2d(13, 4, -1);
    conv_2d(14, 4, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 3, 1296);
    conv_2d(15, 3, -1);
    add(3, 3);
    conv_2d(16, 3, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 2, 1296);
    conv_2d(17, 2, -1);
    add(4, 2);
    conv_2d(18, 2, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 1, 1296);
    conv_2d(19, 1, -1);
    add(5, 1);
    conv_2d(20, 1, 224);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 0, 1296);
    conv_2d(21, 0, -1);
    conv_2d(22, 0, -1);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 51, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 50, 1152);
    conv_2d(1, 50, 1152);
    conv_2d(2, 50, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 52, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 51, 8064);
    conv_2d(1, 51, 8064);
    conv_2d(2, 51, 17856);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 25, 27072);
    conv_2d(3, 25, 0);
    conv_2d(4, 25, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 24, 1728);
    conv_2d(5, 24, 2016);
    add(0, 24);
    conv_2d(6, 24, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 53, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 52, 1152);
    conv_2d(1, 52, 1152);
    conv_2d(2, 52, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 54, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 53, 4608);
    conv_2d(1, 53, 4608);
    conv_2d(2, 53, 26496);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 26, 29376);
    conv_2d(3, 26, 0);
    conv_2d(4, 26, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 25, 1728);
    conv_2d(5, 25, 2016);
    add(0, 25);
    conv_2d(6, 25, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 12, 864);
    conv_2d(7, 12, 864);
    conv_2d(8, 12, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 11, 1728);
    conv_2d(9, 11, -1);
    add(1, 11);
    conv_2d(10, 11, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 10, 1728);
    conv_2d(11, 10, -1);
    add(2, 10);
    conv_2d(12, 10, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 55, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 54, 1152);
    conv_2d(1, 54, 1152);
    conv_2d(2, 54, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 56, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 55, 8064);
    conv_2d(1, 55, 8064);
    conv_2d(2, 55, 24768);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 27, 30816);
    conv_2d(3, 27, 0);
    conv_2d(4, 27, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 26, 1728);
    conv_2d(5, 26, 2016);
    add(0, 26);
    conv_2d(6, 26, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 57, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 56, 1152);
    conv_2d(1, 56, 1152);
    conv_2d(2, 56, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 58, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 57, 4608);
    conv_2d(1, 57, 4608);
    conv_2d(2, 57, 29952);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 28, 32400);
    conv_2d(3, 28, 0);
    conv_2d(4, 28, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 27, 1728);
    conv_2d(5, 27, 2016);
    add(0, 27);
    conv_2d(6, 27, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 13, 864);
    conv_2d(7, 13, 864);
    conv_2d(8, 13, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 12, 1728);
    conv_2d(9, 12, -1);
    add(1, 12);
    conv_2d(10, 12, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 11, 1728);
    conv_2d(11, 11, -1);
    add(2, 11);
    conv_2d(12, 11, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 5, 864);
    conv_2d(13, 5, -1);
    conv_2d(14, 5, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 4, 1296);
    conv_2d(15, 4, -1);
    add(3, 4);
    conv_2d(16, 4, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 3, 1296);
    conv_2d(17, 3, -1);
    add(4, 3);
    conv_2d(18, 3, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 2, 1296);
    conv_2d(19, 2, -1);
    add(5, 2);
    conv_2d(20, 2, 224);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 1, 1296);
    conv_2d(21, 1, -1);
    conv_2d(22, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 0, 1728);
    conv_2d(23, 0, -1);
    add(6, 0);
    conv_2d(24, 0, -1);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 59, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 58, 1152);
    conv_2d(1, 58, 1152);
    conv_2d(2, 58, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 60, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 59, 8064);
    conv_2d(1, 59, 8064);
    conv_2d(2, 59, 28224);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 29, 33408);
    conv_2d(3, 29, 0);
    conv_2d(4, 29, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 28, 1728);
    conv_2d(5, 28, 2016);
    add(0, 28);
    conv_2d(6, 28, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 61, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 60, 1152);
    conv_2d(1, 60, 1152);
    conv_2d(2, 60, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 62, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 61, 4608);
    conv_2d(1, 61, 4608);
    conv_2d(2, 61, 31680);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 30, 32832);
    conv_2d(3, 30, 0);
    conv_2d(4, 30, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 29, 1728);
    conv_2d(5, 29, 2016);
    add(0, 29);
    conv_2d(6, 29, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 14, 864);
    conv_2d(7, 14, 864);
    conv_2d(8, 14, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 13, 1728);
    conv_2d(9, 13, -1);
    add(1, 13);
    conv_2d(10, 13, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 12, 1728);
    conv_2d(11, 12, -1);
    add(2, 12);
    conv_2d(12, 12, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 63, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 62, 1152);
    conv_2d(1, 62, 1152);
    conv_2d(2, 62, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 64, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 63, 8064);
    conv_2d(1, 63, 8064);
    conv_2d(2, 63, 26496);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 31, 33984);
    conv_2d(3, 31, 0);
    conv_2d(4, 31, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 30, 1728);
    conv_2d(5, 30, 2016);
    add(0, 30);
    conv_2d(6, 30, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 65, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 64, 1152);
    conv_2d(1, 64, 1152);
    conv_2d(2, 64, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 66, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 65, 4608);
    conv_2d(1, 65, 4608);
    conv_2d(2, 65, 33408);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 32, 37872);
    conv_2d(3, 32, 0);
    conv_2d(4, 32, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 31, 1728);
    conv_2d(5, 31, 2016);
    add(0, 31);
    conv_2d(6, 31, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 15, 864);
    conv_2d(7, 15, 864);
    conv_2d(8, 15, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 14, 1728);
    conv_2d(9, 14, -1);
    add(1, 14);
    conv_2d(10, 14, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 13, 1728);
    conv_2d(11, 13, -1);
    add(2, 13);
    conv_2d(12, 13, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 6, 864);
    conv_2d(13, 6, -1);
    conv_2d(14, 6, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 5, 1296);
    conv_2d(15, 5, -1);
    add(3, 5);
    conv_2d(16, 5, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 4, 1296);
    conv_2d(17, 4, -1);
    add(4, 4);
    conv_2d(18, 4, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 3, 1296);
    conv_2d(19, 3, -1);
    add(5, 3);
    conv_2d(20, 3, 224);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 2, 1296);
    conv_2d(21, 2, -1);
    conv_2d(22, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 1, 1728);
    conv_2d(23, 1, -1);
    add(6, 1);
    conv_2d(24, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 0, 1728);
    conv_2d(25, 0, -1);
    add(7, 0);
    conv_2d(26, 0, -1);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 67, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 66, 1152);
    conv_2d(1, 66, 1152);
    conv_2d(2, 66, 7488);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 68, 6912);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 67, 8064);
    conv_2d(1, 67, 8064);
    conv_2d(2, 67, 29952);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 33, 35712);
    conv_2d(3, 33, 0);
    conv_2d(4, 33, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 32, 1728);
    conv_2d(5, 32, 2016);
    add(0, 32);
    conv_2d(6, 32, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 69, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 68, 1152);
    conv_2d(1, 68, 1152);
    conv_2d(2, 68, 4032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 70, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 69, 4608);
    conv_2d(1, 69, 4608);
    conv_2d(2, 69, 35136);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 34, 48672);
    conv_2d(3, 34, 0);
    conv_2d(4, 34, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 33, 1728);
    conv_2d(5, 33, 2016);
    add(0, 33);
    conv_2d(6, 33, 288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 16, 864);
    conv_2d(7, 16, 864);
    conv_2d(8, 16, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 15, 1728);
    conv_2d(9, 15, -1);
    add(1, 15);
    conv_2d(10, 15, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 14, 1728);
    conv_2d(11, 14, -1);
    add(2, 14);
    conv_2d(12, 14, 288);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 71, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 70, 1152);
    conv_2d(1, 70, 1152);
    conv_2d(2, 70, 8640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 71, 9216);
    conv_2d(1, 71, 6912);
    conv_2d(2, 71, 31680);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 35, 47520);
    conv_2d(3, 35, 0);
    conv_2d(4, 35, 1728);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 34, 3456);
    conv_2d(5, 34, 3744);
    add(0, 34);
    conv_2d(6, 34, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 35, 5472);
    conv_2d(5, 35, 288);
    add(0, 35);
    conv_2d(6, 35, 3744);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 17, 4320);
    conv_2d(7, 17, 0);
    conv_2d(8, 17, 1728);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 16, 3456);
    conv_2d(9, 16, -1);
    add(1, 16);
    conv_2d(10, 16, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 15, 5184);
    conv_2d(11, 15, -1);
    add(2, 15);
    conv_2d(12, 15, 5184);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 7, 6048);
    conv_2d(13, 7, -1);
    conv_2d(14, 7, 5184);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 6, 6480);
    conv_2d(15, 6, -1);
    add(3, 6);
    conv_2d(16, 6, 5184);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 5, 6480);
    conv_2d(17, 5, -1);
    add(4, 5);
    conv_2d(18, 5, 5184);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 4, 6480);
    conv_2d(19, 4, -1);
    add(5, 4);
    conv_2d(20, 4, 5408);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 3, 6480);
    conv_2d(21, 3, -1);
    conv_2d(22, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 2, 8640);
    conv_2d(23, 2, -1);
    add(6, 2);
    conv_2d(24, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 1, 12096);
    conv_2d(25, 1, -1);
    add(7, 1);
    conv_2d(26, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(13, 0, 13056);
    conv_2d(27, 0, -1);
    conv_2d(28, 0, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 17, 15552);
    conv_2d(9, 17, -1);
    add(1, 17);
    conv_2d(10, 17, 12096);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 16, 15552);
    conv_2d(11, 16, -1);
    add(2, 16);
    conv_2d(12, 16, 15552);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 17, 20544);
    conv_2d(11, 17, -1);
    add(2, 17);
    conv_2d(12, 17, 2016);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 8, 2592);
    conv_2d(13, 8, -1);
    conv_2d(14, 8, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 7, 1296);
    conv_2d(15, 7, -1);
    add(3, 7);
    conv_2d(16, 7, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 6, 1296);
    conv_2d(17, 6, -1);
    add(4, 6);
    conv_2d(18, 6, 0);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 5, 1296);
    conv_2d(19, 5, -1);
    add(5, 5);
    conv_2d(20, 5, 224);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 4, 1296);
    conv_2d(21, 4, -1);
    conv_2d(22, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 3, 3456);
    conv_2d(23, 3, -1);
    add(6, 3);
    conv_2d(24, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 2, 10368);
    conv_2d(25, 2, -1);
    add(7, 2);
    conv_2d(26, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 8, 11664);
    conv_2d(15, 8, -1);
    add(3, 8);
    conv_2d(16, 8, 10368);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 7, 11664);
    conv_2d(17, 7, -1);
    add(4, 7);
    conv_2d(18, 7, 10368);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 6, 11664);
    conv_2d(19, 6, -1);
    add(5, 6);
    conv_2d(20, 6, 10592);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 5, 11664);
    conv_2d(21, 5, -1);
    conv_2d(22, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 4, 13824);
    conv_2d(23, 4, -1);
    add(6, 4);
    conv_2d(24, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 3, 13824);
    conv_2d(25, 3, -1);
    add(7, 3);
    conv_2d(26, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(13, 1, 13056);
    conv_2d(27, 1, -1);
    conv_2d(28, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(14, 0, 5136);
    conv_2d(29, 0, -1);
    add(8, 0);
    conv_2d(30, 0, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 8, 4752);
    conv_2d(17, 8, -1);
    add(4, 8);
    conv_2d(18, 8, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 7, 4752);
    conv_2d(19, 7, -1);
    add(5, 7);
    conv_2d(20, 7, 3680);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 6, 4752);
    conv_2d(21, 6, -1);
    conv_2d(22, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 5, 13616);
    conv_2d(23, 5, -1);
    add(6, 5);
    conv_2d(24, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 4, 13616);
    conv_2d(25, 4, -1);
    add(7, 4);
    conv_2d(26, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 8, 9936);
    conv_2d(19, 8, -1);
    add(5, 8);
    conv_2d(20, 8, 8864);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 7, 9936);
    conv_2d(21, 7, -1);
    conv_2d(22, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 6, 17184);
    conv_2d(23, 6, -1);
    add(6, 6);
    conv_2d(24, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 5, 17184);
    conv_2d(25, 5, -1);
    add(7, 5);
    conv_2d(26, 5, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(13, 2, 13056);
    conv_2d(27, 2, -1);
    conv_2d(28, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(14, 1, 3408);
    conv_2d(29, 1, -1);
    add(8, 1);
    conv_2d(30, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(15, 0, 3408);
    conv_2d(31, 0, -1);
    add(9, 0);
    conv_2d(32, 0, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 8, 3024);
    conv_2d(21, 8, -1);
    conv_2d(22, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 7, 12672);
    conv_2d(23, 7, -1);
    add(6, 7);
    conv_2d(24, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 6, 12960);
    conv_2d(25, 6, -1);
    add(7, 6);
    conv_2d(26, 6, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 8, 12960);
    conv_2d(23, 8, -1);
    add(6, 8);
    conv_2d(24, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 7, 8640);
    conv_2d(25, 7, -1);
    add(7, 7);
    conv_2d(26, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(13, 3, 9600);
    conv_2d(27, 3, -1);
    conv_2d(28, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(14, 2, 1680);
    conv_2d(29, 2, -1);
    add(8, 2);
    conv_2d(30, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(15, 1, 1680);
    conv_2d(31, 1, -1);
    add(9, 1);
    conv_2d(32, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(16, 0, 1680);
    conv_2d(33, 0, -1);
    conv_2d(34, 0, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 8, 6864);
    conv_2d(25, 8, -1);
    add(7, 8);
    conv_2d(26, 8, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(13, 4, 2688);
    conv_2d(27, 4, -1);
    conv_2d(28, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(14, 3, 3360);
    conv_2d(29, 3, -1);
    add(8, 3);
    conv_2d(30, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(15, 2, 5040);
    conv_2d(31, 2, -1);
    add(9, 2);
    conv_2d(32, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(16, 1, 5040);
    conv_2d(33, 1, -1);
    conv_2d(34, 1, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(14, 4, 5040);
    conv_2d(29, 4, -1);
    add(8, 4);
    conv_2d(30, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(15, 3, 6720);
    conv_2d(31, 3, -1);
    add(9, 3);
    conv_2d(32, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(16, 2, 8400);
    conv_2d(33, 2, -1);
    conv_2d(34, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(15, 4, 8400);
    conv_2d(31, 4, -1);
    add(9, 4);
    conv_2d(32, 4, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(16, 3, 6720);
    conv_2d(33, 3, -1);
    conv_2d(34, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(16, 4, 3360);
    conv_2d(33, 4, -1);
    conv_2d(34, 4, -1);
    concatenation(0);
    average_pool_2d(0, 11648);
    conv_2d(35, -1);
    reshape(0);
}
