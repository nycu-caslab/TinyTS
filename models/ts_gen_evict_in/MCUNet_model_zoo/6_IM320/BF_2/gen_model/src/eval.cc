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
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 0, 2816);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 1, 5632);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 2, 8448);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 3, 11264);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 4, 14080);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 5, 16896);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 6, 19712);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 7, 22528);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 8, 25344);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 9, 28160);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 10, 30976);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 11, 33792);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 12, 36608);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 13, 39424);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 14, 42240);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 15, 45056);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 16, 47872);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 17, 50688);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 18, 53504);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 19, 56320);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 20, 59136);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 21, 61952);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 22, 64768);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 23, 67584);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 24, 70400);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 25, 73216);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 26, 76032);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 27, 78848);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 28, 81664);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 29, 84480);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 30, 87296);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 31, 90112);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 32, 92928);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 33, 95744);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 34, 98560);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 35, 101376);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 36, 104192);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 37, 107008);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 38, 109824);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 39, 112640);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 40, 115456);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 41, 118272);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 42, 121088);
    conv_2d_tiny_3x3_ich3_st2_pad1(0, 43, 123904);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 0, 126720);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 1, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 2, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 3, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 4, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 5, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 6, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 7, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 8, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 9, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 10, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 11, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 12, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 13, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 14, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 15, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 16, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 17, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 18, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 19, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 20, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 21, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 22, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 23, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 24, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 25, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 26, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 27, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 28, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 29, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 30, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 31, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 32, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 33, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 34, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 35, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 36, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 37, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 38, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 39, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 40, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 41, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 42, 129536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 43, 129536);
    conv_2d(1, 0, 119680);
    conv_2d(1, 1, 121088);
    conv_2d(1, 2, 122496);
    conv_2d(1, 3, 0);
    conv_2d(1, 4, 0);
    conv_2d(1, 5, 0);
    conv_2d(1, 6, 0);
    conv_2d(1, 7, 0);
    conv_2d(1, 8, 0);
    conv_2d(1, 9, 0);
    conv_2d(1, 10, 0);
    conv_2d(1, 11, 0);
    conv_2d(1, 12, 0);
    conv_2d(1, 13, 0);
    conv_2d(1, 14, 0);
    conv_2d(1, 15, 0);
    conv_2d(1, 16, 0);
    conv_2d(1, 17, 0);
    conv_2d(1, 18, 0);
    conv_2d(1, 19, 0);
    conv_2d(1, 20, 0);
    conv_2d(1, 21, 0);
    conv_2d(1, 22, 0);
    conv_2d(1, 23, 0);
    conv_2d(1, 24, 0);
    conv_2d(1, 25, 0);
    conv_2d(1, 26, 0);
    conv_2d(1, 27, 0);
    conv_2d(1, 28, 0);
    conv_2d(1, 29, 0);
    conv_2d(1, 30, 0);
    conv_2d(1, 31, 0);
    conv_2d(1, 32, 0);
    conv_2d(1, 33, 0);
    conv_2d(1, 34, 0);
    conv_2d(1, 35, 0);
    conv_2d(1, 36, 0);
    conv_2d(1, 37, 0);
    conv_2d(1, 38, 0);
    conv_2d(1, 39, 0);
    conv_2d(1, 40, 0);
    conv_2d(1, 41, 0);
    conv_2d(1, 42, 0);
    conv_2d(1, 43, 0);
    conv_2d(2, 0, 4224);
    conv_2d(2, 1, 8448);
    conv_2d(2, 2, 12672);
    conv_2d(2, 3, 16896);
    conv_2d(2, 4, 21120);
    conv_2d(2, 5, 25344);
    conv_2d(2, 6, 29568);
    conv_2d(2, 7, 33792);
    conv_2d(2, 8, 38016);
    conv_2d(2, 9, 42240);
    conv_2d(2, 10, 46464);
    conv_2d(2, 11, 50688);
    conv_2d(2, 12, 54912);
    conv_2d(2, 13, 59136);
    conv_2d(2, 14, 63360);
    conv_2d(2, 15, 67584);
    conv_2d(2, 16, 71808);
    conv_2d(2, 17, 76032);
    conv_2d(2, 18, 80256);
    conv_2d(2, 19, 84480);
    conv_2d(2, 20, 88704);
    conv_2d(2, 21, 92928);
    conv_2d(2, 22, 97152);
    conv_2d(2, 23, 101376);
    conv_2d(2, 24, 105600);
    conv_2d(2, 25, 109824);
    conv_2d(2, 26, 114048);
    conv_2d(2, 27, 118272);
    conv_2d(2, 28, 122496);
    conv_2d(2, 29, 126720);
    conv_2d(2, 30, 130944);
    conv_2d(2, 31, 135168);
    conv_2d(2, 32, 139392);
    conv_2d(2, 33, 143616);
    conv_2d(2, 34, 147840);
    conv_2d(2, 35, 152064);
    conv_2d(2, 36, 156288);
    conv_2d(2, 37, 160512);
    conv_2d(2, 38, 164736);
    conv_2d(2, 39, 168960);
    conv_2d(2, 40, 173184);
    conv_2d(2, 41, 178816);
    conv_2d(2, 42, 183040);
    conv_2d(2, 43, 187264);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 0, 187968);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 1, 190080);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 2, 2112);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 3, 4224);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 4, 6336);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 5, 8448);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 6, 10560);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 7, 12672);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 8, 14784);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 9, 16896);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 10, 19008);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 11, 21120);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 12, 23232);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 13, 25344);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 14, 27456);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 15, 29568);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 16, 31680);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 17, 33792);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 18, 35904);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 19, 38016);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 20, 40128);
    depthwise_conv_2d_tiny_kernel7x7_stride2(1, 21, 42240);
    conv_2d(3, 0, 42240);
    conv_2d(3, 1, 42240);
    conv_2d(3, 2, 42240);
    conv_2d(3, 3, 0);
    conv_2d(3, 4, 0);
    conv_2d(3, 5, 0);
    conv_2d(3, 6, 0);
    conv_2d(3, 7, 0);
    conv_2d(3, 8, 0);
    conv_2d(3, 9, 0);
    conv_2d(3, 10, 0);
    conv_2d(3, 11, 0);
    conv_2d(3, 12, 0);
    conv_2d(3, 13, 0);
    conv_2d(3, 14, 0);
    conv_2d(3, 15, 0);
    conv_2d(3, 16, 0);
    conv_2d(3, 17, 0);
    conv_2d(3, 18, 0);
    conv_2d(3, 19, 0);
    conv_2d(3, 20, 0);
    conv_2d(3, 21, 0);
    conv_2d(4, 0, 7040);
    conv_2d(4, 1, 14080);
    conv_2d(4, 2, 21120);
    conv_2d(4, 3, 28160);
    conv_2d(4, 4, 35200);
    conv_2d(4, 5, 42240);
    conv_2d(4, 6, 49280);
    conv_2d(4, 7, 56320);
    conv_2d(4, 8, 63360);
    conv_2d(4, 9, 70400);
    conv_2d(4, 10, 77440);
    conv_2d(4, 11, 84480);
    conv_2d(4, 12, 91520);
    conv_2d(4, 13, 98560);
    conv_2d(4, 14, 105600);
    conv_2d(4, 15, 112640);
    conv_2d(4, 16, 119680);
    conv_2d(4, 17, 126720);
    conv_2d(4, 18, 133760);
    conv_2d(4, 19, 140800);
    conv_2d(4, 20, 147840);
    conv_2d(4, 21, 154880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 161920);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 8, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 9, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 10, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 11, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 12, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 13, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 14, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 15, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 16, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 17, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 18, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 19, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 20, 199936);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 21, 199936);
    conv_2d(5, 0, -1);
    conv_2d(5, 1, -1);
    conv_2d(5, 2, -1);
    conv_2d(5, 3, -1);
    conv_2d(5, 4, -1);
    conv_2d(5, 5, -1);
    conv_2d(5, 6, -1);
    conv_2d(5, 7, -1);
    conv_2d(5, 8, -1);
    conv_2d(5, 9, -1);
    conv_2d(5, 10, -1);
    conv_2d(5, 11, -1);
    conv_2d(5, 12, -1);
    conv_2d(5, 13, -1);
    conv_2d(5, 14, -1);
    conv_2d(5, 15, -1);
    conv_2d(5, 16, -1);
    conv_2d(5, 17, -1);
    conv_2d(5, 18, -1);
    conv_2d(5, 19, -1);
    conv_2d(5, 20, -1);
    conv_2d(5, 21, -1);
    add(0, 0);
    add(0, 1);
    add(0, 2);
    add(0, 3);
    add(0, 4);
    add(0, 5);
    add(0, 6);
    add(0, 7);
    add(0, 8);
    add(0, 9);
    add(0, 10);
    add(0, 11);
    add(0, 12);
    add(0, 13);
    add(0, 14);
    add(0, 15);
    add(0, 16);
    add(0, 17);
    add(0, 18);
    add(0, 19);
    add(0, 20);
    add(0, 21);
    conv_2d(6, 0, 7040);
    conv_2d(6, 1, 14080);
    conv_2d(6, 2, 21120);
    conv_2d(6, 3, 28160);
    conv_2d(6, 4, 35200);
    conv_2d(6, 5, 42240);
    conv_2d(6, 6, 49280);
    conv_2d(6, 7, 56320);
    conv_2d(6, 8, 63360);
    conv_2d(6, 9, 70400);
    conv_2d(6, 10, 77440);
    conv_2d(6, 11, 84480);
    conv_2d(6, 12, 91520);
    conv_2d(6, 13, 98560);
    conv_2d(6, 14, 105600);
    conv_2d(6, 15, 112640);
    conv_2d(6, 16, 119680);
    conv_2d(6, 17, 126720);
    conv_2d(6, 18, 133760);
    conv_2d(6, 19, 140800);
    conv_2d(6, 20, 147840);
    conv_2d(6, 21, 154880);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 0, 161920);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 1, 168960);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 2, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 3, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 4, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 5, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 6, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 7, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 8, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 9, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 10, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 11, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 12, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 13, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 14, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 15, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 16, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 17, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 18, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 19, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 20, 198528);
    depthwise_conv_2d_tiny_kernel7x7_stride1(3, 21, 198528);
    conv_2d(7, 0, -1);
    conv_2d(7, 1, -1);
    conv_2d(7, 2, -1);
    conv_2d(7, 3, -1);
    conv_2d(7, 4, -1);
    conv_2d(7, 5, -1);
    conv_2d(7, 6, -1);
    conv_2d(7, 7, -1);
    conv_2d(7, 8, -1);
    conv_2d(7, 9, -1);
    conv_2d(7, 10, -1);
    conv_2d(7, 11, -1);
    conv_2d(7, 12, -1);
    conv_2d(7, 13, -1);
    conv_2d(7, 14, -1);
    conv_2d(7, 15, -1);
    conv_2d(7, 16, -1);
    conv_2d(7, 17, -1);
    conv_2d(7, 18, -1);
    conv_2d(7, 19, -1);
    conv_2d(7, 20, -1);
    conv_2d(7, 21, -1);
    add(1, 0);
    add(1, 1);
    add(1, 2);
    add(1, 3);
    add(1, 4);
    add(1, 5);
    add(1, 6);
    add(1, 7);
    add(1, 8);
    add(1, 9);
    add(1, 10);
    add(1, 11);
    add(1, 12);
    add(1, 13);
    add(1, 14);
    add(1, 15);
    add(1, 16);
    add(1, 17);
    add(1, 18);
    add(1, 19);
    add(1, 20);
    add(1, 21);
    conv_2d(8, 0, 5632);
    conv_2d(8, 1, 11264);
    conv_2d(8, 2, 16896);
    conv_2d(8, 3, 22528);
    conv_2d(8, 4, 28160);
    conv_2d(8, 5, 33792);
    conv_2d(8, 6, 39424);
    conv_2d(8, 7, 45056);
    conv_2d(8, 8, 50688);
    conv_2d(8, 9, 56320);
    conv_2d(8, 10, 61952);
    conv_2d(8, 11, 67584);
    conv_2d(8, 12, 73216);
    conv_2d(8, 13, 78848);
    conv_2d(8, 14, 84480);
    conv_2d(8, 15, 90112);
    conv_2d(8, 16, 95744);
    conv_2d(8, 17, 101376);
    conv_2d(8, 18, 107008);
    conv_2d(8, 19, 112640);
    conv_2d(8, 20, 118272);
    conv_2d(8, 21, 123904);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 0, 129536);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 1, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 2, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 3, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 4, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 5, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 6, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 7, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 8, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 9, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 10, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 11, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 12, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 13, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 14, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 15, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 16, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 17, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 18, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 19, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 20, 166144);
    depthwise_conv_2d_tiny_kernel5x5_stride1(4, 21, 166144);
    conv_2d(9, 0, -1);
    conv_2d(9, 1, -1);
    conv_2d(9, 2, -1);
    conv_2d(9, 3, -1);
    conv_2d(9, 4, -1);
    conv_2d(9, 5, -1);
    conv_2d(9, 6, -1);
    conv_2d(9, 7, -1);
    conv_2d(9, 8, -1);
    conv_2d(9, 9, -1);
    conv_2d(9, 10, -1);
    conv_2d(9, 11, -1);
    conv_2d(9, 12, -1);
    conv_2d(9, 13, -1);
    conv_2d(9, 14, -1);
    conv_2d(9, 15, -1);
    conv_2d(9, 16, -1);
    conv_2d(9, 17, -1);
    conv_2d(9, 18, -1);
    conv_2d(9, 19, -1);
    conv_2d(9, 20, -1);
    conv_2d(9, 21, -1);
    add(2, 0);
    add(2, 1);
    add(2, 2);
    add(2, 3);
    add(2, 4);
    add(2, 5);
    add(2, 6);
    add(2, 7);
    add(2, 8);
    add(2, 9);
    add(2, 10);
    add(2, 11);
    add(2, 12);
    add(2, 13);
    add(2, 14);
    add(2, 15);
    add(2, 16);
    add(2, 17);
    add(2, 18);
    add(2, 19);
    add(2, 20);
    add(2, 21);
    conv_2d(10, 0, 7040);
    conv_2d(10, 1, 14080);
    conv_2d(10, 2, 21120);
    conv_2d(10, 3, 28160);
    conv_2d(10, 4, 36608);
    conv_2d(10, 5, 43648);
    conv_2d(10, 6, 50688);
    conv_2d(10, 7, 57728);
    conv_2d(10, 8, 64768);
    conv_2d(10, 9, 71808);
    conv_2d(10, 10, 78848);
    conv_2d(10, 11, 85888);
    conv_2d(10, 12, 92928);
    conv_2d(10, 13, 99968);
    conv_2d(10, 14, 107008);
    conv_2d(10, 15, 114048);
    conv_2d(10, 16, 121088);
    conv_2d(10, 17, 128128);
    conv_2d(10, 18, 135168);
    conv_2d(10, 19, 142208);
    conv_2d(10, 20, 149248);
    conv_2d(10, 21, 156288);
    depthwise_conv_2d_tiny_kernel5x5_stride2(5, 0, 158400);
    depthwise_conv_2d_tiny_kernel5x5_stride2(5, 1, 3520);
    depthwise_conv_2d_tiny_kernel5x5_stride2(5, 2, 7040);
    depthwise_conv_2d_tiny_kernel5x5_stride2(5, 3, 10560);
    depthwise_conv_2d_tiny_kernel5x5_stride2(5, 4, 14080);
    depthwise_conv_2d_tiny_kernel5x5_stride2(5, 5, 17600);
    depthwise_conv_2d_tiny_kernel5x5_stride2(5, 6, 21120);
    depthwise_conv_2d_tiny_kernel5x5_stride2(5, 7, 24640);
    depthwise_conv_2d_tiny_kernel5x5_stride2(5, 8, 28160);
    depthwise_conv_2d_tiny_kernel5x5_stride2(5, 9, 31680);
    depthwise_conv_2d_tiny_kernel5x5_stride2(5, 10, 35200);
    conv_2d(11, 0, -1);
    conv_2d(11, 1, -1);
    conv_2d(11, 2, -1);
    conv_2d(11, 3, -1);
    conv_2d(11, 4, -1);
    conv_2d(11, 5, -1);
    conv_2d(11, 6, -1);
    conv_2d(11, 7, -1);
    conv_2d(11, 8, -1);
    conv_2d(11, 9, -1);
    conv_2d(11, 10, -1);
    conv_2d(12, 0, 5280);
    conv_2d(12, 1, 10560);
    conv_2d(12, 2, 15840);
    conv_2d(12, 3, 21120);
    conv_2d(12, 4, 26400);
    conv_2d(12, 5, 31680);
    conv_2d(12, 6, 36960);
    conv_2d(12, 7, 42240);
    conv_2d(12, 8, 47520);
    conv_2d(12, 9, 52800);
    conv_2d(12, 10, 58080);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 0, 63360);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 1, 80256);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 2, 80256);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 3, 80256);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 4, 80256);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 5, 80256);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 6, 80256);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 7, 80256);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 8, 80256);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 9, 80256);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 10, 80256);
    conv_2d(13, 0, -1);
    conv_2d(13, 1, -1);
    conv_2d(13, 2, -1);
    conv_2d(13, 3, -1);
    conv_2d(13, 4, -1);
    conv_2d(13, 5, -1);
    conv_2d(13, 6, -1);
    conv_2d(13, 7, -1);
    conv_2d(13, 8, -1);
    conv_2d(13, 9, -1);
    conv_2d(13, 10, -1);
    add(3, 0);
    add(3, 1);
    add(3, 2);
    add(3, 3);
    add(3, 4);
    add(3, 5);
    add(3, 6);
    add(3, 7);
    add(3, 8);
    add(3, 9);
    add(3, 10);
    conv_2d(14, 0, 5280);
    conv_2d(14, 1, 10560);
    conv_2d(14, 2, 15840);
    conv_2d(14, 3, 21120);
    conv_2d(14, 4, 26400);
    conv_2d(14, 5, 31680);
    conv_2d(14, 6, 36960);
    conv_2d(14, 7, 42240);
    conv_2d(14, 8, 47520);
    conv_2d(14, 9, 52800);
    conv_2d(14, 10, 58080);
    depthwise_conv_2d_tiny_kernel5x5_stride1(7, 0, 63360);
    depthwise_conv_2d_tiny_kernel5x5_stride1(7, 1, 79200);
    depthwise_conv_2d_tiny_kernel5x5_stride1(7, 2, 79200);
    depthwise_conv_2d_tiny_kernel5x5_stride1(7, 3, 79200);
    depthwise_conv_2d_tiny_kernel5x5_stride1(7, 4, 79200);
    depthwise_conv_2d_tiny_kernel5x5_stride1(7, 5, 79200);
    depthwise_conv_2d_tiny_kernel5x5_stride1(7, 6, 79200);
    depthwise_conv_2d_tiny_kernel5x5_stride1(7, 7, 79200);
    depthwise_conv_2d_tiny_kernel5x5_stride1(7, 8, 79200);
    depthwise_conv_2d_tiny_kernel5x5_stride1(7, 9, 79200);
    depthwise_conv_2d_tiny_kernel5x5_stride1(7, 10, 79200);
    conv_2d(15, 0, -1);
    conv_2d(15, 1, -1);
    conv_2d(15, 2, -1);
    conv_2d(15, 3, -1);
    conv_2d(15, 4, -1);
    conv_2d(15, 5, -1);
    conv_2d(15, 6, -1);
    conv_2d(15, 7, -1);
    conv_2d(15, 8, -1);
    conv_2d(15, 9, -1);
    conv_2d(15, 10, -1);
    add(4, 0);
    add(4, 1);
    add(4, 2);
    add(4, 3);
    add(4, 4);
    add(4, 5);
    add(4, 6);
    add(4, 7);
    add(4, 8);
    add(4, 9);
    add(4, 10);
    conv_2d(16, 0, 5280);
    conv_2d(16, 1, 11616);
    conv_2d(16, 2, 16896);
    conv_2d(16, 3, 22176);
    conv_2d(16, 4, 27456);
    conv_2d(16, 5, 32736);
    conv_2d(16, 6, 38016);
    conv_2d(16, 7, 43296);
    conv_2d(16, 8, 48576);
    conv_2d(16, 9, 53856);
    conv_2d(16, 10, 59136);
    depthwise_conv_2d_tiny_kernel3x3_stride2(8, 0, 60720);
    depthwise_conv_2d_tiny_kernel3x3_stride2(8, 1, 2640);
    depthwise_conv_2d_tiny_kernel3x3_stride2(8, 2, 5280);
    depthwise_conv_2d_tiny_kernel3x3_stride2(8, 3, 7920);
    depthwise_conv_2d_tiny_kernel3x3_stride2(8, 4, 10560);
    depthwise_conv_2d_tiny_kernel3x3_stride2(8, 5, 11888);
    conv_2d(17, 0, -1);
    conv_2d(17, 1, -1);
    conv_2d(17, 2, -1);
    conv_2d(17, 3, -1);
    conv_2d(17, 4, -1);
    conv_2d(17, 5, -1);
    conv_2d(18, 0, -1);
    conv_2d(18, 1, -1);
    conv_2d(18, 2, -1);
    conv_2d(18, 3, -1);
    conv_2d(18, 4, -1);
    conv_2d(18, 5, -1);
    depthwise_conv_2d_tiny_kernel7x7_stride1(9, 0, 31680);
    depthwise_conv_2d_tiny_kernel7x7_stride1(9, 1, 36960);
    depthwise_conv_2d_tiny_kernel7x7_stride1(9, 2, 49728);
    depthwise_conv_2d_tiny_kernel7x7_stride1(9, 3, 49728);
    depthwise_conv_2d_tiny_kernel7x7_stride1(9, 4, 49728);
    depthwise_conv_2d_tiny_kernel7x7_stride1(9, 5, 13200);
    conv_2d(19, 0, -1);
    conv_2d(19, 1, -1);
    conv_2d(19, 2, -1);
    conv_2d(19, 3, -1);
    conv_2d(19, 4, -1);
    conv_2d(19, 5, -1);
    add(5, 0);
    add(5, 1);
    add(5, 2);
    add(5, 3);
    add(5, 4);
    add(5, 5);
    conv_2d(20, 0, -1);
    conv_2d(20, 1, -1);
    conv_2d(20, 2, -1);
    conv_2d(20, 3, -1);
    conv_2d(20, 4, -1);
    conv_2d(20, 5, -1);
    depthwise_conv_2d_tiny_kernel5x5_stride1(10, 0, 21120);
    depthwise_conv_2d_tiny_kernel5x5_stride1(10, 1, 31248);
    depthwise_conv_2d_tiny_kernel5x5_stride1(10, 2, 31248);
    depthwise_conv_2d_tiny_kernel5x5_stride1(10, 3, 31248);
    depthwise_conv_2d_tiny_kernel5x5_stride1(10, 4, 31248);
    depthwise_conv_2d_tiny_kernel5x5_stride1(10, 5, 12320);
    conv_2d(21, 0, -1);
    conv_2d(21, 1, -1);
    conv_2d(21, 2, -1);
    conv_2d(21, 3, -1);
    conv_2d(21, 4, -1);
    conv_2d(21, 5, -1);
    add(6, 0);
    add(6, 1);
    add(6, 2);
    add(6, 3);
    add(6, 4);
    add(6, 5);
    conv_2d(22, 0, -1);
    conv_2d(22, 1, -1);
    conv_2d(22, 2, -1);
    conv_2d(22, 3, -1);
    conv_2d(22, 4, -1);
    conv_2d(22, 5, -1);
    depthwise_conv_2d_tiny_kernel5x5_stride1(11, 0, 26400);
    depthwise_conv_2d_tiny_kernel5x5_stride1(11, 1, 33008);
    depthwise_conv_2d_tiny_kernel5x5_stride1(11, 2, 33008);
    depthwise_conv_2d_tiny_kernel5x5_stride1(11, 3, 33008);
    depthwise_conv_2d_tiny_kernel5x5_stride1(11, 4, 33008);
    depthwise_conv_2d_tiny_kernel5x5_stride1(11, 5, 15408);
    conv_2d(23, 0, -1);
    conv_2d(23, 1, -1);
    conv_2d(23, 2, -1);
    conv_2d(23, 3, -1);
    conv_2d(23, 4, -1);
    conv_2d(23, 5, -1);
    conv_2d(24, 0, 5280);
    conv_2d(24, 1, 10560);
    conv_2d(24, 2, 15840);
    conv_2d(24, 3, 21120);
    conv_2d(24, 4, 26400);
    conv_2d(24, 5, 26400);
    depthwise_conv_2d_tiny_kernel7x7_stride1(12, 0, 31680);
    depthwise_conv_2d_tiny_kernel7x7_stride1(12, 1, 36960);
    depthwise_conv_2d_tiny_kernel7x7_stride1(12, 2, 50688);
    depthwise_conv_2d_tiny_kernel7x7_stride1(12, 3, 50688);
    depthwise_conv_2d_tiny_kernel7x7_stride1(12, 4, 50688);
    depthwise_conv_2d_tiny_kernel7x7_stride1(12, 5, 13200);
    conv_2d(25, 0, -1);
    conv_2d(25, 1, -1);
    conv_2d(25, 2, -1);
    conv_2d(25, 3, -1);
    conv_2d(25, 4, -1);
    conv_2d(25, 5, -1);
    add(7, 0);
    add(7, 1);
    add(7, 2);
    add(7, 3);
    add(7, 4);
    add(7, 5);
    conv_2d(26, 0, 5280);
    conv_2d(26, 1, 10560);
    conv_2d(26, 2, 15840);
    conv_2d(26, 3, 21120);
    conv_2d(26, 4, 26400);
    conv_2d(26, 5, 26400);
    depthwise_conv_2d_tiny_kernel3x3_stride1(13, 0, 31680);
    depthwise_conv_2d_tiny_kernel3x3_stride1(13, 1, 45408);
    depthwise_conv_2d_tiny_kernel3x3_stride1(13, 2, 45408);
    depthwise_conv_2d_tiny_kernel3x3_stride1(13, 3, 45408);
    depthwise_conv_2d_tiny_kernel3x3_stride1(13, 4, 45408);
    depthwise_conv_2d_tiny_kernel3x3_stride1(13, 5, 18480);
    conv_2d(27, 0, -1);
    conv_2d(27, 1, -1);
    conv_2d(27, 2, -1);
    conv_2d(27, 3, -1);
    conv_2d(27, 4, -1);
    conv_2d(27, 5, -1);
    add(8, 0);
    add(8, 1);
    add(8, 2);
    add(8, 3);
    add(8, 4);
    add(8, 5);
    conv_2d(28, 0, 7392);
    conv_2d(28, 1, 13728);
    conv_2d(28, 2, 20064);
    conv_2d(28, 3, 26400);
    conv_2d(28, 4, 33264);
    conv_2d(28, 5, 31680);
    depthwise_conv_2d_tiny_kernel3x3_stride2(14, 0, 38304);
    depthwise_conv_2d_tiny_kernel3x3_stride2(14, 1, 3456);
    depthwise_conv_2d_tiny_kernel3x3_stride2(14, 2, 6912);
    conv_2d(29, 0, -1);
    conv_2d(29, 1, -1);
    conv_2d(29, 2, -1);
    conv_2d(30, 0, -1);
    conv_2d(30, 1, -1);
    conv_2d(30, 2, -1);
    depthwise_conv_2d_tiny_kernel7x7_stride1(15, 0, 23040);
    depthwise_conv_2d_tiny_kernel7x7_stride1(15, 1, 28800);
    depthwise_conv_2d_tiny_kernel7x7_stride1(15, 2, 34560);
    conv_2d(31, 0, -1);
    conv_2d(31, 1, -1);
    conv_2d(31, 2, -1);
    add(9, 0);
    add(9, 1);
    add(9, 2);
    conv_2d(32, 0, -1);
    conv_2d(32, 1, -1);
    conv_2d(32, 2, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(16, 0, 18432);
    depthwise_conv_2d_tiny_kernel3x3_stride1(16, 1, 26496);
    depthwise_conv_2d_tiny_kernel3x3_stride1(16, 2, 26496);
    conv_2d(33, 0, -1);
    conv_2d(33, 1, -1);
    conv_2d(33, 2, -1);
    add(10, 0);
    add(10, 1);
    add(10, 2);
    conv_2d(34, 0, -1);
    conv_2d(34, 1, -1);
    conv_2d(34, 2, -1);
    depthwise_conv_2d_tiny_kernel7x7_stride1(17, 0, 23040);
    depthwise_conv_2d_tiny_kernel7x7_stride1(17, 1, 28800);
    depthwise_conv_2d_tiny_kernel7x7_stride1(17, 2, 34560);
    conv_2d(35, 0, -1);
    conv_2d(35, 1, -1);
    conv_2d(35, 2, -1);
    concatenation(0);
    average_pool_2d(0, 5920);
    conv_2d(36, -1);
    reshape(0);
}
