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
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 0, 187968);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 1, 2112);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 2, 4224);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 3, 6336);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 4, 8448);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 5, 10560);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 6, 12672);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 7, 14784);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 8, 16896);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 9, 19008);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 10, 21120);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 11, 23232);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 12, 25344);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 13, 27456);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 14, 29568);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 15, 31680);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 16, 33792);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 17, 35904);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 18, 38016);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 19, 40128);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 20, 42240);
    depthwise_conv_2d_tiny_kernel5x5_stride2(1, 21, 44352);
    conv_2d(3, 0, 44352);
    conv_2d(3, 1, 44352);
    conv_2d(3, 2, 0);
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
    conv_2d(4, 0, 4224);
    conv_2d(4, 1, 8448);
    conv_2d(4, 2, 12672);
    conv_2d(4, 3, 16896);
    conv_2d(4, 4, 21120);
    conv_2d(4, 5, 25344);
    conv_2d(4, 6, 29568);
    conv_2d(4, 7, 33792);
    conv_2d(4, 8, 38016);
    conv_2d(4, 9, 42240);
    conv_2d(4, 10, 46464);
    conv_2d(4, 11, 50688);
    conv_2d(4, 12, 54912);
    conv_2d(4, 13, 59136);
    conv_2d(4, 14, 63360);
    conv_2d(4, 15, 67584);
    conv_2d(4, 16, 71808);
    conv_2d(4, 17, 76032);
    conv_2d(4, 18, 80256);
    conv_2d(4, 19, 84480);
    conv_2d(4, 20, 88704);
    conv_2d(4, 21, 92928);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 97152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 8, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 9, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 10, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 11, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 12, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 13, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 14, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 15, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 16, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 17, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 18, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 19, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 20, 132352);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 21, 132352);
    conv_2d(5, 0, 85888);
    conv_2d(5, 1, 87296);
    conv_2d(5, 2, 88704);
    conv_2d(5, 3, 1408);
    conv_2d(5, 4, 2816);
    conv_2d(5, 5, 4224);
    conv_2d(5, 6, 5632);
    conv_2d(5, 7, 7040);
    conv_2d(5, 8, 8448);
    conv_2d(5, 9, 9856);
    conv_2d(5, 10, 11264);
    conv_2d(5, 11, 12672);
    conv_2d(5, 12, 14080);
    conv_2d(5, 13, 15488);
    conv_2d(5, 14, 16896);
    conv_2d(5, 15, 18304);
    conv_2d(5, 16, 19712);
    conv_2d(5, 17, 21120);
    conv_2d(5, 18, 22528);
    conv_2d(5, 19, 23936);
    conv_2d(5, 20, 25344);
    conv_2d(5, 21, 26752);
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
    conv_2d(6, 0, 4224);
    conv_2d(6, 1, 8448);
    conv_2d(6, 2, 12672);
    conv_2d(6, 3, 16896);
    conv_2d(6, 4, 21120);
    conv_2d(6, 5, 25344);
    conv_2d(6, 6, 29568);
    conv_2d(6, 7, 33792);
    conv_2d(6, 8, 39424);
    conv_2d(6, 9, 43648);
    conv_2d(6, 10, 47872);
    conv_2d(6, 11, 52096);
    conv_2d(6, 12, 56320);
    conv_2d(6, 13, 60544);
    conv_2d(6, 14, 64768);
    conv_2d(6, 15, 68992);
    conv_2d(6, 16, 73216);
    conv_2d(6, 17, 77440);
    conv_2d(6, 18, 81664);
    conv_2d(6, 19, 85888);
    conv_2d(6, 20, 90112);
    conv_2d(6, 21, 94336);
    depthwise_conv_2d_tiny_kernel7x7_stride2(3, 0, 95040);
    depthwise_conv_2d_tiny_kernel7x7_stride2(3, 1, 97152);
    depthwise_conv_2d_tiny_kernel7x7_stride2(3, 2, 2112);
    depthwise_conv_2d_tiny_kernel7x7_stride2(3, 3, 4224);
    depthwise_conv_2d_tiny_kernel7x7_stride2(3, 4, 6336);
    depthwise_conv_2d_tiny_kernel7x7_stride2(3, 5, 8448);
    depthwise_conv_2d_tiny_kernel7x7_stride2(3, 6, 10560);
    depthwise_conv_2d_tiny_kernel7x7_stride2(3, 7, 12672);
    depthwise_conv_2d_tiny_kernel7x7_stride2(3, 8, 14784);
    depthwise_conv_2d_tiny_kernel7x7_stride2(3, 9, 16896);
    depthwise_conv_2d_tiny_kernel7x7_stride2(3, 10, 19008);
    conv_2d(7, 0, 19008);
    conv_2d(7, 1, 19008);
    conv_2d(7, 2, 19008);
    conv_2d(7, 3, 0);
    conv_2d(7, 4, 0);
    conv_2d(7, 5, 0);
    conv_2d(7, 6, 0);
    conv_2d(7, 7, 0);
    conv_2d(7, 8, 0);
    conv_2d(7, 9, 0);
    conv_2d(7, 10, 0);
    conv_2d(8, 0, 2112);
    conv_2d(8, 1, 4224);
    conv_2d(8, 2, 6336);
    conv_2d(8, 3, 8448);
    conv_2d(8, 4, 10560);
    conv_2d(8, 5, 12672);
    conv_2d(8, 6, 14784);
    conv_2d(8, 7, 16896);
    conv_2d(8, 8, 19008);
    conv_2d(8, 9, 21120);
    conv_2d(8, 10, 23232);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 0, 25344);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 1, 35200);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 2, 35200);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 3, 35200);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 4, 35200);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 5, 35200);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 6, 35200);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 7, 35200);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 8, 35200);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 9, 35200);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 10, 35200);
    conv_2d(9, 0, 19712);
    conv_2d(9, 1, 20416);
    conv_2d(9, 2, 21120);
    conv_2d(9, 3, 704);
    conv_2d(9, 4, 1408);
    conv_2d(9, 5, 2112);
    conv_2d(9, 6, 2816);
    conv_2d(9, 7, 3520);
    conv_2d(9, 8, 4224);
    conv_2d(9, 9, 4928);
    conv_2d(9, 10, 5632);
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
    conv_2d(10, 0, 2112);
    conv_2d(10, 1, 4224);
    conv_2d(10, 2, 6336);
    conv_2d(10, 3, 8448);
    conv_2d(10, 4, 10560);
    conv_2d(10, 5, 12672);
    conv_2d(10, 6, 14784);
    conv_2d(10, 7, 16896);
    conv_2d(10, 8, 19008);
    conv_2d(10, 9, 21120);
    conv_2d(10, 10, 23232);
    depthwise_conv_2d_tiny_kernel5x5_stride1(5, 0, 25344);
    depthwise_conv_2d_tiny_kernel5x5_stride1(5, 1, 34496);
    depthwise_conv_2d_tiny_kernel5x5_stride1(5, 2, 34496);
    depthwise_conv_2d_tiny_kernel5x5_stride1(5, 3, 34496);
    depthwise_conv_2d_tiny_kernel5x5_stride1(5, 4, 34496);
    depthwise_conv_2d_tiny_kernel5x5_stride1(5, 5, 34496);
    depthwise_conv_2d_tiny_kernel5x5_stride1(5, 6, 34496);
    depthwise_conv_2d_tiny_kernel5x5_stride1(5, 7, 34496);
    depthwise_conv_2d_tiny_kernel5x5_stride1(5, 8, 34496);
    depthwise_conv_2d_tiny_kernel5x5_stride1(5, 9, 34496);
    depthwise_conv_2d_tiny_kernel5x5_stride1(5, 10, 34496);
    conv_2d(11, 0, 19712);
    conv_2d(11, 1, 20416);
    conv_2d(11, 2, 21120);
    conv_2d(11, 3, 704);
    conv_2d(11, 4, 1408);
    conv_2d(11, 5, 2112);
    conv_2d(11, 6, 2816);
    conv_2d(11, 7, 3520);
    conv_2d(11, 8, 4224);
    conv_2d(11, 9, 4928);
    conv_2d(11, 10, 5632);
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
    conv_2d(12, 0, 2112);
    conv_2d(12, 1, 4224);
    conv_2d(12, 2, 6336);
    conv_2d(12, 3, 8448);
    conv_2d(12, 4, 10560);
    conv_2d(12, 5, 12672);
    conv_2d(12, 6, 14784);
    conv_2d(12, 7, 16896);
    conv_2d(12, 8, 19008);
    conv_2d(12, 9, 21120);
    conv_2d(12, 10, 23232);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 0, 25344);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 1, 33792);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 2, 33792);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 3, 33792);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 4, 33792);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 5, 33792);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 6, 33792);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 7, 33792);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 8, 33792);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 9, 33792);
    depthwise_conv_2d_tiny_kernel5x5_stride1(6, 10, 33792);
    conv_2d(13, 0, 19712);
    conv_2d(13, 1, 20416);
    conv_2d(13, 2, 21120);
    conv_2d(13, 3, 704);
    conv_2d(13, 4, 1408);
    conv_2d(13, 5, 2112);
    conv_2d(13, 6, 2816);
    conv_2d(13, 7, 3520);
    conv_2d(13, 8, 4224);
    conv_2d(13, 9, 4928);
    conv_2d(13, 10, 5632);
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
    conv_2d(14, 0, 4224);
    conv_2d(14, 1, 9152);
    conv_2d(14, 2, 13376);
    conv_2d(14, 3, 17600);
    conv_2d(14, 4, 21824);
    conv_2d(14, 5, 26048);
    conv_2d(14, 6, 30272);
    conv_2d(14, 7, 34496);
    conv_2d(14, 8, 38720);
    conv_2d(14, 9, 42944);
    conv_2d(14, 10, 47168);
    depthwise_conv_2d_tiny_kernel7x7_stride2(7, 0, 48576);
    depthwise_conv_2d_tiny_kernel7x7_stride2(7, 1, 50688);
    depthwise_conv_2d_tiny_kernel7x7_stride2(7, 2, 2112);
    depthwise_conv_2d_tiny_kernel7x7_stride2(7, 3, 4224);
    depthwise_conv_2d_tiny_kernel7x7_stride2(7, 4, 6336);
    depthwise_conv_2d_tiny_kernel7x7_stride2(7, 5, 7392);
    conv_2d(15, 0, -1);
    conv_2d(15, 1, -1);
    conv_2d(15, 2, -1);
    conv_2d(15, 3, -1);
    conv_2d(15, 4, -1);
    conv_2d(15, 5, -1);
    conv_2d(16, 0, 1584);
    conv_2d(16, 1, 3168);
    conv_2d(16, 2, 4752);
    conv_2d(16, 3, 6336);
    conv_2d(16, 4, 7920);
    conv_2d(16, 5, 7920);
    depthwise_conv_2d_tiny_kernel5x5_stride1(8, 0, 9504);
    depthwise_conv_2d_tiny_kernel5x5_stride1(8, 1, 14528);
    depthwise_conv_2d_tiny_kernel5x5_stride1(8, 2, 14528);
    depthwise_conv_2d_tiny_kernel5x5_stride1(8, 3, 14528);
    depthwise_conv_2d_tiny_kernel5x5_stride1(8, 4, 14528);
    depthwise_conv_2d_tiny_kernel5x5_stride1(8, 5, 5552);
    conv_2d(17, 0, -1);
    conv_2d(17, 1, -1);
    conv_2d(17, 2, -1);
    conv_2d(17, 3, -1);
    conv_2d(17, 4, -1);
    conv_2d(17, 5, -1);
    add(4, 0);
    add(4, 1);
    add(4, 2);
    add(4, 3);
    add(4, 4);
    add(4, 5);
    conv_2d(18, 0, 1584);
    conv_2d(18, 1, 3168);
    conv_2d(18, 2, 4752);
    conv_2d(18, 3, 6336);
    conv_2d(18, 4, 7920);
    conv_2d(18, 5, 7920);
    depthwise_conv_2d_tiny_kernel5x5_stride1(9, 0, 9504);
    depthwise_conv_2d_tiny_kernel5x5_stride1(9, 1, 14000);
    depthwise_conv_2d_tiny_kernel5x5_stride1(9, 2, 14000);
    depthwise_conv_2d_tiny_kernel5x5_stride1(9, 3, 14000);
    depthwise_conv_2d_tiny_kernel5x5_stride1(9, 4, 14000);
    depthwise_conv_2d_tiny_kernel5x5_stride1(9, 5, 5552);
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
    conv_2d(20, 0, 1584);
    conv_2d(20, 1, 3168);
    conv_2d(20, 2, 4752);
    conv_2d(20, 3, 6336);
    conv_2d(20, 4, 7920);
    conv_2d(20, 5, 7920);
    depthwise_conv_2d_tiny_kernel5x5_stride1(10, 0, 9504);
    depthwise_conv_2d_tiny_kernel5x5_stride1(10, 1, 13472);
    depthwise_conv_2d_tiny_kernel5x5_stride1(10, 2, 13472);
    depthwise_conv_2d_tiny_kernel5x5_stride1(10, 3, 13472);
    depthwise_conv_2d_tiny_kernel5x5_stride1(10, 4, 13472);
    depthwise_conv_2d_tiny_kernel5x5_stride1(10, 5, 5552);
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
    conv_2d(22, 0, 3696);
    conv_2d(22, 1, 6336);
    conv_2d(22, 2, 10032);
    conv_2d(22, 3, 12672);
    conv_2d(22, 4, 16640);
    conv_2d(22, 5, 15840);
    depthwise_conv_2d_tiny_kernel5x5_stride1(11, 0, 19008);
    depthwise_conv_2d_tiny_kernel5x5_stride1(11, 1, 23760);
    depthwise_conv_2d_tiny_kernel5x5_stride1(11, 2, 23760);
    depthwise_conv_2d_tiny_kernel5x5_stride1(11, 3, 23760);
    depthwise_conv_2d_tiny_kernel5x5_stride1(11, 4, 23760);
    depthwise_conv_2d_tiny_kernel5x5_stride1(11, 5, 11088);
    conv_2d(23, 0, -1);
    conv_2d(23, 1, -1);
    conv_2d(23, 2, -1);
    conv_2d(23, 3, -1);
    conv_2d(23, 4, -1);
    conv_2d(23, 5, -1);
    conv_2d(24, 0, -1);
    conv_2d(24, 1, -1);
    conv_2d(24, 2, -1);
    conv_2d(24, 3, -1);
    conv_2d(24, 4, -1);
    conv_2d(24, 5, -1);
    depthwise_conv_2d_tiny_kernel5x5_stride1(12, 0, 12672);
    depthwise_conv_2d_tiny_kernel5x5_stride1(12, 1, 18656);
    depthwise_conv_2d_tiny_kernel5x5_stride1(12, 2, 18656);
    depthwise_conv_2d_tiny_kernel5x5_stride1(12, 3, 18656);
    depthwise_conv_2d_tiny_kernel5x5_stride1(12, 4, 18656);
    depthwise_conv_2d_tiny_kernel5x5_stride1(12, 5, 7392);
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
    conv_2d(26, 0, -1);
    conv_2d(26, 1, -1);
    conv_2d(26, 2, -1);
    conv_2d(26, 3, -1);
    conv_2d(26, 4, -1);
    conv_2d(26, 5, -1);
    depthwise_conv_2d_tiny_kernel5x5_stride1(13, 0, 12672);
    depthwise_conv_2d_tiny_kernel5x5_stride1(13, 1, 17952);
    depthwise_conv_2d_tiny_kernel5x5_stride1(13, 2, 17952);
    depthwise_conv_2d_tiny_kernel5x5_stride1(13, 3, 17952);
    depthwise_conv_2d_tiny_kernel5x5_stride1(13, 4, 17952);
    depthwise_conv_2d_tiny_kernel5x5_stride1(13, 5, 7392);
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
    conv_2d(28, 0, -1);
    conv_2d(28, 1, -1);
    conv_2d(28, 2, -1);
    conv_2d(28, 3, -1);
    conv_2d(28, 4, -1);
    conv_2d(28, 5, -1);
    depthwise_conv_2d_tiny_kernel5x5_stride1(14, 0, 12672);
    depthwise_conv_2d_tiny_kernel5x5_stride1(14, 1, 17248);
    depthwise_conv_2d_tiny_kernel5x5_stride1(14, 2, 17248);
    depthwise_conv_2d_tiny_kernel5x5_stride1(14, 3, 17248);
    depthwise_conv_2d_tiny_kernel5x5_stride1(14, 4, 17248);
    depthwise_conv_2d_tiny_kernel5x5_stride1(14, 5, 7392);
    conv_2d(29, 0, -1);
    conv_2d(29, 1, -1);
    conv_2d(29, 2, -1);
    conv_2d(29, 3, -1);
    conv_2d(29, 4, -1);
    conv_2d(29, 5, -1);
    add(9, 0);
    add(9, 1);
    add(9, 2);
    add(9, 3);
    add(9, 4);
    add(9, 5);
    conv_2d(30, 0, -1);
    conv_2d(30, 1, -1);
    conv_2d(30, 2, -1);
    conv_2d(30, 3, -1);
    conv_2d(30, 4, -1);
    conv_2d(30, 5, -1);
    depthwise_conv_2d_tiny_kernel7x7_stride2(15, 0, 23424);
    depthwise_conv_2d_tiny_kernel7x7_stride2(15, 1, 27840);
    depthwise_conv_2d_tiny_kernel7x7_stride2(15, 2, 2304);
    conv_2d(31, 0, -1);
    conv_2d(31, 1, -1);
    conv_2d(31, 2, -1);
    conv_2d(32, 0, -1);
    conv_2d(32, 1, -1);
    conv_2d(32, 2, -1);
    depthwise_conv_2d_tiny_kernel7x7_stride1(16, 0, 18432);
    depthwise_conv_2d_tiny_kernel7x7_stride1(16, 1, 23040);
    depthwise_conv_2d_tiny_kernel7x7_stride1(16, 2, 29952);
    conv_2d(33, 0, -1);
    conv_2d(33, 1, -1);
    conv_2d(33, 2, -1);
    add(10, 0);
    add(10, 1);
    add(10, 2);
    conv_2d(34, 0, -1);
    conv_2d(34, 1, -1);
    conv_2d(34, 2, -1);
    depthwise_conv_2d_tiny_kernel7x7_stride1(17, 0, 9216);
    depthwise_conv_2d_tiny_kernel7x7_stride1(17, 1, 11520);
    depthwise_conv_2d_tiny_kernel7x7_stride1(17, 2, 16128);
    conv_2d(35, 0, -1);
    conv_2d(35, 1, -1);
    conv_2d(35, 2, -1);
    add(11, 0);
    add(11, 1);
    add(11, 2);
    conv_2d(36, 0, -1);
    conv_2d(36, 1, -1);
    conv_2d(36, 2, -1);
    depthwise_conv_2d_tiny_kernel7x7_stride1(18, 0, 9216);
    depthwise_conv_2d_tiny_kernel7x7_stride1(18, 1, 11520);
    depthwise_conv_2d_tiny_kernel7x7_stride1(18, 2, 15360);
    conv_2d(37, 0, -1);
    conv_2d(37, 1, -1);
    conv_2d(37, 2, -1);
    add(12, 0);
    add(12, 1);
    add(12, 2);
    conv_2d(38, 0, -1);
    conv_2d(38, 1, -1);
    conv_2d(38, 2, -1);
    depthwise_conv_2d_tiny_kernel7x7_stride1(19, 0, 18432);
    depthwise_conv_2d_tiny_kernel7x7_stride1(19, 1, 23040);
    depthwise_conv_2d_tiny_kernel7x7_stride1(19, 2, 27648);
    conv_2d(39, 0, -1);
    conv_2d(39, 1, -1);
    conv_2d(39, 2, -1);
    conv_2d(40, 0, -1);
    conv_2d(40, 1, -1);
    conv_2d(40, 2, -1);
    concatenation(0);
    average_pool_2d(0, 14208);
    conv_2d(41, -1);
    reshape(0);
}
