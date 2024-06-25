#include "gen_model/include/eval.h"
#include "gen_model/include/ctx.h"
#include "gen_lib/include/ctx_util.h"
#include "gen_lib/OpImpl/ADD.h"
#include "gen_lib/OpImpl/AVERAGE_POOL_2D.h"
#include "gen_lib/OpImpl/CONCATENATION.h"
#include "gen_lib/OpImpl/CONV_2D.h"
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
    conv_2d(0, 0, 512);
    conv_2d(0, 1, 1024);
    conv_2d(0, 2, 1536);
    conv_2d(0, 3, 2048);
    conv_2d(0, 4, 2560);
    conv_2d(0, 5, 3072);
    conv_2d(0, 6, 3680);
    conv_2d(0, 7, 4192);
    conv_2d(0, 8, 4704);
    conv_2d(0, 9, 5216);
    conv_2d(0, 10, 5728);
    conv_2d(0, 11, 6240);
    conv_2d(0, 12, 6752);
    conv_2d(0, 13, 7264);
    conv_2d(0, 14, 7776);
    conv_2d(0, 15, 8288);
    conv_2d(0, 16, 8800);
    conv_2d(0, 17, 9312);
    conv_2d(0, 18, 9824);
    conv_2d(0, 19, 10336);
    conv_2d(0, 20, 10848);
    conv_2d(0, 21, 11360);
    conv_2d(0, 22, 11872);
    conv_2d(0, 23, 12384);
    conv_2d(0, 24, 12896);
    conv_2d(0, 25, 13408);
    conv_2d(0, 26, 13920);
    conv_2d(0, 27, 14432);
    conv_2d(0, 28, 14944);
    conv_2d(0, 29, 15456);
    conv_2d(0, 30, 15968);
    conv_2d(0, 31, 16576);
    conv_2d(1, 0, 16896);
    conv_2d(1, 1, 17408);
    conv_2d(1, 2, 17920);
    conv_2d(1, 3, 18432);
    conv_2d(1, 4, 18944);
    conv_2d(1, 5, 19456);
    conv_2d(1, 6, 19968);
    conv_2d(1, 7, 20480);
    conv_2d(1, 8, 20992);
    conv_2d(1, 9, 21504);
    conv_2d(1, 10, 22016);
    conv_2d(1, 11, 22528);
    conv_2d(1, 12, 23040);
    conv_2d(1, 13, 23552);
    conv_2d(1, 14, 24064);
    conv_2d(1, 15, 24576);
    conv_2d(1, 16, 25088);
    conv_2d(1, 17, 25600);
    conv_2d(1, 18, 26112);
    conv_2d(1, 19, 26624);
    conv_2d(1, 20, 27136);
    conv_2d(1, 21, 27648);
    conv_2d(1, 22, 28160);
    conv_2d(1, 23, 28672);
    conv_2d(1, 24, 29184);
    conv_2d(1, 25, 29696);
    conv_2d(1, 26, 30208);
    conv_2d(1, 27, 30720);
    conv_2d(1, 28, 31232);
    conv_2d(1, 29, 31744);
    conv_2d(1, 30, 32256);
    conv_2d(1, 31, 32768);
    conv_2d(2, 0, 33280);
    conv_2d(2, 1, 33792);
    conv_2d(2, 2, 33792);
    conv_2d(2, 3, 33792);
    conv_2d(2, 4, 33792);
    conv_2d(2, 5, 33792);
    conv_2d(2, 6, 33792);
    conv_2d(2, 7, 33792);
    conv_2d(2, 8, 33792);
    conv_2d(2, 9, 33792);
    conv_2d(2, 10, 33792);
    conv_2d(2, 11, 33792);
    conv_2d(2, 12, 33792);
    conv_2d(2, 13, 33792);
    conv_2d(2, 14, 33792);
    conv_2d(2, 15, 33792);
    conv_2d(2, 16, 33792);
    conv_2d(2, 17, 33792);
    conv_2d(2, 18, 33792);
    conv_2d(2, 19, 33792);
    conv_2d(2, 20, 33792);
    conv_2d(2, 21, 33792);
    conv_2d(2, 22, 33792);
    conv_2d(2, 23, 33792);
    conv_2d(2, 24, 33792);
    conv_2d(2, 25, 33792);
    conv_2d(2, 26, 33792);
    conv_2d(2, 27, 33792);
    conv_2d(2, 28, 33792);
    conv_2d(2, 29, 33792);
    conv_2d(2, 30, 33792);
    conv_2d(2, 31, 33792);
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
    add(0, 22);
    add(0, 23);
    add(0, 24);
    add(0, 25);
    add(0, 26);
    add(0, 27);
    add(0, 28);
    add(0, 29);
    add(0, 30);
    add(0, 31);
    conv_2d(3, 0, 16384);
    conv_2d(4, 0, 16896);
    conv_2d(3, 1, 16896);
    conv_2d(4, 1, 17408);
    conv_2d(3, 2, 1024);
    conv_2d(4, 2, 17408);
    conv_2d(3, 3, 2048);
    conv_2d(4, 3, 17408);
    conv_2d(3, 4, 3072);
    conv_2d(4, 4, 17408);
    conv_2d(3, 5, 4096);
    conv_2d(4, 5, 17408);
    conv_2d(3, 6, 5120);
    conv_2d(4, 6, 17408);
    conv_2d(3, 7, 6144);
    conv_2d(4, 7, 17408);
    conv_2d(3, 8, 7168);
    conv_2d(4, 8, 17408);
    conv_2d(3, 9, 8192);
    conv_2d(4, 9, 17408);
    conv_2d(3, 10, 9216);
    conv_2d(4, 10, 17408);
    conv_2d(3, 11, 10240);
    conv_2d(4, 11, 17408);
    conv_2d(3, 12, 11264);
    conv_2d(4, 12, 17408);
    conv_2d(3, 13, 12288);
    conv_2d(4, 13, 17408);
    conv_2d(3, 14, 13312);
    conv_2d(4, 14, 17408);
    conv_2d(3, 15, 14336);
    conv_2d(4, 15, 17408);
    conv_2d(5, 0, 17408);
    conv_2d(5, 1, 17408);
    conv_2d(5, 2, 17408);
    conv_2d(5, 3, 17408);
    conv_2d(5, 4, 17408);
    conv_2d(5, 5, 17408);
    conv_2d(5, 6, 17408);
    conv_2d(5, 7, 17408);
    conv_2d(5, 8, 17408);
    conv_2d(5, 9, 17408);
    conv_2d(5, 10, 17408);
    conv_2d(5, 11, 17408);
    conv_2d(5, 12, 17408);
    conv_2d(5, 13, 17408);
    conv_2d(5, 14, 17408);
    conv_2d(5, 15, 17408);
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
    conv_2d(6, 0, 7680);
    conv_2d(7, 0, 8192);
    conv_2d(6, 1, 8704);
    conv_2d(7, 1, 9216);
    conv_2d(6, 2, 512);
    conv_2d(7, 2, 9216);
    conv_2d(6, 3, 1536);
    conv_2d(7, 3, 9216);
    conv_2d(6, 4, 2560);
    conv_2d(7, 4, 9216);
    conv_2d(6, 5, 3584);
    conv_2d(7, 5, 9216);
    conv_2d(6, 6, 4608);
    conv_2d(7, 6, 9216);
    conv_2d(6, 7, 5632);
    conv_2d(7, 7, 9216);
    conv_2d(8, 0, 9216);
    conv_2d(8, 1, 9216);
    conv_2d(8, 2, 9216);
    conv_2d(8, 3, 9216);
    conv_2d(8, 4, 9216);
    conv_2d(8, 5, 9216);
    conv_2d(8, 6, 9216);
    conv_2d(8, 7, 9216);
    add(2, 0);
    add(2, 1);
    add(2, 2);
    add(2, 3);
    add(2, 4);
    add(2, 5);
    add(2, 6);
    add(2, 7);
    concatenation(0);
    average_pool_2d(0, 4160);
    reshape(0);
    fully_connected(0);
    softmax(0);
}
