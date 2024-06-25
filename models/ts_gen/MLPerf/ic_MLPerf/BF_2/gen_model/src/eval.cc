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
    conv_2d(0, 0, 1024);
    conv_2d(0, 1, 2048);
    conv_2d(0, 2, 3072);
    conv_2d(0, 3, 4288);
    conv_2d(0, 4, 5312);
    conv_2d(0, 5, 6336);
    conv_2d(0, 6, 7360);
    conv_2d(0, 7, 8384);
    conv_2d(0, 8, 9408);
    conv_2d(0, 9, 10432);
    conv_2d(0, 10, 11456);
    conv_2d(0, 11, 12480);
    conv_2d(0, 12, 13504);
    conv_2d(0, 13, 14528);
    conv_2d(0, 14, 15552);
    conv_2d(0, 15, 16768);
    conv_2d(1, 0, 17408);
    conv_2d(1, 1, 18432);
    conv_2d(1, 2, 19456);
    conv_2d(1, 3, 20480);
    conv_2d(1, 4, 21504);
    conv_2d(1, 5, 22528);
    conv_2d(1, 6, 23552);
    conv_2d(1, 7, 24576);
    conv_2d(1, 8, 25600);
    conv_2d(1, 9, 26624);
    conv_2d(1, 10, 27648);
    conv_2d(1, 11, 28672);
    conv_2d(1, 12, 29696);
    conv_2d(1, 13, 30720);
    conv_2d(1, 14, 31744);
    conv_2d(1, 15, 32768);
    conv_2d(2, 0, 33792);
    conv_2d(2, 1, 34816);
    conv_2d(2, 2, 34816);
    conv_2d(2, 3, 34816);
    conv_2d(2, 4, 34816);
    conv_2d(2, 5, 34816);
    conv_2d(2, 6, 34816);
    conv_2d(2, 7, 34816);
    conv_2d(2, 8, 34816);
    conv_2d(2, 9, 34816);
    conv_2d(2, 10, 34816);
    conv_2d(2, 11, 34816);
    conv_2d(2, 12, 34816);
    conv_2d(2, 13, 34816);
    conv_2d(2, 14, 34816);
    conv_2d(2, 15, 34816);
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
    conv_2d(3, 0, 16384);
    conv_2d(4, 0, 17408);
    conv_2d(3, 1, 17408);
    conv_2d(4, 1, 18432);
    conv_2d(3, 2, 2048);
    conv_2d(4, 2, 18432);
    conv_2d(3, 3, 4096);
    conv_2d(4, 3, 18432);
    conv_2d(3, 4, 6144);
    conv_2d(4, 4, 18432);
    conv_2d(3, 5, 8192);
    conv_2d(4, 5, 18432);
    conv_2d(3, 6, 10240);
    conv_2d(4, 6, 18432);
    conv_2d(4, 7, 12288);
    conv_2d(3, 7, 18432);
    conv_2d(5, 0, 18432);
    conv_2d(5, 1, 18432);
    conv_2d(5, 2, 18432);
    conv_2d(5, 3, 18432);
    conv_2d(5, 4, 18432);
    conv_2d(5, 5, 18432);
    conv_2d(5, 6, 18432);
    conv_2d(5, 7, 18432);
    add(1, 0);
    add(1, 1);
    add(1, 2);
    add(1, 3);
    add(1, 4);
    add(1, 5);
    add(1, 6);
    add(1, 7);
    conv_2d(6, 0, 7168);
    conv_2d(7, 0, 8192);
    conv_2d(6, 1, 9216);
    conv_2d(7, 1, 10240);
    conv_2d(6, 2, 1024);
    conv_2d(7, 2, 10240);
    conv_2d(7, 3, 10240);
    conv_2d(6, 3, 10240);
    conv_2d(8, 0, 10240);
    conv_2d(8, 1, 10240);
    conv_2d(8, 2, 10240);
    conv_2d(8, 3, 10240);
    add(2, 0);
    add(2, 1);
    add(2, 2);
    add(2, 3);
    concatenation(0);
    average_pool_2d(0, 4160);
    reshape(0);
    fully_connected(0);
    softmax(0);
}
