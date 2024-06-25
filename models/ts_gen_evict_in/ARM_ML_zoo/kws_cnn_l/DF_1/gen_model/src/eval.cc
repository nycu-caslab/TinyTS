#include "gen_model/include/eval.h"
#include "gen_model/include/ctx.h"
#include "gen_lib/include/ctx_util.h"
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
    reshape(0);
    split(0);
    conv_2d(0, 0, 432);
    conv_2d(0, 1, 864);
    conv_2d(0, 2, 1296);
    conv_2d(0, 3, 1728);
    conv_2d(0, 4, 2160);
    conv_2d(0, 5, 2592);
    conv_2d(0, 6, 3024);
    conv_2d(0, 7, 3456);
    conv_2d(0, 8, 3888);
    conv_2d(0, 9, 4320);
    conv_2d(1, 0, 5168);
    conv_2d(0, 10, 432);
    conv_2d(0, 11, 4320);
    conv_2d(1, 1, 5472);
    conv_2d(0, 12, 1296);
    conv_2d(0, 13, 4320);
    conv_2d(1, 2, 5776);
    conv_2d(0, 14, 2160);
    conv_2d(0, 15, 4320);
    conv_2d(1, 3, 6080);
    conv_2d(0, 16, 3024);
    conv_2d(0, 17, 4320);
    conv_2d(1, 4, 6384);
    conv_2d(0, 18, 3888);
    conv_2d(0, 19, 4320);
    conv_2d(1, 5, 6688);
    conv_2d(0, 20, 432);
    conv_2d(0, 21, 4320);
    conv_2d(1, 6, 6992);
    conv_2d(0, 22, 1296);
    conv_2d(0, 23, 4320);
    conv_2d(1, 7, 7296);
    conv_2d(0, 24, 2160);
    conv_2d(0, 25, 4320);
    conv_2d(1, 8, 7600);
    conv_2d(0, 26, 3024);
    conv_2d(0, 27, 4320);
    conv_2d(1, 9, 7904);
    conv_2d(0, 28, 3888);
    conv_2d(0, 29, 4320);
    conv_2d(1, 10, 8208);
    conv_2d(0, 30, 432);
    conv_2d(0, 31, 4320);
    conv_2d(1, 11, 8512);
    conv_2d(0, 32, 1296);
    conv_2d(0, 33, 4320);
    conv_2d(1, 12, 8816);
    conv_2d(0, 34, 2160);
    conv_2d(0, 35, 4320);
    conv_2d(1, 13, 9120);
    conv_2d(0, 36, 3024);
    conv_2d(0, 37, 4320);
    conv_2d(1, 14, 9424);
    conv_2d(0, 38, 3888);
    conv_2d(0, 39, 4320);
    conv_2d(1, 15, 9728);
    concatenation(0);
    reshape(1);
    fully_connected(0);
    fully_connected(1);
    fully_connected(2);
    softmax(0);
}
