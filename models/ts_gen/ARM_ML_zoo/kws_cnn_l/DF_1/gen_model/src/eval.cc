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
    conv_2d(0, 0, 448);
    conv_2d(0, 1, 1008);
    conv_2d(0, 2, 1312);
    conv_2d(0, 3, 1744);
    conv_2d(0, 4, 2176);
    conv_2d(0, 5, 2608);
    conv_2d(0, 6, 3040);
    conv_2d(0, 7, 3472);
    conv_2d(0, 8, 3904);
    conv_2d(0, 9, 4864);
    conv_2d(1, 0, 9520);
    conv_2d(0, 10, 432);
    conv_2d(0, 11, 5168);
    conv_2d(1, 1, 9520);
    conv_2d(0, 12, 1296);
    conv_2d(0, 13, 5472);
    conv_2d(1, 2, 9520);
    conv_2d(0, 14, 2160);
    conv_2d(0, 15, 5776);
    conv_2d(1, 3, 9520);
    conv_2d(0, 16, 3024);
    conv_2d(0, 17, 6080);
    conv_2d(1, 4, 9520);
    conv_2d(0, 18, 3888);
    conv_2d(0, 19, 4320);
    conv_2d(1, 5, 9520);
    conv_2d(0, 20, 432);
    conv_2d(0, 21, 4320);
    conv_2d(1, 6, 9520);
    conv_2d(0, 22, 1296);
    conv_2d(0, 23, 4320);
    conv_2d(1, 7, 9520);
    conv_2d(0, 24, 2160);
    conv_2d(0, 25, 4320);
    conv_2d(1, 8, 9520);
    conv_2d(0, 26, 3024);
    conv_2d(0, 27, 4320);
    conv_2d(1, 9, 9520);
    conv_2d(0, 28, 3888);
    conv_2d(0, 29, 4320);
    conv_2d(1, 10, 9520);
    conv_2d(0, 30, 432);
    conv_2d(0, 31, 4320);
    conv_2d(1, 11, 9520);
    conv_2d(0, 32, 1296);
    conv_2d(0, 33, 4320);
    conv_2d(1, 12, 9520);
    conv_2d(0, 34, 2160);
    conv_2d(0, 35, 4320);
    conv_2d(1, 13, 9520);
    conv_2d(0, 36, 3024);
    conv_2d(0, 37, 4320);
    conv_2d(1, 14, 9520);
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
