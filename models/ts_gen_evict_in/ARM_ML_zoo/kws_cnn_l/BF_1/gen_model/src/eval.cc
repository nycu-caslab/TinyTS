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
    conv_2d(0, 10, 4752);
    conv_2d(0, 11, 5184);
    conv_2d(0, 12, 5616);
    conv_2d(0, 13, 6048);
    conv_2d(0, 14, 6480);
    conv_2d(0, 15, 6912);
    conv_2d(0, 16, 7344);
    conv_2d(0, 17, 7776);
    conv_2d(0, 18, 8208);
    conv_2d(0, 19, 8640);
    conv_2d(0, 20, 9072);
    conv_2d(0, 21, 9504);
    conv_2d(0, 22, 9936);
    conv_2d(0, 23, 10368);
    conv_2d(0, 24, 10800);
    conv_2d(0, 25, 11232);
    conv_2d(0, 26, 11664);
    conv_2d(0, 27, 12096);
    conv_2d(0, 28, 12528);
    conv_2d(0, 29, 12960);
    conv_2d(0, 30, 13392);
    conv_2d(0, 31, 13824);
    conv_2d(0, 32, 14256);
    conv_2d(0, 33, 14688);
    conv_2d(0, 34, 15120);
    conv_2d(0, 35, 15552);
    conv_2d(0, 36, 15984);
    conv_2d(0, 37, 16416);
    conv_2d(0, 38, 16848);
    conv_2d(0, 39, 17280);
    conv_2d(1, 0, 17584);
    conv_2d(1, 1, 17888);
    conv_2d(1, 2, 18192);
    conv_2d(1, 3, 18496);
    conv_2d(1, 4, 18800);
    conv_2d(1, 5, 19104);
    conv_2d(1, 6, 19104);
    conv_2d(1, 7, 19104);
    conv_2d(1, 8, 19104);
    conv_2d(1, 9, 19104);
    conv_2d(1, 10, 19104);
    conv_2d(1, 11, 19104);
    conv_2d(1, 12, 19104);
    conv_2d(1, 13, 19104);
    conv_2d(1, 14, 19104);
    conv_2d(1, 15, 19104);
    concatenation(0);
    reshape(1);
    fully_connected(0);
    fully_connected(1);
    fully_connected(2);
    softmax(0);
}
