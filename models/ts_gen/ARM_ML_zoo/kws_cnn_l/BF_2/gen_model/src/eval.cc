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
    conv_2d(0, 0, 1024);
    conv_2d(0, 1, 1728);
    conv_2d(0, 2, 2576);
    conv_2d(0, 3, 3424);
    conv_2d(0, 4, 4272);
    conv_2d(0, 5, 5120);
    conv_2d(0, 6, 5968);
    conv_2d(0, 7, 6816);
    conv_2d(0, 8, 7664);
    conv_2d(0, 9, 8512);
    conv_2d(0, 10, 9360);
    conv_2d(0, 11, 10208);
    conv_2d(0, 12, 11056);
    conv_2d(0, 13, 11904);
    conv_2d(0, 14, 12752);
    conv_2d(0, 15, 13600);
    conv_2d(0, 16, 14448);
    conv_2d(0, 17, 15296);
    conv_2d(0, 18, 16144);
    conv_2d(0, 19, 17136);
    conv_2d(1, 0, 17568);
    conv_2d(1, 1, 18176);
    conv_2d(1, 2, 18784);
    conv_2d(1, 3, 19392);
    conv_2d(1, 4, 19392);
    conv_2d(1, 5, 19392);
    conv_2d(1, 6, 19392);
    conv_2d(1, 7, 19392);
    concatenation(0);
    reshape(1);
    fully_connected(0);
    fully_connected(1);
    fully_connected(2);
    softmax(0);
}
