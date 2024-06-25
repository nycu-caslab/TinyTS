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
    conv_2d(0, 9, 4336);
    conv_2d(0, 10, 4768);
    conv_2d(0, 11, 5200);
    conv_2d(0, 12, 5632);
    conv_2d(0, 13, 6064);
    conv_2d(0, 14, 6496);
    conv_2d(0, 15, 6928);
    conv_2d(0, 16, 7360);
    conv_2d(0, 17, 7792);
    conv_2d(0, 18, 8224);
    conv_2d(0, 19, 8656);
    conv_2d(0, 20, 9088);
    conv_2d(0, 21, 9520);
    conv_2d(0, 22, 9952);
    conv_2d(0, 23, 10384);
    conv_2d(0, 24, 10816);
    conv_2d(0, 25, 11248);
    conv_2d(0, 26, 11680);
    conv_2d(0, 27, 12112);
    conv_2d(0, 28, 12544);
    conv_2d(0, 29, 12976);
    conv_2d(0, 30, 13408);
    conv_2d(0, 31, 13840);
    conv_2d(0, 32, 14272);
    conv_2d(0, 33, 14704);
    conv_2d(0, 34, 15136);
    conv_2d(0, 35, 15568);
    conv_2d(0, 36, 16000);
    conv_2d(0, 37, 16432);
    conv_2d(0, 38, 16864);
    conv_2d(0, 39, 17440);
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
