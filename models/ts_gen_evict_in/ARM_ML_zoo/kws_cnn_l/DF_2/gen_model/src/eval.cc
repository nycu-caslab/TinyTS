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
    conv_2d(0, 0, 848);
    conv_2d(0, 1, 1696);
    conv_2d(0, 2, 2544);
    conv_2d(0, 3, 3392);
    conv_2d(0, 4, 4240);
    conv_2d(0, 5, 5088);
    conv_2d(1, 0, 5696);
    conv_2d(0, 6, 848);
    conv_2d(0, 7, 5696);
    conv_2d(1, 1, 6304);
    conv_2d(0, 8, 2544);
    conv_2d(0, 9, 6304);
    conv_2d(1, 2, 6912);
    conv_2d(0, 10, 4240);
    conv_2d(0, 11, 6912);
    conv_2d(1, 3, 7520);
    conv_2d(0, 12, 848);
    conv_2d(0, 13, 7520);
    conv_2d(1, 4, 8128);
    conv_2d(0, 14, 2544);
    conv_2d(0, 15, 8128);
    conv_2d(1, 5, 8736);
    conv_2d(0, 16, 4240);
    conv_2d(0, 17, 8736);
    conv_2d(1, 6, 9344);
    conv_2d(0, 18, 848);
    conv_2d(0, 19, 9344);
    conv_2d(1, 7, 9952);
    concatenation(0);
    reshape(1);
    fully_connected(0);
    fully_connected(1);
    fully_connected(2);
    softmax(0);
}