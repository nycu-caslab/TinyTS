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
    model_input_data = input_data;
    split(0);
    conv_2d(0, 0, 1024);
    conv_2d(0, 1, 2048);
    conv_2d(1, 0, 3072);
    conv_2d(0, 2, 4096);
    conv_2d(1, 1, 5120);
    conv_2d(2, 0, 6144);
    add(0, 0);
    conv_2d(0, 3, 5120);
    conv_2d(1, 2, 7168);
    conv_2d(2, 1, 8192);
    add(0, 1);
    conv_2d(3, 0, 7168);
    conv_2d(0, 4, 8192);
    conv_2d(1, 3, 9216);
    conv_2d(2, 2, 10240);
    add(0, 2);
    conv_2d(4, 0, 9216);
    conv_2d(0, 5, 6144);
    conv_2d(1, 4, 9216);
    conv_2d(2, 3, 10240);
    add(0, 3);
    conv_2d(3, 1, 9216);
    conv_2d(0, 6, 10240);
    conv_2d(1, 5, 11264);
    conv_2d(2, 4, 12288);
    add(0, 4);
    conv_2d(4, 1, 11264);
    conv_2d(5, 0, 11264);
    add(1, 0);
    conv_2d(0, 7, 4096);
    conv_2d(1, 6, 11264);
    conv_2d(2, 5, 12288);
    add(0, 5);
    conv_2d(3, 2, 11264);
    conv_2d(0, 8, 12288);
    conv_2d(1, 7, 13312);
    conv_2d(2, 6, 14336);
    add(0, 6);
    conv_2d(4, 2, 13312);
    conv_2d(5, 1, 13312);
    add(1, 1);
    conv_2d(6, 0, 6144);
    conv_2d(0, 9, 8192);
    conv_2d(1, 8, 13312);
    conv_2d(2, 7, 14336);
    add(0, 7);
    conv_2d(3, 3, 13312);
    conv_2d(0, 10, 14336);
    conv_2d(1, 9, 15360);
    conv_2d(2, 8, 16384);
    add(0, 8);
    conv_2d(4, 3, 15360);
    conv_2d(5, 2, 15360);
    add(1, 2);
    conv_2d(7, 0, 15360);
    conv_2d(0, 11, 4096);
    conv_2d(1, 10, 5120);
    conv_2d(2, 9, 10240);
    add(0, 9);
    conv_2d(3, 4, 6144);
    conv_2d(0, 12, 10240);
    conv_2d(1, 11, 15360);
    conv_2d(2, 10, 16384);
    add(0, 10);
    conv_2d(4, 4, 15360);
    conv_2d(5, 3, 15360);
    add(1, 3);
    conv_2d(6, 1, 8192);
    conv_2d(0, 13, 12288);
    conv_2d(1, 12, 15360);
    conv_2d(2, 11, 16384);
    add(0, 11);
    conv_2d(3, 5, 15360);
    conv_2d(0, 14, 16384);
    conv_2d(1, 13, 17408);
    conv_2d(2, 12, 18432);
    add(0, 12);
    conv_2d(4, 5, 17408);
    conv_2d(5, 4, 17408);
    add(1, 4);
    conv_2d(7, 1, 17408);
    conv_2d(8, 0, 17408);
    add(2, 0);
    conv_2d(0, 15, 5120);
    conv_2d(1, 14, 9216);
    conv_2d(2, 13, 14336);
    add(0, 13);
    conv_2d(3, 6, 9216);
    conv_2d(1, 15, 14336);
    conv_2d(2, 14, 17408);
    add(0, 14);
    conv_2d(4, 6, 15360);
    conv_2d(5, 5, 17408);
    add(1, 5);
    conv_2d(6, 2, 10240);
    conv_2d(2, 15, 13312);
    add(0, 15);
    conv_2d(4, 7, 9216);
    conv_2d(5, 6, 17408);
    add(1, 6);
    conv_2d(7, 2, 9216);
    conv_2d(8, 1, 10240);
    add(2, 1);
    conv_2d(5, 7, 11264);
    conv_2d(3, 7, 2048);
    add(1, 7);
    conv_2d(7, 3, 11264);
    conv_2d(8, 2, 11264);
    add(2, 2);
    conv_2d(8, 3, 11264);
    conv_2d(6, 3, 3072);
    add(2, 3);
    concatenation(0);
    average_pool_2d(0, 4160);
    reshape(0);
    fully_connected(0);
    softmax(0);
}
