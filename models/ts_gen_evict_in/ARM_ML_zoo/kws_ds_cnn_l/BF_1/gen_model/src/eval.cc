#include "gen_model/include/eval.h"
#include "gen_model/include/ctx.h"
#include "gen_lib/include/ctx_util.h"
#include "gen_lib/OpImpl/AVERAGE_POOL_2D.h"
#include "gen_lib/OpImpl/CONCATENATION.h"
#include "gen_lib/OpImpl/CONV_2D.h"
#include "gen_lib/OpImpl/DEPTHWISE_CONV_2D.h"
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
    conv_2d(0, 0, 2768);
    conv_2d(0, 1, 5536);
    conv_2d(0, 2, 8304);
    conv_2d(0, 3, 11072);
    conv_2d(0, 4, 13840);
    conv_2d(0, 5, 16608);
    conv_2d(0, 6, 19376);
    conv_2d(0, 7, 22144);
    conv_2d(0, 8, 24912);
    conv_2d(0, 9, 27680);
    conv_2d(0, 10, 30448);
    conv_2d(0, 11, 33216);
    conv_2d(0, 12, 35984);
    conv_2d(0, 13, 38752);
    conv_2d(0, 14, 41520);
    conv_2d(0, 15, 44288);
    conv_2d(0, 16, 47056);
    conv_2d(0, 17, 49824);
    conv_2d(0, 18, 52592);
    conv_2d(0, 19, 55360);
    conv_2d(0, 20, 58128);
    conv_2d(0, 21, 60896);
    conv_2d(0, 22, 63664);
    conv_2d(0, 23, 66432);
    conv_2d(0, 24, 69200);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 0, 70592);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 1, 1392);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 2, 2784);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 3, 4176);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 4, 5568);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 5, 6960);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 6, 8352);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 7, 9744);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 8, 11136);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 9, 12528);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 10, 13920);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 11, 15312);
    depthwise_conv_2d_tiny_kernel3x3_stride2(0, 12, 16704);
    conv_2d(1, 0, -1);
    conv_2d(1, 1, -1);
    conv_2d(1, 2, -1);
    conv_2d(1, 3, -1);
    conv_2d(1, 4, -1);
    conv_2d(1, 5, -1);
    conv_2d(1, 6, -1);
    conv_2d(1, 7, -1);
    conv_2d(1, 8, -1);
    conv_2d(1, 9, -1);
    conv_2d(1, 10, -1);
    conv_2d(1, 11, -1);
    conv_2d(1, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 0, 19488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 1, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 2, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 3, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 4, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 5, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 6, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 7, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 8, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 9, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 10, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 11, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(1, 12, 20880);
    conv_2d(2, 0, -1);
    conv_2d(2, 1, -1);
    conv_2d(2, 2, -1);
    conv_2d(2, 3, -1);
    conv_2d(2, 4, -1);
    conv_2d(2, 5, -1);
    conv_2d(2, 6, -1);
    conv_2d(2, 7, -1);
    conv_2d(2, 8, -1);
    conv_2d(2, 9, -1);
    conv_2d(2, 10, -1);
    conv_2d(2, 11, -1);
    conv_2d(2, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 19488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 8, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 9, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 10, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 11, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 12, 20880);
    conv_2d(3, 0, -1);
    conv_2d(3, 1, -1);
    conv_2d(3, 2, -1);
    conv_2d(3, 3, -1);
    conv_2d(3, 4, -1);
    conv_2d(3, 5, -1);
    conv_2d(3, 6, -1);
    conv_2d(3, 7, -1);
    conv_2d(3, 8, -1);
    conv_2d(3, 9, -1);
    conv_2d(3, 10, -1);
    conv_2d(3, 11, -1);
    conv_2d(3, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 0, 19488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 1, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 2, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 3, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 4, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 5, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 6, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 7, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 8, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 9, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 10, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 11, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(3, 12, 20880);
    conv_2d(4, 0, -1);
    conv_2d(4, 1, -1);
    conv_2d(4, 2, -1);
    conv_2d(4, 3, -1);
    conv_2d(4, 4, -1);
    conv_2d(4, 5, -1);
    conv_2d(4, 6, -1);
    conv_2d(4, 7, -1);
    conv_2d(4, 8, -1);
    conv_2d(4, 9, -1);
    conv_2d(4, 10, -1);
    conv_2d(4, 11, -1);
    conv_2d(4, 12, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 0, 19488);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 1, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 2, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 3, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 4, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 5, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 6, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 7, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 8, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 9, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 10, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 11, 20880);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 12, 20880);
    conv_2d(5, 0, -1);
    conv_2d(5, 1, -1);
    conv_2d(5, 2, -1);
    conv_2d(5, 3, -1);
    conv_2d(5, 4, -1);
    conv_2d(5, 5, -1);
    conv_2d(5, 6, -1);
    conv_2d(5, 7, -1);
    conv_2d(5, 8, -1);
    conv_2d(5, 9, -1);
    conv_2d(5, 10, -1);
    conv_2d(5, 11, -1);
    conv_2d(5, 12, -1);
    concatenation(0);
    average_pool_2d(0, 18240);
    reshape(1);
    fully_connected(0);
    softmax(0);
}
