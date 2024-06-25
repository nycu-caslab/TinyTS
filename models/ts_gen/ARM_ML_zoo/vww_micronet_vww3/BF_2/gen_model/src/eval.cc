#include "gen_model/include/eval.h"
#include "gen_model/include/ctx.h"
#include "gen_lib/include/ctx_util.h"
#include "gen_lib/OpImpl/ADD.h"
#include "gen_lib/OpImpl/AVERAGE_POOL_2D.h"
#include "gen_lib/OpImpl/CONCATENATION.h"
#include "gen_lib/OpImpl/CONV_2D.h"
#include "gen_lib/OpImpl/DEPTHWISE_CONV_2D.h"
#include "gen_lib/OpImpl/PAD.h"
#include "gen_lib/OpImpl/RESHAPE.h"
#include "gen_lib/OpImpl/SPLIT.h"

extern "C" {
#include "arm_nnfunctions.h"
#include "genNN.h"
#include "tinyengine_function.h"
}
void eval(int8_t *input_data){
    split(0);
    conv_2d(0, 0, 1152);
    conv_2d(0, 1, 2304);
    conv_2d(0, 2, 3456);
    conv_2d(0, 3, 4608);
    conv_2d(0, 4, 5760);
    conv_2d(0, 5, 6912);
    conv_2d(0, 6, 8064);
    conv_2d(0, 7, 9216);
    conv_2d(0, 8, 10368);
    conv_2d(0, 9, 11520);
    conv_2d(0, 10, 12672);
    conv_2d(0, 11, 13824);
    conv_2d(0, 12, 14976);
    conv_2d(0, 13, 16128);
    conv_2d(0, 14, 17280);
    conv_2d(0, 15, 18432);
    conv_2d(0, 16, 19584);
    conv_2d(0, 17, 20736);
    conv_2d(0, 18, 21888);
    conv_2d(0, 19, 23040);
    conv_2d(0, 20, 24192);
    conv_2d(0, 21, 25344);
    conv_2d(0, 22, 26496);
    conv_2d(0, 23, 27648);
    conv_2d(0, 24, 29312);
    conv_2d(0, 25, 30464);
    conv_2d(0, 26, 31616);
    conv_2d(0, 27, 32768);
    conv_2d(0, 28, 33920);
    conv_2d(0, 29, 35072);
    conv_2d(0, 30, 36224);
    conv_2d(0, 31, 37376);
    conv_2d(1, 0, 37248);
    conv_2d(1, 1, 384);
    conv_2d(1, 2, 768);
    conv_2d(1, 3, 1152);
    conv_2d(1, 4, 1536);
    conv_2d(1, 5, 1920);
    conv_2d(1, 6, 2304);
    conv_2d(1, 7, 2688);
    conv_2d(1, 8, 3072);
    conv_2d(1, 9, 3456);
    conv_2d(1, 10, 3840);
    conv_2d(1, 11, 4224);
    conv_2d(1, 12, 4608);
    conv_2d(1, 13, 4992);
    conv_2d(1, 14, 5376);
    conv_2d(1, 15, 5760);
    conv_2d(1, 16, 6144);
    conv_2d(1, 17, 6528);
    conv_2d(1, 18, 6912);
    conv_2d(1, 19, 7296);
    conv_2d(1, 20, 7680);
    conv_2d(1, 21, 8064);
    conv_2d(1, 22, 8448);
    conv_2d(1, 23, 8832);
    conv_2d(1, 24, 9216);
    conv_2d(1, 25, 9600);
    conv_2d(1, 26, 9984);
    conv_2d(1, 27, 10368);
    conv_2d(1, 28, 10752);
    conv_2d(1, 29, 11136);
    conv_2d(1, 30, 11520);
    conv_2d(1, 31, 11904);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 0, 12288);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 1, 12672);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 2, 13056);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 3, 13056);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 4, 13056);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 5, 13056);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 6, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 7, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 8, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 9, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 10, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 11, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 12, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 13, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 14, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 15, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 16, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 17, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 18, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 19, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 20, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 21, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 22, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 23, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 24, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 25, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 26, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 27, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 28, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 29, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 30, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride1(0, 31, 1152);
    conv_2d(2, 0, 2304);
    conv_2d(2, 1, 3456);
    conv_2d(2, 2, 4608);
    conv_2d(2, 3, 5760);
    conv_2d(2, 4, 0);
    conv_2d(2, 5, 0);
    conv_2d(2, 6, 0);
    conv_2d(2, 7, 0);
    conv_2d(2, 8, 0);
    conv_2d(2, 9, 0);
    conv_2d(2, 10, 0);
    conv_2d(2, 11, 0);
    conv_2d(2, 12, 0);
    conv_2d(2, 13, 0);
    conv_2d(2, 14, 0);
    conv_2d(2, 15, 0);
    conv_2d(2, 16, 0);
    conv_2d(2, 17, 0);
    conv_2d(2, 18, 0);
    conv_2d(2, 19, 0);
    conv_2d(2, 20, 0);
    conv_2d(2, 21, 0);
    conv_2d(2, 22, 0);
    conv_2d(2, 23, 0);
    conv_2d(2, 24, 0);
    conv_2d(2, 25, 0);
    conv_2d(2, 26, 0);
    conv_2d(2, 27, 0);
    conv_2d(2, 28, 0);
    conv_2d(2, 29, 0);
    conv_2d(2, 30, 0);
    conv_2d(2, 31, 0);
    conv_2d(3, 0, 1920);
    conv_2d(3, 1, 3072);
    conv_2d(3, 2, 4224);
    conv_2d(3, 3, 5376);
    conv_2d(3, 4, 6528);
    conv_2d(3, 5, 7680);
    conv_2d(3, 6, 8832);
    conv_2d(3, 7, 9984);
    conv_2d(3, 8, 11136);
    conv_2d(3, 9, 12288);
    conv_2d(3, 10, 13440);
    conv_2d(3, 11, 14592);
    conv_2d(3, 12, 15744);
    conv_2d(3, 13, 16896);
    conv_2d(3, 14, 18048);
    conv_2d(3, 15, 19200);
    conv_2d(3, 16, 20352);
    conv_2d(3, 17, 21504);
    conv_2d(3, 18, 22656);
    conv_2d(3, 19, 23808);
    conv_2d(3, 20, 24960);
    conv_2d(3, 21, 26112);
    conv_2d(3, 22, 27264);
    conv_2d(3, 23, 28416);
    conv_2d(3, 24, 29568);
    conv_2d(3, 25, 30720);
    conv_2d(3, 26, 31872);
    conv_2d(3, 27, 33024);
    conv_2d(3, 28, 34176);
    conv_2d(3, 29, 35328);
    conv_2d(3, 30, 36480);
    conv_2d(3, 31, 37632);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 0, 37440);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 1, 576);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 2, 1152);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 3, 1728);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 4, 1728);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 5, 1728);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 6, 1728);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 7, 1728);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 8, 1728);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 9, 1728);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 10, 1728);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 11, 1728);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 12, 1728);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 13, 1728);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 14, 1728);
    depthwise_conv_2d_tiny_kernel3x3_stride2(1, 15, 1728);
    conv_2d(4, 0, 1728);
    conv_2d(4, 1, 1728);
    conv_2d(4, 2, 0);
    conv_2d(4, 3, 0);
    conv_2d(4, 4, 0);
    conv_2d(4, 5, 0);
    conv_2d(4, 6, 0);
    conv_2d(4, 7, 0);
    conv_2d(4, 8, 0);
    conv_2d(4, 9, 0);
    conv_2d(4, 10, 0);
    conv_2d(4, 11, 0);
    conv_2d(4, 12, 0);
    conv_2d(4, 13, 0);
    conv_2d(4, 14, 0);
    conv_2d(4, 15, 0);
    pad(0, 0);
    conv_2d(5, 0, 2688);
    pad(0, 1);
    conv_2d(5, 1, 4480);
    pad(0, 2);
    conv_2d(5, 2, 6272);
    pad(0, 3);
    conv_2d(5, 3, 8064);
    pad(0, 4);
    conv_2d(5, 4, 9856);
    pad(0, 5);
    conv_2d(5, 5, 11648);
    pad(0, 6);
    conv_2d(5, 6, 13440);
    pad(0, 7);
    conv_2d(5, 7, 15232);
    pad(0, 8);
    conv_2d(5, 8, 17024);
    pad(0, 9);
    conv_2d(5, 9, 18816);
    pad(0, 10);
    conv_2d(5, 10, 20608);
    pad(0, 11);
    conv_2d(5, 11, 22400);
    pad(0, 12);
    conv_2d(5, 12, 24192);
    pad(0, 13);
    conv_2d(5, 13, 25984);
    pad(0, 14);
    conv_2d(5, 14, 27776);
    pad(0, 15);
    conv_2d(5, 15, 29568);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 0, 30464);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 1, 48640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 2, 48640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 3, 48640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 4, 48640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 5, 48640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 6, 48640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 7, 48640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 8, 48640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 9, 48640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 10, 48640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 11, 48640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 12, 48640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 13, 48640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 14, 48640);
    depthwise_conv_2d_tiny_kernel3x3_stride1(2, 15, 48640);
    conv_2d(6, 0, -1);
    conv_2d(6, 1, -1);
    conv_2d(6, 2, -1);
    conv_2d(6, 3, -1);
    conv_2d(6, 4, -1);
    conv_2d(6, 5, -1);
    conv_2d(6, 6, -1);
    conv_2d(6, 7, -1);
    conv_2d(6, 8, -1);
    conv_2d(6, 9, -1);
    conv_2d(6, 10, -1);
    conv_2d(6, 11, -1);
    conv_2d(6, 12, -1);
    conv_2d(6, 13, -1);
    conv_2d(6, 14, -1);
    conv_2d(6, 15, -1);
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
    conv_2d(7, 0, 1792);
    conv_2d(7, 1, 3584);
    conv_2d(7, 2, 5376);
    conv_2d(7, 3, 7168);
    conv_2d(7, 4, 8960);
    conv_2d(7, 5, 10752);
    conv_2d(7, 6, 12544);
    conv_2d(7, 7, 14336);
    conv_2d(7, 8, 16128);
    conv_2d(7, 9, 17920);
    conv_2d(7, 10, 19712);
    conv_2d(7, 11, 21504);
    conv_2d(7, 12, 23296);
    conv_2d(7, 13, 25088);
    conv_2d(7, 14, 26880);
    conv_2d(7, 15, 29696);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 0, 29568);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 1, 896);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 2, 1792);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 3, 2688);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 4, 3584);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 5, 4480);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 6, 5376);
    depthwise_conv_2d_tiny_kernel3x3_stride2(3, 7, 6272);
    conv_2d(8, 0, -1);
    conv_2d(8, 1, -1);
    conv_2d(8, 2, -1);
    conv_2d(8, 3, -1);
    conv_2d(8, 4, -1);
    conv_2d(8, 5, -1);
    conv_2d(8, 6, -1);
    conv_2d(8, 7, -1);
    conv_2d(9, 0, -1);
    conv_2d(9, 1, -1);
    conv_2d(9, 2, -1);
    conv_2d(9, 3, -1);
    conv_2d(9, 4, -1);
    conv_2d(9, 5, -1);
    conv_2d(9, 6, -1);
    conv_2d(9, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 0, 33120);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 1, 44992);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 2, 44992);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 3, 44992);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 4, 44992);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 5, 44992);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 6, 44992);
    depthwise_conv_2d_tiny_kernel3x3_stride1(4, 7, 44992);
    conv_2d(10, 0, 22880);
    conv_2d(10, 1, 23680);
    conv_2d(10, 2, 24480);
    conv_2d(10, 3, 0);
    conv_2d(10, 4, 0);
    conv_2d(10, 5, 0);
    conv_2d(10, 6, 0);
    conv_2d(10, 7, 0);
    pad(1, 0);
    pad(1, 1);
    pad(1, 2);
    pad(1, 3);
    pad(1, 4);
    pad(1, 5);
    pad(1, 6);
    pad(1, 7);
    add(1, 0);
    add(1, 1);
    add(1, 2);
    add(1, 3);
    add(1, 4);
    add(1, 5);
    add(1, 6);
    add(1, 7);
    conv_2d(11, 0, -1);
    conv_2d(11, 1, -1);
    conv_2d(11, 2, -1);
    conv_2d(11, 3, -1);
    conv_2d(11, 4, -1);
    conv_2d(11, 5, -1);
    conv_2d(11, 6, -1);
    conv_2d(11, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 0, 33120);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 1, 43968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 2, 43968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 3, 43968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 4, 43968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 5, 43968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 6, 43968);
    depthwise_conv_2d_tiny_kernel3x3_stride1(5, 7, 43968);
    conv_2d(12, 0, 23104);
    conv_2d(12, 1, 24128);
    conv_2d(12, 2, 25152);
    conv_2d(12, 3, 1024);
    conv_2d(12, 4, 2048);
    conv_2d(12, 5, 3072);
    conv_2d(12, 6, 4096);
    conv_2d(12, 7, 5120);
    add(2, 0);
    add(2, 1);
    add(2, 2);
    add(2, 3);
    add(2, 4);
    add(2, 5);
    add(2, 6);
    add(2, 7);
    conv_2d(13, 0, -1);
    conv_2d(13, 1, -1);
    conv_2d(13, 2, -1);
    conv_2d(13, 3, -1);
    conv_2d(13, 4, -1);
    conv_2d(13, 5, -1);
    conv_2d(13, 6, -1);
    conv_2d(13, 7, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 0, 46784);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 1, 2752);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 2, 5504);
    depthwise_conv_2d_tiny_kernel3x3_stride2(6, 3, 8256);
    conv_2d(14, 0, -1);
    conv_2d(14, 1, -1);
    conv_2d(14, 2, -1);
    conv_2d(14, 3, -1);
    conv_2d(15, 0, 4912);
    conv_2d(15, 1, 9824);
    conv_2d(15, 2, 14736);
    conv_2d(15, 3, 19648);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 0, 24560);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 1, 32736);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 2, 32736);
    depthwise_conv_2d_tiny_kernel3x3_stride1(7, 3, 32736);
    conv_2d(16, 0, 10640);
    conv_2d(16, 1, 11456);
    conv_2d(16, 2, 12272);
    conv_2d(16, 3, 816);
    add(3, 0);
    add(3, 1);
    add(3, 2);
    add(3, 3);
    conv_2d(17, 0, 4912);
    conv_2d(17, 1, 9824);
    conv_2d(17, 2, 14736);
    conv_2d(17, 3, 19648);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 0, 24560);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 1, 31920);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 2, 31920);
    depthwise_conv_2d_tiny_kernel3x3_stride1(8, 3, 31920);
    conv_2d(18, 0, 10528);
    conv_2d(18, 1, 11232);
    conv_2d(18, 2, 11936);
    conv_2d(18, 3, 0);
    pad(2, 0);
    pad(2, 1);
    pad(2, 2);
    pad(2, 3);
    add(4, 0);
    add(4, 1);
    add(4, 2);
    add(4, 3);
    conv_2d(19, 0, 5728);
    pad(3, 0);
    conv_2d(19, 1, 10640);
    pad(3, 1);
    conv_2d(19, 2, 15552);
    pad(3, 2);
    conv_2d(19, 3, 20464);
    pad(3, 3);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 0, 24560);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 1, 33120);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 2, 33120);
    depthwise_conv_2d_tiny_kernel3x3_stride1(9, 3, 33120);
    conv_2d(20, 0, 10736);
    conv_2d(20, 1, 11648);
    conv_2d(20, 2, 12560);
    conv_2d(20, 3, 912);
    add(5, 0);
    add(5, 1);
    add(5, 2);
    add(5, 3);
    conv_2d(21, 0, 4592);
    conv_2d(21, 1, 8272);
    conv_2d(21, 2, 11040);
    conv_2d(21, 3, 15632);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 0, 18400);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 1, 22080);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 2, 22080);
    depthwise_conv_2d_tiny_kernel3x3_stride1(10, 3, 22080);
    conv_2d(22, 0, 7360);
    conv_2d(22, 1, 7360);
    conv_2d(22, 2, 7360);
    conv_2d(22, 3, 0);
    conv_2d(23, 0, 4608);
    conv_2d(23, 1, 9216);
    conv_2d(23, 2, 13824);
    conv_2d(23, 3, 18432);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 0, 23040);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 1, 30720);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 2, 30720);
    depthwise_conv_2d_tiny_kernel3x3_stride1(11, 3, 30720);
    conv_2d(24, 0, -1);
    conv_2d(24, 1, -1);
    conv_2d(24, 2, -1);
    conv_2d(24, 3, -1);
    add(6, 0);
    add(6, 1);
    add(6, 2);
    add(6, 3);
    conv_2d(25, 0, 5376);
    pad(4, 0);
    conv_2d(25, 1, 9984);
    pad(4, 1);
    conv_2d(25, 2, 14592);
    pad(4, 2);
    conv_2d(25, 3, 19200);
    pad(4, 3);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 0, 23040);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 1, 32512);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 2, 32512);
    depthwise_conv_2d_tiny_kernel3x3_stride1(12, 3, 32512);
    conv_2d(26, 0, -1);
    conv_2d(26, 1, -1);
    conv_2d(26, 2, -1);
    conv_2d(26, 3, -1);
    add(7, 0);
    add(7, 1);
    add(7, 2);
    add(7, 3);
    conv_2d(27, 0, -1);
    conv_2d(27, 1, -1);
    conv_2d(27, 2, -1);
    conv_2d(27, 3, -1);
    depthwise_conv_2d_tiny_kernel3x3_stride2(13, 0, 8288);
    depthwise_conv_2d_tiny_kernel3x3_stride2(13, 1, 928);
    conv_2d(28, 0, 928);
    conv_2d(28, 1, 928);
    conv_2d(29, 0, 768);
    conv_2d(29, 1, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(14, 0, 2304);
    depthwise_conv_2d_tiny_kernel3x3_stride1(14, 1, 3328);
    conv_2d(30, 0, -1);
    conv_2d(30, 1, -1);
    add(8, 0);
    add(8, 1);
    conv_2d(31, 0, 768);
    conv_2d(31, 1, 1536);
    depthwise_conv_2d_tiny_kernel3x3_stride1(15, 0, 2304);
    depthwise_conv_2d_tiny_kernel3x3_stride1(15, 1, 3200);
    conv_2d(32, 0, -1);
    conv_2d(32, 1, -1);
    add(9, 0);
    add(9, 1);
    conv_2d(33, 0, 896);
    conv_2d(33, 1, 1664);
    depthwise_conv_2d_tiny_kernel3x3_stride1(16, 0, 2304);
    depthwise_conv_2d_tiny_kernel3x3_stride1(16, 1, 3072);
    conv_2d(34, 0, -1);
    conv_2d(34, 1, -1);
    conv_2d(35, 0, -1);
    conv_2d(35, 1, -1);
    concatenation(0);
    average_pool_2d(0, 2176);
    conv_2d(36, -1);
    reshape(0);
}
