#include "gen_model/include/ctx.h"
#include <cstdio>

alignas(16) int8_t arena[146016];
const int32_t arena_size = 146016;
const int32_t input_tid = 0;
const int32_t output_tid = 230;
const Tensor tensors[231] = {
    {{1, 144, 144, 3}, -1, 0, 0}, 
    {{16, 3, 3, 3}, 0, -1, 3}, 
    {{8, 1, 1, 16}, 432, -1, 3}, 
    {{24, 1, 1, 8}, 560, -1, 3}, 
    {{16, 1, 1, 24}, 752, -1, 3}, 
    {{80, 1, 1, 16}, 1136, -1, 3}, 
    {{16, 1, 1, 80}, 2416, -1, 3}, 
    {{96, 1, 1, 16}, 3696, -1, 3}, 
    {{16, 1, 1, 96}, 5232, -1, 3}, 
    {{48, 1, 1, 16}, 6768, -1, 3}, 
    {{16, 1, 1, 48}, 7536, -1, 3}, 
    {{80, 1, 1, 16}, 8304, -1, 3}, 
    {{24, 1, 1, 80}, 9584, -1, 3}, 
    {{96, 1, 1, 24}, 11504, -1, 3}, 
    {{24, 1, 1, 96}, 13808, -1, 3}, 
    {{96, 1, 1, 24}, 16112, -1, 3}, 
    {{24, 1, 1, 96}, 18416, -1, 3}, 
    {{144, 1, 1, 24}, 20720, -1, 3}, 
    {{24, 1, 1, 144}, 24176, -1, 3}, 
    {{144, 1, 1, 24}, 27632, -1, 3}, 
    {{40, 1, 1, 144}, 31088, -1, 3}, 
    {{240, 1, 1, 40}, 36848, -1, 3}, 
    {{40, 1, 1, 240}, 46448, -1, 3}, 
    {{160, 1, 1, 40}, 56048, -1, 3}, 
    {{40, 1, 1, 160}, 62448, -1, 3}, 
    {{200, 1, 1, 40}, 68848, -1, 3}, 
    {{40, 1, 1, 200}, 76848, -1, 3}, 
    {{200, 1, 1, 40}, 84848, -1, 3}, 
    {{48, 1, 1, 200}, 92848, -1, 3}, 
    {{144, 1, 1, 48}, 102448, -1, 3}, 
    {{48, 1, 1, 144}, 109360, -1, 3}, 
    {{192, 1, 1, 48}, 116272, -1, 3}, 
    {{48, 1, 1, 192}, 125488, -1, 3}, 
    {{144, 1, 1, 48}, 134704, -1, 3}, 
    {{48, 1, 1, 144}, 141616, -1, 3}, 
    {{192, 1, 1, 48}, 148528, -1, 3}, 
    {{96, 1, 1, 192}, 157744, -1, 3}, 
    {{480, 1, 1, 96}, 176176, -1, 3}, 
    {{96, 1, 1, 480}, 222256, -1, 3}, 
    {{384, 1, 1, 96}, 268336, -1, 3}, 
    {{96, 1, 1, 384}, 305200, -1, 3}, 
    {{384, 1, 1, 96}, 342064, -1, 3}, 
    {{96, 1, 1, 384}, 378928, -1, 3}, 
    {{480, 1, 1, 96}, 415792, -1, 3}, 
    {{160, 1, 1, 480}, 461872, -1, 3}, 
    {{2, 1, 1, 160}, 538672, -1, 3}, 
    {{1, 16, 1, 1}, 538992, -1, -1}, 
    {{1, 8, 1, 1}, 539056, -1, -1}, 
    {{1, 24, 1, 1}, 539088, -1, -1}, 
    {{1, 16, 1, 1}, 539184, -1, -1}, 
    {{1, 80, 1, 1}, 539248, -1, -1}, 
    {{1, 16, 1, 1}, 539568, -1, -1}, 
    {{1, 96, 1, 1}, 539632, -1, -1}, 
    {{1, 16, 1, 1}, 540016, -1, -1}, 
    {{1, 48, 1, 1}, 540080, -1, -1}, 
    {{1, 16, 1, 1}, 540272, -1, -1}, 
    {{1, 80, 1, 1}, 540336, -1, -1}, 
    {{1, 24, 1, 1}, 540656, -1, -1}, 
    {{1, 96, 1, 1}, 540752, -1, -1}, 
    {{1, 24, 1, 1}, 541136, -1, -1}, 
    {{1, 96, 1, 1}, 541232, -1, -1}, 
    {{1, 24, 1, 1}, 541616, -1, -1}, 
    {{1, 144, 1, 1}, 541712, -1, -1}, 
    {{1, 24, 1, 1}, 542288, -1, -1}, 
    {{1, 144, 1, 1}, 542384, -1, -1}, 
    {{1, 40, 1, 1}, 542960, -1, -1}, 
    {{1, 240, 1, 1}, 543120, -1, -1}, 
    {{1, 40, 1, 1}, 544080, -1, -1}, 
    {{1, 160, 1, 1}, 544240, -1, -1}, 
    {{1, 40, 1, 1}, 544880, -1, -1}, 
    {{1, 200, 1, 1}, 545040, -1, -1}, 
    {{1, 40, 1, 1}, 545840, -1, -1}, 
    {{1, 200, 1, 1}, 546000, -1, -1}, 
    {{1, 48, 1, 1}, 546800, -1, -1}, 
    {{1, 144, 1, 1}, 546992, -1, -1}, 
    {{1, 48, 1, 1}, 547568, -1, -1}, 
    {{1, 192, 1, 1}, 547760, -1, -1}, 
    {{1, 48, 1, 1}, 548528, -1, -1}, 
    {{1, 144, 1, 1}, 548720, -1, -1}, 
    {{1, 48, 1, 1}, 549296, -1, -1}, 
    {{1, 192, 1, 1}, 549488, -1, -1}, 
    {{1, 96, 1, 1}, 550256, -1, -1}, 
    {{1, 480, 1, 1}, 550640, -1, -1}, 
    {{1, 96, 1, 1}, 552560, -1, -1}, 
    {{1, 384, 1, 1}, 552944, -1, -1}, 
    {{1, 96, 1, 1}, 554480, -1, -1}, 
    {{1, 384, 1, 1}, 554864, -1, -1}, 
    {{1, 96, 1, 1}, 556400, -1, -1}, 
    {{1, 480, 1, 1}, 556784, -1, -1}, 
    {{1, 160, 1, 1}, 558704, -1, -1}, 
    {{1, 2, 1, 1}, 559344, -1, -1}, 
    {{1, 2, 1, 1}, 559360, -1, 5}, 
    {{1, 2, 1, 1}, 559376, -1, 5}, 
    {{1, 2, 1, 1}, 559392, -1, 5}, 
    {{1, 1, 1, 1}, 559408, -1, 5}, 
    {{1, 144, 144, 3}, -72, 0, 7}, 
    {{1, 72, 72, 16}, -36, 72, 8}, 
    {{1, 3, 3, 16}, 559424, -1, 3}, 
    {{1, 16, 1, 1}, 559568, -1, -1}, 
    {{1, 72, 72, 16}, -36, 108, 9}, 
    {{1, 72, 72, 8}, -36, 144, 10}, 
    {{1, 72, 72, 24}, -36, 180, 11}, 
    {{1, 3, 3, 24}, 559632, -1, 3}, 
    {{1, 24, 1, 1}, 559856, -1, -1}, 
    {{1, 36, 36, 24}, -18, 216, 12}, 
    {{1, 36, 36, 16}, -18, 234, 13}, 
    {{1, 36, 36, 80}, -18, 252, 14}, 
    {{1, 3, 3, 80}, 559952, -1, 3}, 
    {{1, 80, 1, 1}, 560672, -1, -1}, 
    {{1, 36, 36, 80}, -18, 270, 15}, 
    {{1, 36, 36, 16}, -18, 288, 16}, 
    {{1, 36, 36, 16}, -18, 306, 17}, 
    {{1, 36, 36, 96}, -18, 324, 18}, 
    {{1, 5, 5, 96}, 560992, -1, 3}, 
    {{1, 96, 1, 1}, 563392, -1, -1}, 
    {{1, 2, 1, 1}, 563776, -1, 5}, 
    {{1, 36, 36, 96}, -18, 342, 19}, 
    {{1, 36, 36, 16}, -18, 360, 20}, 
    {{1, 36, 36, 16}, -18, 378, 21}, 
    {{1, 36, 36, 48}, -18, 396, 22}, 
    {{1, 2, 1, 1}, 563792, -1, 5}, 
    {{1, 3, 3, 48}, 563808, -1, 3}, 
    {{1, 48, 1, 1}, 564240, -1, -1}, 
    {{1, 36, 36, 48}, -18, 414, 23}, 
    {{1, 36, 36, 16}, -18, 432, 24}, 
    {{1, 36, 36, 16}, -18, 450, 25}, 
    {{1, 36, 36, 80}, -18, 468, 26}, 
    {{1, 7, 7, 80}, 564432, -1, 3}, 
    {{1, 80, 1, 1}, 568352, -1, -1}, 
    {{1, 2, 1, 1}, 568672, -1, 5}, 
    {{1, 18, 18, 80}, -9, 486, 27}, 
    {{1, 18, 18, 24}, -9, 495, 28}, 
    {{1, 18, 18, 96}, -9, 504, 29}, 
    {{1, 2, 1, 1}, 568688, -1, 5}, 
    {{1, 5, 5, 96}, 568704, -1, 3}, 
    {{1, 96, 1, 1}, 571104, -1, -1}, 
    {{1, 18, 18, 96}, -9, 513, 30}, 
    {{1, 18, 18, 24}, -9, 522, 3}, 
    {{1, 18, 18, 24}, -9, 531, 31}, 
    {{1, 18, 18, 96}, -9, 540, 32}, 
    {{1, 3, 3, 96}, 571488, -1, 3}, 
    {{1, 96, 1, 1}, 572352, -1, -1}, 
    {{1, 18, 18, 96}, -9, 549, 33}, 
    {{1, 18, 18, 24}, -9, 558, 34}, 
    {{1, 18, 18, 24}, -9, 567, 35}, 
    {{1, 18, 18, 144}, -9, 576, 36}, 
    {{1, 7, 7, 144}, 572736, -1, 3}, 
    {{1, 144, 1, 1}, 579792, -1, -1}, 
    {{1, 18, 18, 144}, -9, 585, 37}, 
    {{1, 18, 18, 24}, -9, 594, 38}, 
    {{1, 18, 18, 24}, -9, 603, 39}, 
    {{1, 18, 18, 144}, -9, 612, 40}, 
    {{1, 2, 1, 1}, 580368, -1, 5}, 
    {{1, 3, 3, 144}, 580384, -1, 3}, 
    {{1, 144, 1, 1}, 581680, -1, -1}, 
    {{1, 9, 9, 144}, -5, 621, 41}, 
    {{1, 9, 9, 40}, -5, 626, 42}, 
    {{1, 9, 9, 240}, -5, 631, 43}, 
    {{1, 7, 7, 240}, 582256, -1, 3}, 
    {{1, 240, 1, 1}, 594016, -1, -1}, 
    {{1, 9, 9, 240}, -5, 636, 44}, 
    {{1, 9, 9, 40}, -5, 641, 45}, 
    {{1, 9, 9, 40}, -5, 646, 46}, 
    {{1, 9, 9, 160}, -5, 651, 47}, 
    {{1, 5, 5, 160}, 594976, -1, 3}, 
    {{1, 160, 1, 1}, 598976, -1, -1}, 
    {{1, 9, 9, 160}, -5, 656, 48}, 
    {{1, 9, 9, 40}, -5, 661, 49}, 
    {{1, 9, 9, 40}, -5, 666, 50}, 
    {{1, 9, 9, 200}, -5, 671, 51}, 
    {{1, 3, 3, 200}, 599616, -1, 3}, 
    {{1, 200, 1, 1}, 601424, -1, -1}, 
    {{1, 9, 9, 200}, -5, 676, 52}, 
    {{1, 9, 9, 40}, -5, 681, 53}, 
    {{1, 9, 9, 40}, -5, 686, 54}, 
    {{1, 9, 9, 200}, -5, 691, 55}, 
    {{1, 5, 5, 200}, 602224, -1, 3}, 
    {{1, 200, 1, 1}, 607232, -1, -1}, 
    {{1, 9, 9, 200}, -5, 696, 56}, 
    {{1, 9, 9, 48}, -5, 701, 57}, 
    {{1, 9, 9, 144}, -5, 706, 58}, 
    {{1, 5, 5, 144}, 608032, -1, 3}, 
    {{1, 144, 1, 1}, 611632, -1, -1}, 
    {{1, 9, 9, 144}, -5, 711, 59}, 
    {{1, 9, 9, 48}, -5, 716, 60}, 
    {{1, 9, 9, 48}, -5, 721, 61}, 
    {{1, 9, 9, 192}, -5, 726, 62}, 
    {{1, 3, 3, 192}, 612208, -1, 3}, 
    {{1, 192, 1, 1}, 613936, -1, -1}, 
    {{1, 9, 9, 192}, -5, 731, 63}, 
    {{1, 9, 9, 48}, -5, 736, 64}, 
    {{1, 9, 9, 48}, -5, 741, 65}, 
    {{1, 9, 9, 144}, -5, 746, 66}, 
    {{1, 5, 5, 144}, 614704, -1, 3}, 
    {{1, 144, 1, 1}, 618304, -1, -1}, 
    {{1, 9, 9, 144}, -5, 751, 67}, 
    {{1, 9, 9, 48}, -5, 756, 68}, 
    {{1, 9, 9, 48}, -5, 761, 69}, 
    {{1, 9, 9, 192}, -5, 766, 70}, 
    {{1, 3, 3, 192}, 618880, -1, 3}, 
    {{1, 192, 1, 1}, 620608, -1, -1}, 
    {{1, 5, 5, 192}, -3, 771, 71}, 
    {{1, 5, 5, 96}, -3, 774, 72}, 
    {{1, 5, 5, 480}, -3, 777, 73}, 
    {{1, 5, 5, 480}, 621376, -1, 3}, 
    {{1, 480, 1, 1}, 633376, -1, -1}, 
    {{1, 5, 5, 480}, -3, 780, 74}, 
    {{1, 5, 5, 96}, -3, 783, 75}, 
    {{1, 5, 5, 96}, -3, 786, 76}, 
    {{1, 5, 5, 384}, -3, 789, 77}, 
    {{1, 5, 5, 384}, 635296, -1, 3}, 
    {{1, 384, 1, 1}, 644896, -1, -1}, 
    {{1, 5, 5, 384}, -3, 792, 78}, 
    {{1, 5, 5, 96}, -3, 795, 79}, 
    {{1, 5, 5, 96}, -3, 798, 3}, 
    {{1, 5, 5, 384}, -3, 801, 80}, 
    {{1, 3, 3, 384}, 646432, -1, 3}, 
    {{1, 384, 1, 1}, 649888, -1, -1}, 
    {{1, 5, 5, 384}, -3, 804, 81}, 
    {{1, 5, 5, 96}, -3, 807, 82}, 
    {{1, 5, 5, 96}, -3, 810, 83}, 
    {{1, 5, 5, 480}, -3, 813, 84}, 
    {{1, 3, 3, 480}, 651424, -1, 3}, 
    {{1, 480, 1, 1}, 655744, -1, -1}, 
    {{1, 5, 5, 480}, -3, 816, 85}, 
    {{1, 5, 5, 160}, -3, 819, 86}, 
    {{1, 5, 5, 160}, -1, 0, 1}, 
    {{1, 1, 1, 160}, -1, 4000, 2}, 
    {{1, 1, 1, 2}, -1, 0, 5}, 
    {{1, 2, 1, 1}, 657664, -1, 6}, 
    {{1, 2, 1, 1}, -1, 16, 6}
};
const int all_0_zp_cursor = 3;
const int only_zp_cursor = 7;
const int only_zp_start = 483;
const int32_t quant_scale[7] = {
    1006665857, 1031728915, 1031728915, 995552123, 994973054, 1049603279, 1049603279
};
const int8_t quant_zeropoint[563] = {
    -1, -7, -7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -128, -128, 6, -128, -128, 8, -128, -128, -3, 5, -128, -128, -4, -1, -128, -128, -11, 4, -128, -128, 10, -128, -128, 16, -128, -128, 9, 18, -128, -128, -3, 17, -128, -128, 3, -128, -128, 17, 2, -128, -128, -22, 3, -128, -128, 8, 2, -128, -128, -12, -128, -128, -10, -8, -128, -128, -5, 5, -128, -128, 12, 6, -128, -128, 12, -128, -128, -8, 1, -128, -128, -2, -128, -128, -26, -2, -128, -128, -7
};
const int32_t split_offset[822] = {
    0, 864, 1728, 2592, 3456, 4320, 5184, 6048, 6912, 7776, 8640, 9504, 10368, 11232, 12096, 12960, 13824, 14688, 15552, 16416, 17280, 18144, 19008, 19872, 20736, 21600, 22464, 23328, 24192, 25056, 25920, 26784, 27648, 28512, 29376, 30240, 31104, 31968, 32832, 33696, 34560, 35424, 36288, 37152, 38016, 38880, 39744, 40608, 41472, 42336, 43200, 44064, 44928, 45792, 46656, 47520, 48384, 49248, 50112, 50976, 51840, 52704, 53568, 54432, 55296, 56160, 57024, 57888, 58752, 59616, 60480, 61344, 65664, 67968, 71424, 81792, 84096, 87552, 89856, 96768, 105984, 109440, 111744, 116352, 118656, 120960, 123264, 126720, 129024, 79488, 126720, 105984, 79488, 126720, 105984, 130176, 132480, 77184, 84672, 103104, 109440, 71424, 84672, 133632, 135936, 84672, 116352, 17280, 0, 0, 3456, 3456, 6912, 6912, 10368, 10368, 0, 3456, 17280, 17280, 6912, 10368, 0, 3456, 13824, 17280, 6912, 10368, 0, 3456, 13824, 17280, 6912, 10368, 0, 3456, 13824, 17280, 6912, 10368, 0, 3456, 13824, 19584, 70272, 65664, 73728, 5760, 74880, 74880, 72576, 12672, 3456, 5760, 65664, 19584, 10368, 12672, 3456, 5760, 17280, 19584, 10368, 12672, 3456, 5760, 17280, 19584, 10368, 12672, 3456, 5760, 17280, 19584, 10368, 12672, 3456, 5760, 19584, 21888, 62208, 0, 67968, 71424, 67968, 6912, 69120, 93312, 0, 13824, 62208, 99072, 6912, 20736, 0, 104832, 13824, 27648, 6912, 102528, 0, 33408, 13824, 99072, 6912, 73728, 0, 44928, 13824, 67968, 6912, 50688, 0, 54144, 13824, 17280, 65664, 3456, 81792, 10368, 3456, 17280, 10368, 3456, 17280, 10368, 3456, 17280, 10368, 3456, 17280, 10368, 3456, 20736, 74880, 92160, 102528, 114048, 120960, 125568, 74880, 131328, 111744, 131328, 79488, 134784, 99072, 133632, 103104, 125568, 57600, 84672, 62208, 76032, 81792, 87552, 93312, 81792, 99072, 93312, 67968, 27648, 62208, 33408, 39168, 73728, 44928, 67968, 27648, 50688, 0, 67968, 6912, 0, 13824, 6912, 0, 13824, 6912, 0, 13824, 6912, 0, 13824, 6912, 0, 13824, 6912, 5760, 62208, 12672, 5760, 19584, 12672, 5760, 19584, 12672, 5760, 19584, 12672, 5760, 19584, 12672, 5760, 19584, 12672, 93312, 103680, 115200, 122112, 126720, 119808, 132480, 129024, 132480, 80640, 135936, 100224, 134784, 104256, 138240, 108288, 70848, 27648, 0, 62208, 6912, 0, 13824, 6912, 0, 13824, 6912, 0, 13824, 6912, 0, 13824, 6912, 0, 13824, 6912, 69120, 69120, 69120, 62208, 62208, 20736, 20736, 20736, 20736, 20736, 20736, 20736, 20736, 20736, 20736, 20736, 20736, 0, 92160, 0, 62208, 6912, 0, 13824, 6912, 0, 13824, 6912, 0, 13824, 6912, 0, 13824, 6912, 0, 6912, 108288, 116352, 74880, 127872, 67968, 74880, 130176, 133632, 115200, 137088, 101376, 135936, 112896, 139392, 125568, 72000, 26496, 31680, 99072, 102528, 105984, 109440, 116352, 119808, 99072, 116352, 102528, 126720, 123264, 130176, 109440, 122112, 133632, 67392, 50688, 16704, 0, 62208, 6912, 0, 13824, 6912, 0, 13824, 6912, 0, 13824, 6912, 0, 13824, 6912, 0, 0, 5760, 3456, 65664, 10368, 3456, 17280, 10368, 3456, 17280, 10368, 3456, 17280, 10368, 3456, 17280, 10368, 3456, 3456, 9216, 0, 62208, 6912, 0, 13824, 6912, 0, 13824, 6912, 0, 13824, 6912, 0, 13824, 6912, 0, 5760, 16704, 76032, 69120, 87552, 62208, 81792, 73728, 93312, 67968, 27648, 62208, 33408, 39168, 73728, 44928, 56448, 20736, 0, 5760, 6912, 13824, 0, 6912, 13824, 0, 6912, 5760, 16704, 133632, 134784, 138240, 118656, 142560, 140544, 43776, 48096, 10368, 112896, 108288, 119808, 115200, 126720, 130176, 108288, 67392, 16704, 13824, 0, 6912, 13824, 0, 6912, 5760, 0, 38592, 17280, 3456, 10368, 17280, 3456, 10368, 9216, 3456, 16704, 135648, 139104, 140832, 143424, 145152, 85824, 48960, 20160, 25344, 123264, 111744, 105984, 118656, 112896, 122112, 70848, 21888, 52416, 0, 6912, 13824, 0, 6912, 5760, 0, 16704, 5184, 3456, 10368, 17280, 3456, 10368, 9216, 3456, 10368, 8640, 139968, 141696, 144288, 142560, 86688, 49824, 31680, 32544, 52416, 87552, 79488, 92736, 62208, 33408, 11520, 0, 16704, 5184, 13824, 0, 6912, 5760, 5184, 38592, 21888, 27072, 21888, 19008, 5184, 12096, 26496, 21024, 10368, 10368, 10368, 0, 13824, 0, 6912, 5760, 25344, 48096, 27072, 15552, 5184, 97920, 103104, 38592, 26496, 5184, 38592, 21888, 10368, 0, 0, 5760, 48096, 27072, 5184, 102240, 74304, 58752, 15552, 21600, 97920, 43776, 48096, 27072, 37872, 5184, 21888, 0, 0, 15120, 9504, 26208, 4320, 12240, 17280, 59472, 40032, 11520, 3600, 30960, 55872, 34992, 18720, 22320, 27792, 21888, 0, 0, 15120, 7200, 24768, 2880, 2880, 18000, 8640, 40752, 16272, 25200, 35792, 24768, 31392, 4320, 0, 15120, 33984, 0, 11520, 18720, 7200, 7200, 3600, 15120, 3600, 0, 9008, 0, 16992, 7200, 7200, 7200, 7920, 11520, 3600, 0, 32976, 15120, 18720, 7200, 7200, 8256, 33984, 29232, 14256, 17712, 27360, 31392, 25200, 22176, 30384, 32976, 7200, 7200, 3600, 0, 11712, 9792, 9792, 6192, 2592, 13008, 30096, 35792, 3600, 23760, 16896, 18720, 10800, 4800, 17712, 15168, 7200, 14256, 0, 11712, 11712, 27792, 8256, 8256, 4800, 13440, 36656, 34784, 35648, 14304, 29952, 27792, 24768, 21168, 27360, 23760, 14256, 0, 4800, 11712, 9600, 8256, 2592, 7392, 15168, 10896, 9120, 0, 11712, 11712, 11328, 14256, 8256, 4800, 18240, 9600, 0, 11712, 11328, 34272, 25056, 26976, 0, 4800, 22080, 9600, 9600, 0, 14400, 0, 2400, 26016, 3840, 13440, 14400, 9600, 4800, 0, 0, 6720, 13440, 6720, 4800, 24480, 8640, 24000, 18240, 14400, 22080, 0, 4800, 9600, 3840, 9600, 11520, 6720, 10560, 9600, 0, 4800, 14400, 9600, 9600, 0, 18240, 19840, 4000
};
alignas(16) const uint8_t offline_tensor_data[657680] = {
};

int CtxSummary(){
    printf("Arena Size: %d\n", arena_size);
    printf("Tensor Metadata Summary:\n");
    printf("\ttensors: %d\n",sizeof(tensors));
    printf("\tquant_scale: %d\n",36804);
    printf("\tquant_zeropoint: %d\n",563);
    printf("\tsplit_offset: %d\n",sizeof(split_offset));
    printf("\toffline_tensor_data: %d\n",sizeof(offline_tensor_data));

    int byte_tensor =   sizeof(tensors) +
                        36804 + 563 +
                        sizeof(split_offset) + sizeof(offline_tensor_data);
    return byte_tensor;
}