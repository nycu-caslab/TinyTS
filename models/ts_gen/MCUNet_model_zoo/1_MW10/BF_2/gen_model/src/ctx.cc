#include "gen_model/include/ctx.h"
#include <cstdio>

alignas(16) int8_t arena[50944];
const int32_t arena_size = 50944;
const int32_t input_tid = 0;
const int32_t output_tid = 149;
const Tensor tensors[150] = {
    {{1, 64, 64, 3}, -1, 0, 0}, 
    {{16, 3, 3, 3}, 0, -1, 3}, 
    {{8, 1, 1, 16}, 432, -1, 3}, 
    {{48, 1, 1, 8}, 560, -1, 3}, 
    {{16, 1, 1, 48}, 944, -1, 3}, 
    {{64, 1, 1, 16}, 1712, -1, 3}, 
    {{16, 1, 1, 64}, 2736, -1, 3}, 
    {{80, 1, 1, 16}, 3760, -1, 3}, 
    {{24, 1, 1, 80}, 5040, -1, 3}, 
    {{96, 1, 1, 24}, 6960, -1, 3}, 
    {{24, 1, 1, 96}, 9264, -1, 3}, 
    {{96, 1, 1, 24}, 11568, -1, 3}, 
    {{24, 1, 1, 96}, 13872, -1, 3}, 
    {{120, 1, 1, 24}, 16176, -1, 3}, 
    {{40, 1, 1, 120}, 19056, -1, 3}, 
    {{160, 1, 1, 40}, 23856, -1, 3}, 
    {{40, 1, 1, 160}, 30256, -1, 3}, 
    {{120, 1, 1, 40}, 36656, -1, 3}, 
    {{40, 1, 1, 120}, 41456, -1, 3}, 
    {{120, 1, 1, 40}, 46256, -1, 3}, 
    {{48, 1, 1, 120}, 51056, -1, 3}, 
    {{192, 1, 1, 48}, 56816, -1, 3}, 
    {{48, 1, 1, 192}, 66032, -1, 3}, 
    {{240, 1, 1, 48}, 75248, -1, 3}, 
    {{96, 1, 1, 240}, 86768, -1, 3}, 
    {{480, 1, 1, 96}, 109808, -1, 3}, 
    {{96, 1, 1, 480}, 155888, -1, 3}, 
    {{384, 1, 1, 96}, 201968, -1, 3}, 
    {{160, 1, 1, 384}, 238832, -1, 3}, 
    {{2, 1, 1, 160}, 300272, -1, 3}, 
    {{1, 16, 1, 1}, 300592, -1, -1}, 
    {{1, 8, 1, 1}, 300656, -1, -1}, 
    {{1, 48, 1, 1}, 300688, -1, -1}, 
    {{1, 16, 1, 1}, 300880, -1, -1}, 
    {{1, 64, 1, 1}, 300944, -1, -1}, 
    {{1, 16, 1, 1}, 301200, -1, -1}, 
    {{1, 80, 1, 1}, 301264, -1, -1}, 
    {{1, 24, 1, 1}, 301584, -1, -1}, 
    {{1, 96, 1, 1}, 301680, -1, -1}, 
    {{1, 24, 1, 1}, 302064, -1, -1}, 
    {{1, 96, 1, 1}, 302160, -1, -1}, 
    {{1, 24, 1, 1}, 302544, -1, -1}, 
    {{1, 120, 1, 1}, 302640, -1, -1}, 
    {{1, 40, 1, 1}, 303120, -1, -1}, 
    {{1, 160, 1, 1}, 303280, -1, -1}, 
    {{1, 40, 1, 1}, 303920, -1, -1}, 
    {{1, 120, 1, 1}, 304080, -1, -1}, 
    {{1, 40, 1, 1}, 304560, -1, -1}, 
    {{1, 120, 1, 1}, 304720, -1, -1}, 
    {{1, 48, 1, 1}, 305200, -1, -1}, 
    {{1, 192, 1, 1}, 305392, -1, -1}, 
    {{1, 48, 1, 1}, 306160, -1, -1}, 
    {{1, 240, 1, 1}, 306352, -1, -1}, 
    {{1, 96, 1, 1}, 307312, -1, -1}, 
    {{1, 480, 1, 1}, 307696, -1, -1}, 
    {{1, 96, 1, 1}, 309616, -1, -1}, 
    {{1, 384, 1, 1}, 310000, -1, -1}, 
    {{1, 160, 1, 1}, 311536, -1, -1}, 
    {{1, 2, 1, 1}, 312176, -1, -1}, 
    {{1, 2, 1, 1}, 312192, -1, 5}, 
    {{1, 2, 1, 1}, 312208, -1, 5}, 
    {{1, 2, 1, 1}, 312224, -1, 5}, 
    {{1, 1, 1, 1}, 312240, -1, 5}, 
    {{1, 64, 64, 3}, -32, 0, 7}, 
    {{1, 32, 32, 16}, -16, 32, 8}, 
    {{1, 3, 3, 16}, 312256, -1, 3}, 
    {{1, 16, 1, 1}, 312400, -1, -1}, 
    {{1, 32, 32, 16}, -16, 48, 9}, 
    {{1, 32, 32, 8}, -16, 64, 3}, 
    {{1, 32, 32, 48}, -16, 80, 10}, 
    {{1, 5, 5, 48}, 312464, -1, 3}, 
    {{1, 48, 1, 1}, 313664, -1, -1}, 
    {{1, 2, 1, 1}, 313856, -1, 5}, 
    {{1, 16, 16, 48}, -8, 96, 11}, 
    {{1, 2, 1, 1}, 313872, -1, 5}, 
    {{1, 16, 16, 16}, -8, 104, 12}, 
    {{1, 16, 16, 64}, -8, 112, 13}, 
    {{1, 3, 3, 64}, 313888, -1, 3}, 
    {{1, 64, 1, 1}, 314464, -1, -1}, 
    {{1, 16, 16, 64}, -8, 120, 14}, 
    {{1, 16, 16, 16}, -8, 128, 15}, 
    {{1, 16, 16, 16}, -8, 136, 16}, 
    {{1, 16, 16, 80}, -8, 144, 17}, 
    {{1, 3, 3, 80}, 314720, -1, 3}, 
    {{1, 80, 1, 1}, 315440, -1, -1}, 
    {{1, 8, 8, 80}, -4, 152, 18}, 
    {{1, 8, 8, 24}, -4, 156, 19}, 
    {{1, 8, 8, 96}, -4, 160, 20}, 
    {{1, 3, 3, 96}, 315760, -1, 3}, 
    {{1, 96, 1, 1}, 316624, -1, -1}, 
    {{1, 8, 8, 96}, -4, 164, 21}, 
    {{1, 8, 8, 24}, -4, 168, 22}, 
    {{1, 8, 8, 24}, -4, 172, 23}, 
    {{1, 8, 8, 96}, -4, 176, 24}, 
    {{1, 3, 3, 96}, 317008, -1, 3}, 
    {{1, 96, 1, 1}, 317872, -1, -1}, 
    {{1, 8, 8, 96}, -4, 180, 25}, 
    {{1, 8, 8, 24}, -4, 184, 26}, 
    {{1, 8, 8, 24}, -4, 188, 27}, 
    {{1, 8, 8, 120}, -4, 192, 28}, 
    {{1, 3, 3, 120}, 318256, -1, 3}, 
    {{1, 120, 1, 1}, 319344, -1, -1}, 
    {{1, 4, 4, 120}, -2, 196, 29}, 
    {{1, 4, 4, 40}, -2, 198, 30}, 
    {{1, 4, 4, 160}, -2, 200, 31}, 
    {{1, 3, 3, 160}, 319824, -1, 3}, 
    {{1, 160, 1, 1}, 321264, -1, -1}, 
    {{1, 4, 4, 160}, -2, 202, 32}, 
    {{1, 4, 4, 40}, -2, 204, 33}, 
    {{1, 4, 4, 40}, -2, 206, 34}, 
    {{1, 4, 4, 120}, -2, 208, 35}, 
    {{1, 7, 7, 120}, 321904, -1, 3}, 
    {{1, 120, 1, 1}, 327792, -1, -1}, 
    {{1, 2, 1, 1}, 328272, -1, 5}, 
    {{1, 4, 4, 120}, -2, 210, 36}, 
    {{1, 2, 1, 1}, 328288, -1, 5}, 
    {{1, 4, 4, 40}, -2, 212, 37}, 
    {{1, 4, 4, 40}, -2, 214, 38}, 
    {{1, 4, 4, 120}, -2, 216, 39}, 
    {{1, 3, 3, 120}, 328304, -1, 3}, 
    {{1, 120, 1, 1}, 329392, -1, -1}, 
    {{1, 4, 4, 120}, -2, 218, 40}, 
    {{1, 4, 4, 48}, -2, 220, 41}, 
    {{1, 4, 4, 192}, -2, 222, 42}, 
    {{1, 3, 3, 192}, 329872, -1, 3}, 
    {{1, 192, 1, 1}, 331600, -1, -1}, 
    {{1, 4, 4, 192}, -2, 224, 43}, 
    {{1, 4, 4, 48}, -2, 226, 44}, 
    {{1, 4, 4, 48}, -2, 228, 45}, 
    {{1, 4, 4, 240}, -2, 230, 46}, 
    {{1, 7, 7, 240}, 332368, -1, 3}, 
    {{1, 240, 1, 1}, 344128, -1, -1}, 
    {{1, 2, 2, 240}, -1, 232, 47}, 
    {{1, 2, 2, 96}, -1, 233, 48}, 
    {{1, 2, 2, 480}, -1, 234, 49}, 
    {{1, 5, 5, 480}, 345088, -1, 3}, 
    {{1, 480, 1, 1}, 357088, -1, -1}, 
    {{1, 2, 2, 480}, -1, 235, 50}, 
    {{1, 2, 2, 96}, -1, 236, 51}, 
    {{1, 2, 2, 96}, -1, 237, 3}, 
    {{1, 2, 2, 384}, -1, 238, 52}, 
    {{1, 7, 7, 384}, 359008, -1, 3}, 
    {{1, 384, 1, 1}, 377824, -1, -1}, 
    {{1, 2, 2, 384}, -1, 239, 53}, 
    {{1, 2, 2, 160}, -1, 240, 54}, 
    {{1, 2, 2, 160}, -1, 640, 1}, 
    {{1, 1, 1, 160}, -1, 0, 2}, 
    {{1, 1, 1, 2}, -1, 160, 5}, 
    {{1, 2, 1, 1}, 379360, -1, 6}, 
    {{1, 2, 1, 1}, -1, 0, 6}
};
const int all_0_zp_cursor = 3;
const int only_zp_cursor = 7;
const int only_zp_start = 483;
const int32_t quant_scale[7] = {
    1006665857, 1026804189, 1026804189, 989903502, 990978304, 1036589333, 1036589333
};
const int8_t quant_zeropoint[531] = {
    -1, 12, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -128, -128, -128, -128, -22, -128, -128, 6, -7, -128, -128, 5, -128, -128, 2, 1, -128, -128, -6, 7, -128, -128, -18, -128, -128, -8, -9, -128, -128, 25, -11, -128, -128, -2, -128, -128, -11, -1, -128, -128, -15, -128, -128, 4, -128, -128, 12
};
const int32_t split_offset[241] = {
    0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3072, 3456, 3840, 4224, 4608, 4992, 5376, 5760, 6144, 6528, 6912, 7296, 7680, 8064, 8448, 8832, 9216, 9600, 9984, 10368, 10752, 11136, 11520, 11904, 12288, 13312, 0, 14336, 1024, 2048, 3072, 15360, 4096, 5120, 6144, 16384, 7168, 8192, 9216, 17408, 10240, 11264, 12288, 13312, 0, 14336, 1024, 2048, 3072, 15360, 4096, 5120, 6144, 16384, 7168, 8192, 9216, 9728, 10240, 12288, 17408, 18432, 21504, 24576, 27648, 30720, 33792, 36864, 39936, 43008, 46080, 49152, 0, 3072, 6144, 9216, 12288, 15360, 18432, 21504, 24576, 27648, 30720, 33792, 36864, 39936, 43008, 46080, 49152, 0, 1536, 3072, 4608, 6144, 7680, 9216, 20480, 20992, 21504, 22016, 22528, 23040, 23552, 24064, 0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 0, 2048, 4096, 6144, 8192, 10240, 12288, 12800, 13312, 0, 512, 1024, 1536, 2048, 2560, 5120, 7680, 10240, 12800, 15360, 17920, 20480, 0, 2560, 5120, 7680, 10240, 12800, 15360, 17920, 20480, 0, 1280, 2560, 9216, 9600, 9984, 10368, 0, 1536, 3072, 4608, 6144, 7680, 0, 1536, 3072, 3456, 3840, 0, 10752, 9216, 9600, 9984, 0, 1536, 3072, 4608, 6144, 7680, 0, 1536, 3072, 3456, 3840, 0, 1920, 4224, 5760, 7680, 0, 1920, 3840, 5760, 7680, 0, 5120, 5440, 0, 1280, 2560, 3840, 0, 320, 3840, 4160, 0, 960, 1920, 2880, 0, 320, 960, 1920, 0, 960, 1920, 2880, 6144, 6528, 0, 1536, 3072, 4608, 0, 384, 1920, 3840, 0, 1920, 3840, 4800, 0, 1920, 0, 1536, 0, 1536, 0
};
alignas(16) const uint8_t offline_tensor_data[379376] = {
};

int CtxSummary(){
    printf("Arena Size: %d\n", arena_size);
    printf("Tensor Metadata Summary:\n");
    printf("\ttensors: %d\n",sizeof(tensors));
    printf("\tquant_scale: %d\n",20676);
    printf("\tquant_zeropoint: %d\n",531);
    printf("\tsplit_offset: %d\n",sizeof(split_offset));
    printf("\toffline_tensor_data: %d\n",sizeof(offline_tensor_data));

    int byte_tensor =   sizeof(tensors) +
                        20676 + 531 +
                        sizeof(split_offset) + sizeof(offline_tensor_data);
    return byte_tensor;
}