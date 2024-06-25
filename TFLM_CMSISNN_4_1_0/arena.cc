#include "arena.h"

// variables or literal macro defs
const int kTensorArenaSize = 285*1024;
uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(32)));

// function defs

