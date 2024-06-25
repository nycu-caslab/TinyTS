#ifndef ARENA_H_
#define ARENA_H_
#include <stdint.h>

// AAML tinyML Lab: ARENA_SIZE is defined in arena.h
#define ARENA_SIZE kTensorArenaSize
extern const int kTensorArenaSize;
extern uint8_t tensor_arena[];

#endif