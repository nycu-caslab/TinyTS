/* Copyright 2020 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/// \file
/// \brief Main function to run benchmark on device.

#include "api/internally_implemented.h"
#include "api/submitter_implemented.h"
#include "prof_ctrl.h"
#include "mbed.h"

#define EXEC_CNT 1

extern  char model_name[];
extern  char compile_time[];

extern uint32_t cc_min;
extern uint32_t cc_max;
extern int64_t  cc_tot;

uint32_t tot_time = 0;
uint32_t lat_min = UINT32_MAX;
uint32_t lat_max = 0;

int main(int argc, char *argv[]) {
  // MLPerf Tiny initialization
  ee_benchmark_initialize();
  
  // profile cycle count
  // Timer configuration
  DWT_CYCCNT_EN();
  #if defined(STM32F767xx)
    DWT_CYCCNT_UNLOCK();
  #endif
  wait_us(500000);

  // hard-coded MLPerf command
  char cmd[] = "infer 1 0%";
  // Directly feed infer cmd to MLPerf Tiny's CLI
  // Run for EXEC_CNT times
  for(int j = 0; j < EXEC_CNT; j++){
    for(int i = 0; i < strlen(cmd); i++) {
      ee_serial_callback(cmd[i]);
    }
  }

  // Print cycle count statistics
  // uint32_t cc_avg = cc_tot/EXEC_CNT;
  // printf("\r\nMin cycle count: %d\r\n", cc_min);
  // printf("Max cycle count: %d\r\n", cc_max);
  // printf("Average cycle count: %d\r\n", cc_avg);
  // printf("%s %s\n", compile_time, model_name);

  printf("\r\nMin latency: %d\r\n", lat_min);
  printf("Max latency: %d\r\n", lat_max);
  printf("Average latency: %d\r\n", tot_time/EXEC_CNT);
  printf("%s %s\n", compile_time, model_name);
  
  #if defined(STM32F767xx)
    DWT_CYCCNT_LOCK();
  #endif
  while(1){}
  return 0;
}
