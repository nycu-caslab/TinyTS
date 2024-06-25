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

#include "gen_model/include/ctx.h"

#define EXEC_CNT 3

int main(int argc, char *argv[]) {
  // original code
  // ee_benchmark_initialize();
  // while (1) {
  //   int c;
  //   c = th_getchar();
  //   ee_serial_callback(c);
  // }


  ee_benchmark_initialize();
  printf("%s %s\n", MODEL_NAME, __TIMESTAMP__);
  char cmd[] = "infer 1 0%";
  int i = EXEC_CNT;
  while(i--){
    // printf("press enter to start...");
    // char c = getchar();
    // if (c != '\n') putchar('\n');
    for (int i=0;i<10; ++i) {
      ee_serial_callback(cmd[i]);
    }
  }
  printf("\nMin latency: %d\n", latency_min);
  printf("Max latency: %d\n", latency_max);
  printf("Average latency: %d\n", tot_time/EXEC_CNT);
  return 0;
}
