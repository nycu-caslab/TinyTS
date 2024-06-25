#ifndef GEN_INCLUDE_VIRTUALFP_H_
#define GEN_INCLUDE_VIRTUALFP_H_

#include "gen_lib/include/ctx_util.h"
#include "gen_lib/include/types.h"
#include <stdio.h>

constexpr int kMaxInputNum = 256;  // Maximum number of input rows

template <typename T>
class VirtualFp{
    public:
        VirtualFp(const int tensor_id, const Tensor tensor, int row_begin, int row_end, int row_size)
        {
            // num_splits =  row_end - row_begin + 1;
            int num_rows = row_end - row_begin + 1;
            // Get the splits data
            int idx = 0;
            // process first split separately to eliminate unaligned first row problem
            T *split_data = GetSplitData(tensor_id, row_begin/SPLIT_HEIGHT);
            for (int j = row_begin % SPLIT_HEIGHT; j<SPLIT_HEIGHT && idx<num_rows; row_begin++, j++)
            {
                inputs_data[idx++] = split_data + j*row_size;
            }
            // aligned rows
            for (int i=row_begin; i<=row_end;)
            {
                T *split_data = GetSplitData(tensor_id, i/SPLIT_HEIGHT);
                for (int j = 0; j<SPLIT_HEIGHT && idx<num_rows; i++, j++)
                {
                    inputs_data[idx++] = split_data + j*row_size;
                }
            }

            // Get the sizes of the partitions
            // split_size = GetSplitSize(tensor_id, 0);

            // Set combined dim
            combined_dim[0] = tensor.dims[0];
            combined_dim[1] = num_rows;       // Default: each split has height 2
            combined_dim[2] = tensor.dims[2];
            combined_dim[3] = tensor.dims[3];

        };
        VirtualFp(const int tensor_id, const Tensor tensor) 
        {
            // Convert a entire tensor to a virtualfp
            // split_size = GetTensorSize(tensor_id);
            inputs_data[0] = GetTensorData(tensor_id);
            
            // Set combined dim
            combined_dim[0] = tensor.dims[0];
            combined_dim[1] = tensor.dims[1];       // Default: each split has height 1
            combined_dim[2] = tensor.dims[2];
            combined_dim[3] = tensor.dims[3];
        };
        // T operator[](int index) const
        // {
        //     int idx = index % split_size;
        //     int partition = index / split_size;
        //     return inputs_data[partition][idx];
        // };

        // int num_splits;     // the number of the splits to combine
        // int split_size;      // sizes of each split
        
        T* inputs_data[kMaxInputNum];
        
        int combined_dim[4];
};

#endif //GEN_INCLUDE_VIRTUALFP_H_