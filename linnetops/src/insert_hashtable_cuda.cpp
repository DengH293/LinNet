#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "insert_hashtable_cuda_kernel.h"

void insert_hashtable_cuda(
    int m,
    int len_hash,
    at::Tensor hashtable_tensor,
    at::Tensor keys_tensor,
    at::Tensor values_tensor) 
{
    uint64_t *hashtable = (uint64_t*)hashtable_tensor.data_ptr();
    uint64_t *keys = (uint64_t*)keys_tensor.data_ptr();
    uint64_t *values = (uint64_t*)values_tensor.data_ptr();

    insert_hashtable_cuda_launcher(m, len_hash, hashtable, keys, values);
}
