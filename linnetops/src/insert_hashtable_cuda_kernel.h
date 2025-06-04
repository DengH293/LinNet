#ifndef INSERT_HASHTABLE_CUDA_KERNEL
#define INSERT_HASHTABLE_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>


void insert_hashtable_cuda(int m, int len_hash,
                      at::Tensor hashtable_tensor, at::Tensor keys_tensor, at::Tensor values_tensor);




void insert_hashtable_cuda_launcher(int m, int len_hash, uint64_t *hashtable,
                                         const uint64_t *__restrict__ keys, const uint64_t *__restrict__ values);


#endif

