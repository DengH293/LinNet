#ifndef _LINEAR_VOXEL_QUERY_CUDA_KERNEL
#define _LINEAR_VOXEL_QUERY_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

const uint64_t kEmpty = 0;


void linear_voxel_query_cuda(int m, int nsample, int len_hash,
                              float grid_size,
                              at::Tensor hashtable_tensor,
                              at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
                              at::Tensor new_grid_id_tensor, at::Tensor new_offset_tensor,
                              at::Tensor idx_tensor, at::Tensor dist_tensor);


#ifdef __cplusplus
extern "C" {
#endif


void linear_voxel_query_cuda_launcher(int m, int nsample, int len_hash, float grid_size,
                               const uint64_t *hashtable,
                               const float *xyz, const float *new_xyz,
                               const uint64_t *new_grid_id, const int *new_offset,
                               int *idx, float *dist) ;

#ifdef __cplusplus
}
#endif
#endif
