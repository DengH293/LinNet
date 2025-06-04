#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "linear_voxel_query_cuda_kernel.h"


void linear_voxel_query_cuda(int m, int nsample, int len_hash,
                              float grid_size,
                              at::Tensor hashtable_tensor,
                              at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
                              at::Tensor new_grid_id_tensor, at::Tensor new_offset_tensor,
                              at::Tensor idx_tensor, at::Tensor dist_tensor)
{
    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const uint64_t *new_grid_id = (uint64_t*)new_grid_id_tensor.data_ptr();
    const int *new_offset = new_offset_tensor.data_ptr<int>();
    const uint64_t *hashtable = (uint64_t*)hashtable_tensor.data_ptr();


    int *idx = idx_tensor.data_ptr<int>();
    float *dist = dist_tensor.data_ptr<float>();

    linear_voxel_query_cuda_launcher(m, nsample, len_hash,
                              grid_size,
                              hashtable,
                              xyz, new_xyz,
                              new_grid_id, new_offset,
                              idx, dist);
}