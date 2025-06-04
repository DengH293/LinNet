#include "./cuda_utils.h"
#include "linear_voxel_query_cuda_kernel.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/serialize/tensor.h>

namespace voxel_query {

__device__ inline void swap(float *a, float *b) {
    float tmp = *a;
    *a = *b;
    *b = tmp;
}

__device__ inline void swap(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

__device__ void heapify(float *dist, int *idx, int k) {
    int root = 0, child = 2 * root + 1;
    while (child < k) {
        if (child + 1 < k && dist[child + 1] > dist[child])
            child++;
        if (dist[root] > dist[child]) return;
        swap(&dist[root], &dist[child]);
        swap(&idx[root], &idx[child]);
        root = child;
        child = 2 * root + 1;
    }
}

__device__ void heap_sort(float *dist, int *idx, int k) {
    for (int i = k - 1; i > 0; --i) {
        swap(&dist[0], &dist[i]);
        swap(&idx[0], &idx[i]);
        heapify(dist, idx, i);
    }
}

__device__ int get_batch_index(int idx, const int *offset) {
    int i = 0;
    while (true) {
        if (idx < offset[i]) return i;
        i++;
    }
}

__device__ uint64_t murmur_hash(uint64_t k, int len_hash) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k & (len_hash - 1);
}

__device__ uint64_t hash_table_lookup(const uint64_t *hashtable, int len_hash, uint64_t key) {
    uint64_t slot = murmur_hash(key, len_hash);
    while (true) {
        if (hashtable[2 * slot] == key)
            return hashtable[2 * slot + 1];
        if (hashtable[2 * slot] == kEmpty)
            return kEmpty;
        slot = (slot + 1) & (len_hash - 1);
    }
}

__device__ uint64_t encode_voxel_id(int x, int y, int z, int batch) {
    return ((uint64_t)batch << 54) |
           ((uint64_t)z << 36) |
           ((uint64_t)y << 18) |
           ((uint64_t)x << 0);
}

} // namespace voxel_query


__global__ void linear_voxel_query_cuda_kernel(
    int m, int nsample, int len_hash, float grid_size,
    const uint64_t *__restrict__ hashtable,
    const float *__restrict__ xyz, const float *__restrict__ new_xyz,
    const uint64_t *__restrict__ new_grid_id,
    const int *__restrict__ new_offset,
    int *__restrict__ idx, float *__restrict__ dist) {

    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= m) return;

    const float *query = new_xyz + pt_idx * 3;
    int *out_idx = idx + pt_idx * nsample;
    float *out_dist = dist + pt_idx * nsample;
    uint64_t gid = new_grid_id[pt_idx];

    int batch_idx = voxel_query::get_batch_index(pt_idx, new_offset);
    float qx = query[0], qy = query[1], qz = query[2];

    int vx = (gid >> 0) & ((1 << 18) - 1);
    int vy = (gid >> 18) & ((1 << 18) - 1);
    int vz = (gid >> 36) & ((1 << 18) - 1);

    float best_dist[128];
    int best_idx[128];
    for (int i = 0; i < nsample; ++i) {
        best_dist[i] = 1e10f;
        best_idx[i] = -1;
    }

    for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dz = -1; dz <= 1; ++dz) {
        uint64_t key = voxel_query::encode_voxel_id(vx + dx, vy + dy, vz + dz, batch_idx);
        uint64_t val = voxel_query::hash_table_lookup(hashtable, len_hash, key);
        if (val != kEmpty) {
            int start = static_cast<int>(val & 0xFFFFFFFF);
            int count = static_cast<int>(val >> 32);
            for (int i = 0; i < count; ++i) {
                int pt = start + i;
                float x = xyz[pt * 3 + 0];
                float y = xyz[pt * 3 + 1];
                float z = xyz[pt * 3 + 2];
                float d2 = (qx - x) * (qx - x) + (qy - y) * (qy - y) + (qz - z) * (qz - z);
                if (d2 < best_dist[0]) {
                    best_dist[0] = d2;
                    best_idx[0] = pt;
                    voxel_query::heapify(best_dist, best_idx, nsample);
                }
            }
        }
    }

    voxel_query::heap_sort(best_dist, best_idx, nsample);
    for (int i = 0; i < nsample; ++i) {
        out_idx[i] = (best_idx[i] != -1) ? best_idx[i] : best_idx[0];
        out_dist[i] = (best_idx[i] != -1) ? best_dist[i] : best_dist[0];
    }
}


void linear_voxel_query_cuda_launcher(
    int m, int nsample, int len_hash, float grid_size,
    const uint64_t *hashtable,
    const float *xyz, const float *new_xyz,
    const uint64_t *new_grid_id,
    const int *new_offset,
    int *idx, float *dist) {

    dim3 blocks((m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);

    linear_voxel_query_cuda_kernel<<<blocks, threads, 0>>>(
        m, nsample, len_hash, grid_size,
        hashtable, xyz, new_xyz,
        new_grid_id, new_offset,
        idx, dist
    );
}
