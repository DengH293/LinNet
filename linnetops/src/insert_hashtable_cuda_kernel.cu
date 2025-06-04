#include "./cuda_utils.h"
#include "insert_hashtable_cuda_kernel.h"
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>


__device__ uint64_t hash(uint64_t k, int len_hash) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccd;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53;
    k ^= k >> 33;
    return k & (len_hash - 1);
}






__global__ void insert_hashtable_cuda_kernel(int m, int len_hash, uint64_t *hashtable,
                                        const uint64_t *__restrict__ keys, const uint64_t *__restrict__ values) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= m) return;

    uint64_t key = keys[pt_idx];
    uint64_t value = values[pt_idx];
    uint64_t slot = hash(key, len_hash);
    unsigned long long kEmpty = 0;

    while (true) {
        unsigned long long prev = atomicCAS((unsigned long long*)&hashtable[slot*2], kEmpty, key);
        if (prev == kEmpty || prev == key) {
            hashtable[slot*2+1] = value;
            return;
        }
        slot = (slot + 1) & (len_hash - 1);
    }
}


void insert_hashtable_cuda_launcher(int m, int len_hash, uint64_t *hashtable,
                                         const uint64_t *__restrict__ keys, const uint64_t *__restrict__ values)
{
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    insert_hashtable_cuda_kernel<<<blocks, threads, 0>>>(m, len_hash, hashtable, keys, values
                                                    );
}


