#include <torch/serialize/tensor.h>
#include <torch/extension.h>


#include "linear_voxel_query_cuda_kernel.h"
#include "insert_hashtable_cuda_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_voxel_query_cuda", &linear_voxel_query_cuda, "linear_voxel_query_cuda");
    m.def("insert_hashtable_cuda", &insert_hashtable_cuda, "insert_hashtable_cuda");
}
