//
// Created by progamers on 9/25/25.
//
#include "gpu/algorithm.h"
#include "gpu/kernel.h"
#include <cuda_runtime.h>

namespace gpu {
void find_path(cudaSurfaceObject_t array, position* path, std::vector<position> path_cpu,
			   type width, type height, bool* flag, cudaStream_t stream) {
	void* kernel_args[] = {&array, &width, &height, &flag};
	cudaLaunchCooperativeKernel(simple_path_finding, gridDim, blockDim, kernel_args, 0, stream);
}

} // namespace gpu