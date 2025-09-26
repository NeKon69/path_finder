//
// Created by progamers on 9/25/25.
//
#include <cuda_runtime.h>

#include "cuda_wrappers/error.h"
#include "gpu/algorithm.h"
#include "gpu/kernel.h"

namespace gpu {
void launch_path_finding(cudaSurfaceObject_t array, position* path, type width, type height,
						 bool* flag, type* path_length, position* points, cudaStream_t stream) {
	void* kernel_args[] = {&array, &width, &height, &flag, &path_length};
	dim3  block(8, 8);
	dim3  grid(width / 32 + 1, height / 32 + 1);
	CUDA_SAFE_CALL(
		cudaLaunchCooperativeKernel(simple_path_finding, grid, block, kernel_args, 0, stream));
	CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
	rebuild_path_simple<<<2, 1, 0, stream>>>(array, path, points, path_length, width, height);
	CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
}

} // namespace gpu