//
// Created by progamers on 9/25/25.
//
#include <cuda_runtime.h>

#include <memory>

#include "cuda_wrappers/error.h"
#include "gpu/algorithm.h"
#include "gpu/kernel.h"

namespace gpu {
size_t operator*(const dim3& lhs, const dim3& rhs) {
	return lhs.x * lhs.y * lhs.z * rhs.x + rhs.y * rhs.z;
}
void launch_path_finding(cudaSurfaceObject_t array, position* path, type width, type height,
						 volatile type* flag, type* path_length, position* points,
						 cudaStream_t stream) {
	void* kernel_params[] = {&array, &points, &width, &height, &flag, &path_length};

	dim3 block(8, 8);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	if (block * grid > width * height) {
		throw std::runtime_error("Whell happened bro, we don't have enough threads");
	}
	CUDA_SAFE_CALL(
		cudaLaunchCooperativeKernel(simple_path_finding, grid, block, kernel_params, 0, stream));
	CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
	rebuild_path_simple<<<1, 1, 0, stream>>>(array, path, points, path_length, width, height);
}

} // namespace gpu