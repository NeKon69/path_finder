//
// Created by progamers on 9/25/25.
//

#include <chrono>
#include <cstdint>
#include <iostream>
// @clang-format off
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
// @clang-format on

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
	auto st = std::chrono::high_resolution_clock::now();
	CUDA_SAFE_CALL(
		cudaLaunchCooperativeKernel(simple_path_finding, grid, block, kernel_params, 0, stream));
	CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "First step took: "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(end - st).count() << " ms"
			  << std::endl;
	rebuild_path_simple<<<1, 1, 0, stream>>>(array, path, points, path_length, width, height);
}

float launch_test_look(cudaSurfaceObject_t array, position* points, type width, type height,
					   cudaEvent_t start, cudaEvent_t end, cudaStream_t stream) {
	// Should be > 255, but also multiple of 32
	constexpr uint32_t threads_per_dim = 32;
	// Should be a multiple of 32
	constexpr uint32_t threads_per_axis = 4;
	constexpr uint32_t cells_for_thread = threads_per_axis * threads_per_axis;

	uint64_t num_cells		  = width * height;
	uint64_t threads_needed_x = width / threads_per_axis;
	uint64_t threads_needed_y = height / threads_per_axis;
	dim3	 block			  = dim3(threads_per_dim, threads_per_dim);
	dim3	 grid;
	grid.x = (threads_needed_x + block.x - 1) / block.x;
	grid.y = (threads_needed_y + block.y - 1) / block.y;

	CUDA_SAFE_CALL(cudaEventRecord(start, stream));
	check_full_array<<<grid, block, 0, stream>>>(array, points, width, height, threads_per_axis);
	if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
		std::cerr << "Error in check_full_array: " << cudaGetErrorString(err) << std::endl;
		throw std::runtime_error("Error in check_full_array");
	}

	CUDA_SAFE_CALL(cudaEventRecord(end, stream));

	float milliseconds = 0;
	CUDA_SAFE_CALL(cudaEventSynchronize(end));
	CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, start, end));
	return milliseconds;
}
} // namespace gpu
