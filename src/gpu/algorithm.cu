//
// Created by progamers on 9/25/25.
//
#include <cuda_runtime.h>

#include "cuda_wrappers/error.h"
#include "gpu/algorithm.h"
#include "gpu/kernel.h"

namespace gpu {
void launch_path_finding(cudaSurfaceObject_t array, position* path, type width, type height,
						 volatile type* flag, type* path_length, position* points,
						 cudaStream_t stream) {
	struct {
		cudaSurfaceObject_t array;
		type				width;
		type				height;
		volatile type*		flag;
		type*				path_length;
	} args {};
	args.array			  = array;
	args.width			  = width;
	args.height			  = height;
	args.flag			  = flag;
	args.path_length	  = path_length;
	void* kernel_params[] = {&args.array,  &points[0], &args.width,
							 &args.height, &args.flag, &args.path_length};
	dim3  block(8, 8);
	dim3  grid(width / 32 + 1, height / 32 + 1);
	CUDA_SAFE_CALL(
		cudaLaunchCooperativeKernel(simple_path_finding, grid, block, kernel_params, 0, stream));
	CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
	rebuild_path_simple<<<2, 1, 0, stream>>>(array, path, points, path_length, width, height);
	CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
}

} // namespace gpu