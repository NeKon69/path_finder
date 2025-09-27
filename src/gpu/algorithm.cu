//
// Created by progamers on 9/25/25.
//
#include <cuda_runtime.h>

#include <memory>

#include "cuda_wrappers/error.h"
#include "gpu/algorithm.h"
#include "gpu/kernel.h"

namespace gpu {
template<class T>
constexpr std::string_view type_name() {
	using namespace std;
#ifdef __clang__
	string_view p = __PRETTY_FUNCTION__;
	return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
	string_view p = __PRETTY_FUNCTION__;
#if __cplusplus < 201402
	return string_view(p.data() + 36, p.size() - 36 - 1);
#else
	return string_view(p.data() + 49, p.find(';', 49) - 49);
#endif
#elif defined(_MSC_VER)
	string_view p = __FUNCSIG__;
	return string_view(p.data() + 84, p.size() - 84 - 7);
#endif
}
void launch_path_finding(cudaSurfaceObject_t array, position* path, type width, type height,
						 volatile type* flag, type* path_length, position* points,
						 cudaStream_t stream) {
	void* kernel_params[] = {&array, &points, &width, &height, &flag, &path_length};

	std::cout << type_name<decltype(kernel_params)>() << std::endl;
	std::cout << type_name<decltype(simple_path_finding)>() << std::endl;
	dim3 block(8, 8);
	dim3 grid(width / 32 + 1, height / 32 + 1);
	CUDA_SAFE_CALL(
		cudaLaunchCooperativeKernel(simple_path_finding, grid, block, kernel_params, 0, stream));
	CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
	rebuild_path_simple<<<1, 1, 0, stream>>>(array, path, points, path_length, width, height);
	CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
}

} // namespace gpu