//
// Created by progamers on 9/25/25.
//

#pragma once

#include <cuda_runtime.h>

#include "common.h"
namespace gpu {
extern void find_path(cudaSurfaceObject_t array, position* path, std::vector<position> path_cpu,
					  type width, type height, bool* flag, cudaStream_t stream);
}