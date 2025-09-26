//
// Created by progamers on 9/25/25.
//

#pragma once

#include <cuda_runtime.h>

#include "common.h"
namespace gpu {
extern __global__ void simple_path_finding(cudaSurfaceObject_t array, type width, type height,
										   volatile bool* global_done_flag, type* length);
extern __global__ void rebuild_path_simple(cudaSurfaceObject_t array, position* path,
										   position* points, type* path_length, type width,
										   type height);
} // namespace gpu