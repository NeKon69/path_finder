//
// Created by progamers on 9/25/25.
//

#pragma once

#include <cuda_runtime.h>

#include "common.h"
namespace gpu {
extern void launch_path_finding(cudaSurfaceObject_t array, position* path,
								type width, type height,
								volatile type* flag, type* path_length, position* points,
								cudaStream_t stream);
}