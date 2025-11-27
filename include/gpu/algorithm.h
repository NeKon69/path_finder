//
// Created by progamers on 9/25/25.
//

#pragma once

#include <cuda_runtime.h>
#include <driver_types.h>
#include <surface_types.h>

#include "common.h"
namespace gpu {
extern void	 launch_path_finding(cudaSurfaceObject_t array, position* path, type width, type height,
								 volatile type* flag, type* path_length, position* points,
								 cudaStream_t stream);
extern float launch_test_look(cudaSurfaceObject_t array, position* points, type width, type height,
							  cudaEvent_t start, cudaEvent_t end, cudaStream_t stream);
extern float launch_queue_pf(type* array, type* q1, type* q2, type* q1_cnt, type* q2_cnt,
							 type width, type height, position start, position end,
							 type* finished_flag, cudaEvent_t start_event, cudaEvent_t end_event,
							 cudaStream_t stream);
} // namespace gpu
