//
// Created by progamers on 9/25/25.
//

#pragma once

#include <cuda_runtime.h>

#include "common.h"
namespace gpu {
extern __global__ inline void check_alignment(position* array) {
	array = array;
}
extern __global__ void simple_path_finding(cudaSurfaceObject_t array, position* starter, type width,
										   type height, volatile type* global_done_flag,
										   type* length);
extern __global__ void rebuild_path_simple(cudaSurfaceObject_t array, position* path,
										   position* points, type* path_length, type width,
										   type height);
extern __global__ void check_full_array(cudaSurfaceObject_t array, position* points, type width,
										type height, type cells_per_thread);
extern __global__ void find_path_queue(type* array, type* q1, type* q2, type* q1_cnt, type* q2_cnt,
									   type width, type height, position start, position end,
									   volatile type* finished_flag);
extern __global__ void rebuild_path_plain(type* array, position* path, position start, position end,
										  type* path_length, type width, type height);

} // namespace gpu
