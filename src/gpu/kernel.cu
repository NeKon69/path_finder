//
// Created by progamers on 9/25/25.
//
#include <cooperative_groups.h>

#include <thread>

#include "gpu/kernel.h"
namespace gpu {
__global__ void simple_path_finding(cudaSurfaceObject_t array, type width, type height,
									volatile bool* global_done_flag, type* length) {
	cooperative_groups::grid_group grid = cooperative_groups::this_grid();
	type						   x	= blockIdx.x * blockDim.x + threadIdx.x;
	type						   y	= blockIdx.y * blockDim.y + threadIdx.y;

	while (!(*global_done_flag)) {
		if (inside_bounds(x, y) && is_target(surf2Dread<type>(array, x, y))) {
			type l = 0, r = 0, u = 0, d = 0;
			if (x > 0)
				l = surf2Dread<type>(array, x - 1, y);
			if (x < width - 1)
				r = surf2Dread<type>(array, x + 1, y);
			if (y > 0)
				d = surf2Dread<type>(array, x, y - 1);
			if (y < height - 1)
				u = surf2Dread<type>(array, x, y + 1);

			type minimal = min(l, r, d, u);
			if (minimal < EMPTY) {
				if (is_real_target(surf2Dread<type>(array, x, y))) {
					printf("Before writing to length\n");
					*global_done_flag = true;
					*length			  = minimal + 1;
					printf("After writing to length\n");
					__threadfence();
				}
				surf2Dwrite<type>(minimal + 1, array, x, y);
			}
		}
		grid.sync();
	}
}
/**
 * @brief Requires to be launched with 2 threads, otherwise unknown things may happen
 */
__global__ void rebuild_path_simple(cudaSurfaceObject_t array, position* path, position* points,
									type* path_length, type width, type height) {
	type path_len = *path_length;
	if (path_len == 0) {
		return;
	}

	if (blockIdx.x == 0 && threadIdx.x == 0) {
		position current_pos = points[0];
		path[0]				 = current_pos;
		type steps_to_take	 = (path_len / 2) - (path_len % 2 == 0 ? 1 : 0);

		for (type i = 0; i < steps_to_take; ++i) {
			type current_value = surf2Dread<type>(array, current_pos.first, current_pos.second);

			if (current_pos.first > 0 &&
				surf2Dread<type>(array, current_pos.first - 1, current_pos.second) ==
					current_value + 1) {
				current_pos.first--;
			} else if (current_pos.first < width - 1 &&
					   surf2Dread<type>(array, current_pos.first + 1, current_pos.second) ==
						   current_value + 1) {
				current_pos.first++;
			} else if (current_pos.second > 0 &&
					   surf2Dread<type>(array, current_pos.first, current_pos.second - 1) ==
						   current_value + 1) {
				current_pos.second--;
			} else if (current_pos.second < height - 1 &&
					   surf2Dread<type>(array, current_pos.first, current_pos.second + 1) ==
						   current_value + 1) {
				current_pos.second++;
			}
			path[i + 1] = current_pos;
		}
	}

	if (blockIdx.x == 1 && threadIdx.x == 0) {
		position current_pos = points[1];
		path[path_len - 1]	 = current_pos;
		type steps_to_take	 = path_len / 2;

		for (type i = 0; i < steps_to_take; ++i) {
			type current_value = surf2Dread<type>(array, current_pos.first, current_pos.second);

			if (current_pos.first > 0 &&
				surf2Dread<type>(array, current_pos.first - 1, current_pos.second) ==
					current_value - 1) {
				current_pos.first--;
			} else if (current_pos.first < width - 1 &&
					   surf2Dread<type>(array, current_pos.first + 1, current_pos.second) ==
						   current_value - 1) {
				current_pos.first++;
			} else if (current_pos.second > 0 &&
					   surf2Dread<type>(array, current_pos.first, current_pos.second - 1) ==
						   current_value - 1) {
				current_pos.second--;
			} else if (current_pos.second < height - 1 &&
					   surf2Dread<type>(array, current_pos.first, current_pos.second + 1) ==
						   current_value - 1) {
				current_pos.second++;
			}
			path[path_len - 2 - i] = current_pos;
		}
	}
}
} // namespace gpu