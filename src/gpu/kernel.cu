//
// Created by progamers on 9/25/25.
//
#include <cooperative_groups.h>

#include <thread>

#include "gpu/kernel.h"
namespace gpu {
__global__ void simple_path_finding(cudaSurfaceObject_t array, position* start, type width,
									type height, volatile type* global_done_flag, type* length) {
	type						   x	= blockIdx.x * blockDim.x + threadIdx.x;
	type						   y	= blockIdx.y * blockDim.y + threadIdx.y;
	cooperative_groups::grid_group grid = cooperative_groups::this_grid();
	if (x == 0 && y == 0) {
		*global_done_flag = 0;
	}

	if (*start == position {x, y}) {
		printf("I AM TARGET!!! %u\n", 1);
		surf2Dwrite<type>(1, array, x * sizeof(type), y);
		__threadfence();
	}

	grid.sync();

	while ((*global_done_flag) != 1) {
		// printf("I AM TARGET!!! x = %u, y = %u\n", x, y);
		// if (is_target(val)) {
		// 	printf("I AM TARGET!!! %u\n", val);
		// }
		if (inside_bounds(x, y) && is_target(surf2Dread<type>(array, x * sizeof(type), y))) {
			auto val = surf2Dread<type>(array, x * sizeof(type), y);
			printf("I AM TARGET!!! %u\n", val);
			type l = 0, r = 0, u = 0, d = 0;
			if (x > 0)
				l = surf2Dread<type>(array, (x - 1) * sizeof(type), y);
			if (x < width - 1)
				r = surf2Dread<type>(array, (x + 1) * sizeof(type), y);
			if (y > 0)
				d = surf2Dread<type>(array, x * sizeof(type), y - 1);
			if (y < height - 1)
				u = surf2Dread<type>(array, x * sizeof(type), y + 1);

			type minimal = min(l, r, d, u);
			if (minimal < EMPTY) {
				if (is_real_target(surf2Dread<type>(array, x * sizeof(type), y))) {
					printf("YAY i am writing to global flag %u!!!\n", minimal);
					*global_done_flag = 1;
					*length			  = minimal + 1;
					__threadfence();
				}
				surf2Dwrite<type>(minimal + 1, array, x * sizeof(type), y);
				printf("After writing to array\n");
			}
		}
		grid.sync();
	}
	if (x == 0 && y == 0) {
		printf("HOOOOOOOOOOORAY EXITING!!!!\n");
	}
}
/**
 * @brief Requires to be launched with 2 threads, otherwise unknown things may happen
 */
__global__ void rebuild_path_simple(cudaSurfaceObject_t array, position* path, position* points,
									type* path_length, type width, type height) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		type path_len = *path_length;
		if (path_len == 0) {
			return;
		}

		position current_pos = points[1];
		auto starter_pos = current_pos;
		path[0]				 = current_pos;

		// printf("My current pos: x = %u, y = %u, val: %u\n", current_pos.first,
		// current_pos.second, surf2Dread<type>(array, current_pos.first * sizeof(type),
		// current_pos.second));

		for (type i = 1; i < path_len; ++i) {
			type min =
				surf2Dread<type>(array, current_pos.first * sizeof(type), current_pos.second);
			for (int j = 0; j < 4; ++j) {
				type next_row = current_pos.first + dr[j];
				type next_col = current_pos.second + dc[j];
				type val;
				if (inside_bounds(next_row, next_col) &&
					((val = surf2Dread<type>(array, next_row * sizeof(type), next_col))) &&
					is_path(val) && min - 1 == val) {

					min = val;
					// printf("My current pos: x = %u, y = %u, val: %u\n", current_pos.first,
					// current_pos.second,
					// surf2Dread<type>(array, current_pos.first * sizeof(type),
					// current_pos.second));
					current_pos = {next_row, next_col};
				}
			}
			path[i] = {current_pos.first, current_pos.second};
			printf("My current pos: x = %u, y = %u\n", current_pos.first, current_pos.second);
		}
	}
}
} // namespace gpu