//
// Created by progamers on 9/25/25.
//
#include <cooperative_groups.h>

#include "gpu/kernel.h"

// check lu
__device__ __host__ bool inside_bounds(type row, type col) {
	return row >= 0 && row < SIZE && col >= 0 && col < SIZE;
}
__device__ __host__ bool is_target(type val) {
	return val == TARGET || val == EMPTY;
}
__device__ __host__ bool is_marked(type val) {
	return val > 0 && val < EMPTY;
}
__device__ __host__ bool is_real_target(type val) {
	return val == TARGET;
}
__device__ type min(type a, type b, type c, type d) {
	type vals[4] = {a, b, c, d};
	type min	 = EMPTY;
	for (auto& val : vals) {
		if (!is_marked(val)) {
			val = WALL;
			continue;
		}
		if (min > val) {
			min = val;
		}
	}
	return min;
}

__global__ void simple_path_finding(cudaSurfaceObject_t array, type width, type height,
									volatile bool* global_done_flag) {
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
					*global_done_flag = true;
					__threadfence();
				}
				surf2Dwrite<type>(minimal + 1, array, x, y);
			}
		}
		grid.sync();
	}
}