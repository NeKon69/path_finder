//
// Created by progamers on 9/25/25.
//
#include <cooperative_groups.h>
#include <device_atomic_functions.h>
#include <surface_indirect_functions.h>
#include <surface_types.h>

#include <cstdint>

#include "common.h"
#include "gpu/kernel.h"
namespace gpu {
__global__ void simple_path_finding(cudaSurfaceObject_t array, position* start, type width,
									type height, volatile type* global_done_flag, type* length) {
	type						   x	   = blockIdx.x * blockDim.x + threadIdx.x;
	type						   y	   = blockIdx.y * blockDim.y + threadIdx.y;
	bool						   updated = false;
	cooperative_groups::grid_group grid	   = cooperative_groups::this_grid();
	if (x == 0 && y == 0) {
		*global_done_flag = 0;
	}

	if (*start == position {x, y}) {
		surf2Dwrite<type>(1, array, x * sizeof(type), y);
	}

	// if (inside_bounds(x, y)) {
	// printf("I am thread %u, %u, and i have value %u\n", x, y,
	// surf2Dread<type>(array, x * sizeof(type), y));
	// }

	__threadfence();
	grid.sync();

	while ((*global_done_flag) != 1) {
		// printf("I AM TARGET!!! x = %u, y = %u\n", x, y);
		// if (is_target(val)) {
		// 	printf("I AM TARGET!!! %u\n", val);
		// }
		if (inside_bounds(x, y, width, height) &&
			is_target(surf2Dread<type>(array, x * sizeof(type), y)) && !updated) {
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
					// printf("YAY i am writing to global flag %u!!!\n", minimal);
					*global_done_flag = 1;
					*length			  = minimal + 1;
					__threadfence();
				}
				surf2Dwrite<type>(minimal + 1, array, x * sizeof(type), y);
				// printf("After writing to array\n");
				updated = true;
			}
			// printf("I am thread %u, %u, and i have value %u\n", x, y,
			// surf2Dread<type>(array, x * sizeof(type), y));
		}
		grid.sync();
	}
	// if (x == 0 && y == 0) {
	// printf("HOOOOOOOOOOORAY EXITING!!!!\n");
	// }
}
/**
 * @brief Requires to be launched with 1 thread, otherwise unknown things may happen
 */
__global__ void rebuild_path_simple(cudaSurfaceObject_t array, position* path, position* points,
									type* path_length, type width, type height) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		type path_len = *path_length;
		if (path_len == 0) {
			return;
		}

		position current_pos = points[1];
		path[0]				 = current_pos;

		for (type i = 1; i < path_len; ++i) {
			type current_value =
				surf2Dread<type>(array, current_pos.x * sizeof(type), current_pos.y);
			for (int j = 0; j < 4; ++j) {
				type next_x = current_pos.x + dc[j];
				type next_y = current_pos.y + dr[j];
				type val;
				if (inside_bounds(next_y, next_x, width, height) &&
					((val = surf2Dread<type>(array, next_x * sizeof(type), next_y))) &&
					is_path(val) && current_value - 1 == val) {
					current_value = val;
					current_pos	  = {next_x, next_y};
					break;
				}
			}
			path[i] = {current_pos.x, current_pos.y};
		}
	}
}

// __global__ void rebuild_path_simple(const type* __restrict__ array, position* path,
// 									position* points, type* path_length, type width, type height) {
// 	if (blockIdx.x == 0 && threadIdx.x == 0) {
// 		type path_len = *path_length;
// 		if (path_len == 0) {
// 			return;
// 		}
//
// 		position current_pos = points[1];
// 		path[0]				 = current_pos;
//
// 		for (type i = 1; i < path_len; ++i) {
// 			type current_value =
// 				surf2Dread<type>(array, current_pos.x * sizeof(type), current_pos.y);
// 			for (int j = 0; j < 4; ++j) {
// 				type next_x = current_pos.x + dc[j];
// 				type next_y = current_pos.y + dr[j];
// 				type val;
// 				if (inside_bounds(next_y, next_x) &&
// 					((val = surf2Dread<type>(array, next_x * sizeof(type), next_y))) &&
// 					is_path(val) && current_value - 1 == val) {
// 					current_value = val;
// 					current_pos	  = {next_x, next_y};
// 					break;
// 				}
// 			}
// 			path[i] = {current_pos.x, current_pos.y};
// 		}
// 	}
// }
//

__global__ void reconstruct_path_fast(const type* __restrict__ matrix, int width, int height,
									  position end, position* out_path) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx > 0)
		return;

	position curr		= end;
	int		 step_count = 0;

	out_path[step_count++] = curr;

	type current_val = __ldg(&matrix[curr.y * width + curr.x]);

	while (current_val > 0 && current_val < TARGET) {
		bool found_prev = false;
#pragma unroll
		for (int i = 0; i < 4; ++i) {
			int next_x = (int)curr.x + dc[i];
			int next_y = (int)curr.y + dr[i];

			if (next_y >= 0 && next_y < height && next_x >= 0 && next_x < width) {
				int next_idx = next_y * width + next_x;

				type val = __ldg(&matrix[next_idx]);

				if (val == current_val - 1) {
					current_val = val;
					curr.x		= (type)next_x;
					curr.y		= (type)next_y;

					out_path[step_count++] = curr;

					found_prev = true;
					break;
				}
			}
		}

		if (!found_prev) {
			break;
		}
	}
}

__global__ void check_full_array(cudaSurfaceObject_t array, position* points, type width,
								 type height, type cells_per_thread) {
	uint64_t segment_idx = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t segment_idy = threadIdx.y + blockIdx.y * blockDim.y;

	uint64_t start_x = segment_idx * cells_per_thread;
	uint64_t start_y = segment_idy * cells_per_thread;
	if (start_x >= width || start_y >= height) {
		return;
	}

	unsigned int checksum = 0;

	for (uint64_t i = start_x; i < start_x + cells_per_thread; ++i) {
		for (uint64_t j = start_y; j < start_y + cells_per_thread; ++j) {
			type val = surf2Dread<type>(array, i * sizeof(type), j);
			checksum += (unsigned int)val;
		}
	}

	if (checksum > 0 && start_x == 0 && start_y == 0) {
		points[0].x = checksum;
	}
}
__global__ void find_path_chunk(cudaSurfaceObject_t array, position* points, type width,
								type height, type* global_done_flag, type cells_per_thread) {
	// Has the size of maximum size of 1024, so lets allocate max
	__shared__ type chunk_mem[1024];

	// Calculates chunk id
	uint64_t segment_idx = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t segment_idy = threadIdx.y + blockIdx.y * blockDim.y;

	// Calculates start position relative to all chnuks
	uint64_t start_x = segment_idx * cells_per_thread;
	uint64_t start_y = segment_idy * cells_per_thread;

	for (int i = 0; i < cells_per_thread; ++i) {
	}
}

namespace cg						  = cooperative_groups;
__device__ auto&			 act_warp = __activemask;
constexpr __constant__ short dx[4]	  = {0, 0, -1, 1};
constexpr __constant__ short dy[4]	  = {1, -1, 0, 0};

__device__ inline void append_to_queue(position pos, type* q, type* q_cnt, type width) {
	// returns what threads are currently active (means barnced on checking if the cell was changed)
	uint32_t active_threads = act_warp();
	// To prevent threads in current warp that wasn't suspeneded from actually working
	uint32_t predicate_mask = __ballot_sync(active_threads, pos.x + width * pos.y);
	uint32_t warp_tid		= threadIdx.x % 32;

	// Produces result like this
	// Let's say our thread in warp is third
	// then in here it would turn into ...0011
	// So in general it just turns all bits behind current thread in warp to 1
	uint32_t warp_mask = (1U << warp_tid) - 1;
	// Returns threads that are active AND behind us
	uint32_t rank = __popc(predicate_mask & warp_mask);

	uint32_t warp_base_offset = 0;
	uint32_t leader_tid		  = __ffs(active_threads) - 1;
	if (warp_tid == leader_tid) {
		// Count how many threads are active
		uint32_t total_warp_add = __popc(predicate_mask);
		if (total_warp_add) {
			// Atomically add this amount
			warp_base_offset = atomicAdd(q_cnt, total_warp_add);
		}
	}
	// Distribute warp_base_offset accross all threads in warp
	warp_base_offset = __shfl_sync(active_threads, warp_base_offset, leader_tid);

	uint32_t curr_offset = warp_base_offset + rank;
	// Pack 2d coordinates to 1d
	q[curr_offset] = pos.x + width * pos.y;
}

__global__ void find_path_queue(type* array, type* q1, type* q2, type* q1_cnt, type* q2_cnt,
								type width, type height, position start, position end,
								type* path_len, volatile type* finished_flag) {
	cg::grid_group grid			 = cg::this_grid();
	uint32_t	   t_id			 = grid.thread_rank();
	uint32_t	   total_threads = grid.size();

	if (t_id == 0) {
		*q2_cnt							 = 0;
		*finished_flag					 = 0;
		q1[0]							 = start.x + start.y * width;
		*q1_cnt							 = 1;
		array[start.x + start.y * width] = 0;
		array[end.x + end.y * width]	 = EMPTY;
	}

	grid.sync();

	type* curr_q	 = q1;
	type* next_q	 = q2;
	type* curr_q_cnt = q1_cnt;
	type* next_q_cnt = q2_cnt;
	type  depth		 = 1;

	while (*finished_flag != 1 && *curr_q_cnt > 0) {
		int curr_q_size = *curr_q_cnt;

		for (int i = t_id; i < curr_q_size; i += total_threads) {
			type	 curr_node = curr_q[i];
			position curr_pos  = {curr_node % width, curr_node / width};

			if (curr_pos == end) {
				*finished_flag = 1;
			}

#pragma unroll
			for (int i = 0; i < 4; ++i) {
				position next = {curr_pos.x + dx[i], curr_pos.y + dy[i]};

				if (next >= position {0, 0} && next < position {width, height} &&
					next.x != UINT32_MAX && next.y != UINT32_MAX) {
					if (__ldg(&array[next.x + next.y * width]) == EMPTY) {
						if (atomicCAS(&array[next.x + next.y * width], EMPTY, depth) == EMPTY) {
							append_to_queue(next, next_q, next_q_cnt, width);
						}
					}
				}
			}
		}

		grid.sync();

		if (t_id == 0) {
			*curr_q_cnt = 0;
		}
		grid.sync();
		using cuda::std::swap;
		swap(curr_q, next_q);
		swap(curr_q_cnt, next_q_cnt);
		depth++;
	}
	if (t_id == 0) {
		*path_len = depth;
	}
}
} // namespace gpu
