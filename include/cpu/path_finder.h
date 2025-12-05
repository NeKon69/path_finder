#pragma once

#include <chrono>
#include <vector>

#include "common.h"
#include "cpu/path_rebuilder.h"

namespace cpu {

class path_finder {
private:
	position		  start, end;
	int				  width, height;
	std::vector<type> matrix;
	// We use 1 vector for double buffering so, the first half of vector is dedicated to one queue,
	// second half to another
	std::vector<type> queue;

	uint64_t offset;

public:
	std::vector<type> preallocate_for_q(const std::vector<std::vector<type>>& mat) {
		std::vector<type> buf;
		// Calcualte maximum possible size and multiply it by 2
		offset = std::hypot(mat.size() / 2, mat[0].size() / 2) * 16 - 1;
		buf.resize(offset * 2 + 1);

		return buf;
	}

	path_finder(const std::vector<std::vector<type>>& matrix_, position start_, position end_)
		: start(start_),
		  end(end_),
		  matrix(matrix_.size() * matrix_[0].size()),
		  width(matrix_[0].size()),
		  height(matrix_.size()),
		  queue(preallocate_for_q(matrix_)) {
		auto dest_iterator = matrix.begin();
		for (const auto& row : matrix_) {
			dest_iterator = std::ranges::copy(row, dest_iterator).out;
		}
	}
	std::vector<position> find_path() {
		auto st		= std::chrono::high_resolution_clock::now();
		auto curr_q = &queue[0];
		auto next_q = &queue[offset];
		bool found	= false;

		uint64_t curr_q_cnt = 0;
		uint64_t next_q_cnt = 0;

		curr_q[curr_q_cnt++] = start.x + start.y * width;

		type depth = 0;

		matrix[start.x + start.y * width] = depth++;
		matrix[end.x + end.y * width]	  = EMPTY;

		type curr_x;
		type curr_y;

		while (curr_q_cnt != 0) {
			for (uint64_t i = 0; i < curr_q_cnt; ++i) {
				type val = curr_q[i];
				curr_x	 = val % width;
				curr_y	 = val / width;

				for (uint8_t dir = 0; dir < 4; ++dir) {
					int64_t next_x = curr_x + dr[dir];
					int64_t next_y = curr_y + dc[dir];
					bool	in_bounds =
						next_x >= 0 && next_x < width && next_y >= 0 && next_y < height;
					if (in_bounds) {
						if (val = matrix[next_x + next_y * width]; val == EMPTY) {
							matrix[next_x + next_y * width] = depth;
							if (position(next_x, next_y) == end) {
								found = true;
							}
							next_q[next_q_cnt++] = next_x + next_y * width;
						}
					}
				}
			}

			curr_q_cnt = 0;

			if (found) {
				break;
			}

			std::swap(curr_q, next_q);
			std::swap(curr_q_cnt, next_q_cnt);

			depth++;
		}
		auto en = std::chrono::high_resolution_clock::now();

		std::cout << "Time spent pathfinding [cpu] is: "
				  << std::chrono::duration_cast<std::chrono::microseconds>(en - st).count() / 1000.f
				  << " ms\n";
		return reconstruct_path_flat(matrix.data(), width, height, end);
	}
};
} // namespace cpu
