#include <type_traits>
#include <vector>

#include "common.h"

inline std::vector<position> reconstruct_path_flat(type* matrix, int width, int height,
												   position end) {
	std::vector<position> path;
	path.reserve(width + height);

	position curr = end;
	path.push_back(curr);
	type current_val = matrix[curr.y * width + curr.x];

	while (current_val > 0 && current_val < TARGET) {
		bool found_prev = false;

		for (int i = 0; i < 4; ++i) {
			int next_x = static_cast<int>(curr.x) + dc[i];
			int next_y = static_cast<int>(curr.y) + dr[i];

			if (next_y >= 0 && next_y < height && next_x >= 0 && next_x < width) {
				int	 next_idx = next_y * width + next_x;
				type val	  = matrix[next_idx];

				if (val == current_val - 1) {
					current_val = val;
					curr = {static_cast<unsigned int>(next_x), static_cast<unsigned int>(next_y)};
					path.push_back(curr);
					found_prev = true;
					break;
				}
			}
		}

		if (!found_prev) {
			printf("Well no path was found");
			break;
		}
	}

	for (const auto& pos : path) {
		matrix[pos.y * width + pos.x] = 1;
	}
	matrix[end.y * width + end.x]	= 1;
	matrix[curr.y * width + curr.x] = 1;

	return path;
}
