#include <vector>

#include "common.h"

inline std::vector<position> reconstruct_path_flat(type* matrix, int width, int height,
												   position end) {
	std::vector<position> path;
	path.reserve(width + height);

	position curr = end;

	type current_val = matrix[curr.x * width + curr.y];

	while (current_val > 1) {
		if (current_val == TARGET)
			continue;

		bool found_prev = false;

		for (int i = 0; i < 4; ++i) {
			type next_r = curr.x + dr[i];
			type next_c = curr.y + dc[i];

			if (next_r >= 0 && next_r < height && next_c >= 0 && next_c < width) {
				int	 next_idx = next_r * width + next_c;
				type val	  = matrix[next_idx];

				if (val == current_val - 1) {
					current_val = val;
					curr		= {next_r, next_c};
					path.push_back(curr);
					found_prev = true;
					break;
				}
			}
		}

		if (!found_prev)
			break;
	}

	int total_cells = width * height;

	for (const auto& pos : path) {
		matrix[pos.x * width + pos.y] = 1;
	}

	matrix[end.x * width + end.y] = 1;

	return path;
}
