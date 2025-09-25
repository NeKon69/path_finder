#include <array>
#include <chrono>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

#include "FastNoiseLite.h"
#include "common.h"
#include "gpu/path_finder.h"

bool inside_bounds(type row, type col) {
	return row >= 0 && row < SIZE && col >= 0 && col < SIZE;
}
bool is_target(type val) {
	return val == TARGET || val == EMPTY;
}
bool is_path(type val) {
	return val > 0;
}

void find_shortest_path(std::vector<std::vector<type>> &mat, position start, position end) {
	std::queue<position> q;
	q.push(start);
	mat[start.first][start.second] = 1;

	while (!q.empty()) {
		position curr = q.front();
		q.pop();

		if (curr == end) {
			break;
		}

		for (int i = 0; i < 4; ++i) {
			type next_row = curr.first + dr[i];
			type next_col = curr.second + dc[i];

			if (inside_bounds(next_row, next_col) && is_target(mat[next_row][next_col])) {
				mat[next_row][next_col] = mat[curr.first][curr.second] + 1;
				q.emplace(next_row, next_col);
			}
		}
	}
}

void reconstruct_the_path(std::vector<std::vector<type>> &mat, position end) {
	position			  curr = end;
	std::vector<position> path;
	path.reserve(SIZE);
	while (mat[curr.first][curr.second] != 1) {
		type min = mat[curr.first][curr.second];

		for (int i = 0; i < 4; ++i) {
			type next_row = curr.first + dr[i];
			type next_col = curr.second + dc[i];
			type val	  = mat[next_row][next_col];
			if (inside_bounds(next_row, next_col) && is_path(val) && min - 1 == val) {
				min	 = val;
				curr = {next_row, next_col};
			}
		}
		path.push_back(curr);
	}

	for (int row = 0; row < SIZE; ++row) {
		for (int col = 0; col < SIZE; ++col) {
			type val = mat[row][col];
			if (val > 0 && val != WALL && val != TARGET) {
				mat[row][col] = 0;
			}
		}
	}

	for (const auto &[row, col] : path) {
		mat[row][col] = 1;
	}
}

std::pair<position, position> prepare_matrix(std::vector<std::vector<type>> &mat,
											 const FastNoiseLite			&noise) {
	std::mt19937						  gen(SEED);
	std::uniform_real_distribution<float> dist(-0.5, 0.5);
	std::uniform_int_distribution<type>	  target_dist(0, SIZE - 1);
	const type							  row1 = target_dist(gen);
	const type							  col1 = target_dist(gen);
	mat[row1][col1]							   = TARGET;
	const type row2							   = target_dist(gen);
	const type col2							   = target_dist(gen);
	mat[row2][col2]							   = TARGET;

	for (auto row = 0; row < mat.size(); row++) {
		for (auto col = 0; col < mat[row].size(); col++) {
			if (mat[row][col] == TARGET)
				continue;
			auto nois_val = (noise.GetNoise(float(row), float(col)) + 1) / 2;
			if (nois_val + dist(gen) > THRESHOLD)
				mat[row][col] = WALL;
			else
				mat[row][col] = EMPTY;
		}
	}

	return {{row1, col1}, {row2, col2}};
}
void prtype_matrix(const std::vector<std::vector<type>> &mat) {
	for (const auto &row : mat) {
		for (const auto &cell : row) {
			if (cell == WALL) {
				std::cout << "██";
			} else if (cell == TARGET) {
				std::cout << "TT";
			} else if (cell == EMPTY) {
				std::cout << "  ";
			} else {
				std::cout << " *";
			}
		}
		std::cout << '\n';
	}
}

int main() {
	// We use (MAX_CURRENT - 1) as starting/ending positions, and (MAX_CURRENT - 2) as walls
	std::vector	  mat(SIZE, std::vector<type>(SIZE));
	FastNoiseLite noise;
	noise.SetSeed(SEED);
	noise.SetFrequency(0.5f);
	noise.SetNoiseType(FastNoiseLite::NoiseType_Cellular);
	noise.SetCellularReturnType(FastNoiseLite::CellularReturnType_Distance2Sub);
	noise.SetCellularDistanceFunction(FastNoiseLite::CellularDistanceFunction_Euclidean);
	noise.SetCellularJitter(0.25);
	auto [start, end] = prepare_matrix(mat, noise);
	gpu::path_finder path_finder(mat, start, end);

	return 0;
}
