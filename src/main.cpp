#include <gpu/noise/FastNoiseLiteCUDA.h>

#include <chrono>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

#include "common.h"
#include "gpu/path_finder_queue.h"

void find_shortest_path(std::vector<std::vector<type>>& mat, position start, position end) {
	std::queue<position> q;
	q.push(start);
	mat[start.x][start.y] = 1;

	while (!q.empty()) {
		position curr = q.front();
		q.pop();

		if (curr == end) {
			break;
		}

		for (int i = 0; i < 4; ++i) {
			type next_row = curr.x + dr[i];
			type next_col = curr.y + dc[i];

			if (inside_bounds(next_row, next_col, SIZE, SIZE) &&
				is_target(mat[next_row][next_col])) {
				mat[next_row][next_col] = mat[curr.x][curr.y] + 1;
				q.emplace(next_row, next_col);
			}
		}
	}
}

void reconstruct_the_path(std::vector<std::vector<type>>& mat, position end) {
	position			  curr = end;
	std::vector<position> path;
	path.reserve(SIZE);
	while (mat[curr.x][curr.y] != 1) {
		type min = mat[curr.x][curr.y];

		for (int i = 0; i < 4; ++i) {
			type next_row = curr.x + dr[i];
			type next_col = curr.y + dc[i];
			type val	  = mat[next_row][next_col];
			if (inside_bounds(next_row, next_col, SIZE, SIZE) && is_path(val) && min - 1 == val) {
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
				mat[row][col] = EMPTY;
			}
		}
	}

	for (const auto& [row, col] : path) {
		mat[row][col] = 1;
	}
}

std::pair<position, position> prepare_matrix(std::vector<std::vector<type>>& mat,
											 const FastNoiseLite&			 noise) {
	std::mt19937						  gen(SEED);
	std::uniform_real_distribution<float> dist(-0.5, 0.5);
	std::uniform_int_distribution<type>	  target_dist(0, SIZE - 1);
	const type							  row1 = target_dist(gen);
	const type							  col1 = target_dist(gen);

	mat[row1][col1] = TARGET;
	const type row2 = target_dist(gen);
	const type col2 = target_dist(gen);
	mat[row2][col2] = TARGET;

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
	return {{col1, row1}, {col2, row2}};
}
void prtype_matrix(const std::vector<std::vector<type>>& mat) {
	for (const auto& row : mat) {
		for (const auto& cell : row) {
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

void print_mat_path(const std::vector<std::vector<type>>& mat, const std::vector<position>& path) {
	for (type i = 0; i < mat.size(); ++i) {
		for (type j = 0; j < mat[i].size(); ++j) {
			position current_pos = {j, i};
			if (std::find(path.begin(), path.end(), current_pos) != path.end()) {
				std::cout << " .";
			} else {
				switch (mat[i][j]) {
				case WALL:
					std::cout << "██";
					break;
				case TARGET:
					std::cout << "TT";
					break;
				case EMPTY:
					std::cout << "  ";
					break;
				default:
					std::cout << " *";
					break;
				}
			}
		}
		std::cout << '\n';
	}
}

void check_matrix(std::vector<type>& mat) {
	long long checksum		  = 0;
	int		  obstacles_found = 0;

	for (int y = 0; y < SIZE; ++y) {
		for (int x = 0; x < SIZE; ++x) {
			type val = mat[y * SIZE + x];

			checksum += val;
		}
	}
	std::cout << "Anti-optimize check: " << checksum << " obstacles: " << obstacles_found
			  << std::endl;
}
type  SIZE		= 16;
type  SEED		= 0;
float THRESHOLD = 0.4;
mode  MODE		= mode::cpu;

int main(int argc, char* argv[]) {
	std::string mode = "cpu";

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--mode" && i + 1 < argc) {
			mode = argv[i + 1];
			if (mode != "cpu" && mode != "gpu") {
				std::cerr << "Unknown mode: " << mode << ". Use 'cpu' or 'gpu'.\n";
				return 1;
			}
		}
	}

	std::cout << "Running in [" << mode << "] mode.\n";
	std::cout << "Welcome to path finder 3000\n";

	std::cout
		<< "Enter square size of your grid (beware, if you write somethign stupid like 64k it will crash the whole system)\n > ";
	std::cin >> SIZE;
	std::cout << "Enter threshold (0-1)\n > ";
	std::cin >> THRESHOLD;
	std::cout << "Enter seed (default: 0)\n > ";
	std::cin >> SEED;

	if (mode == "gpu") {
		std::cout << "Choose path reconstruction algorithm:\n 0: cpu\n 1: gpu\n > ";
		int mode_ = 0;
		std::cin >> mode_;
		MODE = mode_ == 0 ? mode::cpu : mode::gpu;
	}

	std::vector mat(SIZE, std::vector<type>(SIZE, EMPTY));

	FastNoiseLite noise;
	noise.SetSeed(SEED);
	noise.SetFrequency(0.2f);
	noise.SetNoiseType(FastNoiseLite::NoiseType_Cellular);
	noise.SetCellularReturnType(FastNoiseLite::CellularReturnType_Distance2Sub);
	noise.SetCellularDistanceFunction(FastNoiseLite::CellularDistanceFunction_Euclidean);
	noise.SetCellularJitter(0.25);

	auto [start, end] = prepare_matrix(mat, noise);

	if (mode == "cpu") {
		std::cout << "Executing CPU search...\n";
		auto st = std::chrono::high_resolution_clock::now();
		find_shortest_path(mat, start, end);
		auto en = std::chrono::high_resolution_clock::now();
		std::cout << "Time spent pathfinding [cpu] is: "
				  << std::chrono::duration_cast<std::chrono::milliseconds>(en - st).count()
				  << " ms\n";
	} else if (mode == "gpu") {
		std::cout << "Executing GPU search...\n";
		gpu::path_finder_queue pfq(mat, start, end);

		auto path = pfq.find_path();
	}
}
