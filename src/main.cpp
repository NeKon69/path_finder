#include <gpu/noise/FastNoiseLiteCUDA.h>

#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "cpu/path_finder.h"
#include "gpu/path_finder_queue.h"
#include "mat_loader.h"

std::pair<position, position> prepare_matrix(
    std::vector<std::vector<type>>& mat, const FastNoiseLite& noise) {
    std::mt19937                          gen(SEED);
    std::uniform_real_distribution<float> dist(-0.5, 0.5);
    std::uniform_int_distribution<type>   target_dist(0, SIZE - 1);

    const type row1 = target_dist(gen);
    const type col1 = target_dist(gen);

    const type row2 = target_dist(gen);
    const type col2 = target_dist(gen);
    mat[row1][col1] = TARGET;
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

void print_mat_path(const std::vector<std::vector<type>>& mat,
                    const std::vector<position>&          path) {
    for (type i = 0; i < mat.size(); ++i) {
        for (type j = 0; j < mat[i].size(); ++j) {
            position current_pos = {j, i};
            if (std::find(path.begin(), path.end(), current_pos) !=
                path.end()) {
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

// should be preferably (or if luanching old gpu pathfinding) a multiple of 32
type  SIZE      = 10;
type  SEED      = 0;
float THRESHOLD = 0.4;
mode  MODE      = mode::gpu;

int main(int argc, char* argv[]) {
    // This is my first terminal application with actual flags so forgive me for
    // it being so ugly
    std::string mode      = "cpu";
    std::string save_file = "";
    std::string load_file = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[i + 1];
            i++;
        } else if (arg == "--create" && i + 1 < argc) {
            save_file = argv[i + 1];
            i++;
        } else if (arg == "--load" && i + 1 < argc) {
            load_file = argv[i + 1];
            i++;
        }
    }

    if (mode != "cpu" && mode != "gpu") {
        std::cerr << "Unknown mode: " << mode << ". Use 'cpu' or 'gpu'.\n";
        return 1;
    }

    std::cout << "Running in [" << mode << "] mode.\n";
    std::cout << "Welcome to path finder 3000\n";
    std::cout << "Enter square size...\n > ";
    std::cin >> SIZE;
    std::cout << "Enter threshold (0-1)\n > ";
    std::cin >> THRESHOLD;
    std::cout << "Enter seed (default: 0)\n > ";
    std::cin >> SEED;

    if (mode == "gpu") {
        std::cout
            << "Choose path reconstruction algorithm:\n 0: cpu\n 1: gpu\n > ";
        int mode_ = 0;
        std::cin >> mode_;
        MODE = mode_ == 0 ? mode::cpu : mode::gpu;
    }

    std::vector<std::vector<type>> mat;
    position                       start, end;
    cpu::matrix_io                 loader;
    bool                           loaded = false;

    if (!load_file.empty()) {
        if (loader.load(load_file, SIZE, start, end, mat)) {
            loaded = true;
        } else {
            std::cerr << "Warning: Could not load " << load_file
                      << ", generating new map.\n";
        }
    }

    if (!loaded) {
        mat.assign(SIZE, std::vector<type>(SIZE, EMPTY));

        FastNoiseLite noise;
        noise.SetSeed(SEED);
        noise.SetFrequency(0.2f);
        noise.SetNoiseType(FastNoiseLite::NoiseType_Cellular);
        noise.SetCellularReturnType(
            FastNoiseLite::CellularReturnType_Distance2Sub);
        noise.SetCellularDistanceFunction(
            FastNoiseLite::CellularDistanceFunction_Euclidean);
        noise.SetCellularJitter(0.25);

        auto points = prepare_matrix(mat, noise);
        start       = points.first;
        end         = points.second;

        if (!save_file.empty()) {
            loader.save(save_file, SIZE, start, end, mat);
            std::cout << "Generation complete. File saved. Exiting.\n";
            return 0;
        }
    }

    if (mode == "cpu") {
        std::cout << "Executing CPU search...\n";
        cpu::path_finder pf(mat, start, end);
        auto             path = pf.find_path();
        // for (auto const& val : path) {
        //     mat[val.y][val.x] = 1;
        // }
        // prtype_matrix(mat);

    } else if (mode == "gpu") {
        std::cout << "Executing GPU search...\n";
        gpu::path_finder_queue pfq(mat, start, end);
        auto                   path = pfq.find_path();
        print_mat_path(mat, path);
    }
}
