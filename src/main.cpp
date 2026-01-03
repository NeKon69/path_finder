#include <gpu/noise/FastNoiseLiteCUDA.h>

#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "cpu/path_finder.h"
#include "gpu/path_finder_queue.h"

void find_shortest_path(std::vector<std::vector<type>>& mat, position start,
                        position end) {
    std::queue<position> q;
    q.push(start);
    mat[start.y][start.x] = 1;

    while (!q.empty()) {
        position curr = q.front();
        q.pop();

        if (curr == end) {
            break;
        }

        for (int i = 0; i < 4; ++i) {
            type next_x = curr.x + dc[i];
            type next_y = curr.y + dr[i];

            if (inside_bounds(next_y, next_x, SIZE, SIZE) &&
                is_target(mat[next_y][next_x])) {
                mat[next_y][next_x] = mat[curr.y][curr.x] + 1;
                q.emplace(next_x, next_y);
            }
        }
    }
}

void reconstruct_the_path(std::vector<std::vector<type>>& mat, position end) {
    position              curr = end;
    std::vector<position> path;
    path.reserve(SIZE);
    path.push_back(end);

    while (mat[curr.y][curr.x] != 1) {
        type current_val = mat[curr.y][curr.x];
        bool found       = false;

        for (int i = 0; i < 4; ++i) {
            type next_x = curr.x + dc[i];
            type next_y = curr.y + dr[i];

            if (!inside_bounds(next_y, next_x, SIZE, SIZE))
                continue;
            type val = mat[next_y][next_x];

            if (is_path(val) && val == current_val - 1) {
                curr  = {next_x, next_y};
                found = true;
                break;
            }
        }

        if (!found)
            break;
        path.push_back(curr);
    }
    for (type y = 0; y < SIZE; ++y) {
        for (type x = 0; x < SIZE; ++x) {
            type val = mat[y][x];
            if (val > 0 && val != WALL && val != TARGET) {
                mat[y][x] = EMPTY;
            }
        }
    }
    for (const auto& [x, y] : path) {
        mat[y][x] = 1;
    }
}

///
/// @brief asdsd
/// @param mat
/// @param noise
/// @param start
/// @param end
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

void check_matrix(std::vector<type>& mat) {
    int64_t checksum        = 0;
    int     obstacles_found = 0;

    for (int y = 0; y < SIZE; ++y) {
        for (int x = 0; x < SIZE; ++x) {
            type val = mat[y * SIZE + x];

            checksum += val;
        }
    }
    std::cout << "Anti-optimize check: " << checksum
              << " obstacles: " << obstacles_found << std::endl;
}

uint8_t pack_value(type val) {
    if (val == EMPTY)
        return 0;
    if (val == WALL)
        return 1;
    if (val == TARGET)
        return 2;
    return 0;
}

type unpack_value(uint8_t code) {
    if (code == 0)
        return EMPTY;
    if (code == 1)
        return WALL;
    if (code == 2)
        return TARGET;
    return EMPTY;
}

void save_matrix(const std::string_view& filename, int size, position start,
                 position end, const std::vector<std::vector<type>>& mat) {
    std::ofstream out(filename.data(), std::ios::binary);
    if (!out) {
        std::cerr << "Error: Could not create file " << filename << "\n";
        return;
    }

    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(reinterpret_cast<const char*>(&start), sizeof(start));
    out.write(reinterpret_cast<const char*>(&end), sizeof(end));

    uint8_t buffer      = 0;
    size_t  bits_filled = 0;

    for (const auto& row : mat) {
        for (const auto& val : row) {
            uint8_t code = pack_value(val);
            buffer |= (code << (6 - bits_filled));

            bits_filled += 2;
            if (bits_filled == 8) {
                out.put(static_cast<char>(buffer));
                buffer      = 0;
                bits_filled = 0;
            }
        }
    }

    if (bits_filled > 0) {
        out.put(static_cast<char>(buffer));
    }
}

bool load_matrix(const std::string_view& filename, type& size, position& start,
                 position& end, std::vector<std::vector<type>>& mat) {
    std::ifstream in(filename.data(), std::ios::binary);
    if (!in)
        return false;

    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    in.read(reinterpret_cast<char*>(&start), sizeof(start));
    in.read(reinterpret_cast<char*>(&end), sizeof(end));

    mat.assign(size, std::vector<type>(size));

    char byte_char;
    int  row = 0;
    int  col = 0;

    while (in.get(byte_char)) {
        uint8_t buffer = static_cast<uint8_t>(byte_char);
        for (int i = 0; i < 4; ++i) {
            if (row >= size)
                break;
            uint8_t code = (buffer >> (6 - i * 2)) & 0x03;

            mat[row][col] = unpack_value(code);

            col++;
            if (col == size) {
                col = 0;
                row++;
            }
        }
    }

    std::cout << ">> Matrix loaded (2-bit unpacked) from " << filename << "\n";
    return true;
}

// should be preferably (or if luanching old gpu pathfinding) a multiple of 32
type  SIZE      = 10;
type  SEED      = 0;
float THRESHOLD = 0.4;
mode  MODE      = mode::gpu;

int main(int argc, char* argv[]) {
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
    bool                           loaded = false;

    if (!load_file.empty()) {
        if (load_matrix(load_file, SIZE, start, end, mat)) {
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
            save_matrix(save_file, SIZE, start, end, mat);
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
        // for (auto const& val : path) {
        //     mat[val.y][val.x] = 1;
        // }
        // prtype_matrix(mat);
    }
}
