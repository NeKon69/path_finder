#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <random>
#include <vector>
#include "FastNoiseLite.h"

#define MAX_CURRENT UINT32_MAX

using type = uint32_t;
using position = std::pair<type, type>;

inline constexpr float THRESHOLD = 0.4f;
inline constexpr type WALL = MAX_CURRENT - 2;
inline constexpr type TARGET = MAX_CURRENT - 1;
inline constexpr type EMPTY = 0;
inline constexpr type SIZE = 100;
inline constexpr type SEED = 123;


void find_shortest_path(std::vector<std::vector<type>> &mat, position start, position end) {
    position curr = start;
    while (curr != end) {
    }
}

std::pair<position, position> prepare_matrix(std::vector<std::vector<type>> &mat, const FastNoiseLite &noise) {
    std::mt19937 gen(SEED);
    std::uniform_real_distribution<float> dist(-0.5, 0.5);
    std::uniform_int_distribution<type> target_dist(0, SIZE - 1);
    const type row1 = target_dist(gen);
    const type col1 = target_dist(gen);
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
}
void prtype_matrix(const std::vector<std::vector<type>> &mat) {
    for (const auto &row: mat) {
        for (const auto &cell: row) {
            if (cell == WALL) {
                std::cout << "██";
            } else if (cell == TARGET) {
                std::cout << "TT";
            } else if (cell == EMPTY) {
                std::cout << "  ";
            } else {
                std::cout << " .";
            }
        }
        std::cout << '\n';
    }
}

int main() {
    // We use (MAX_CURRENT - 1) as starting/ending positions, and (MAX_CURRENT - 2) as walls
    std::vector mat(SIZE, std::vector<type>(SIZE));
    FastNoiseLite noise;
    noise.SetSeed(SEED);
    noise.SetFrequency(0.5f);
    noise.SetNoiseType(FastNoiseLite::NoiseType_Cellular);
    noise.SetCellularReturnType(FastNoiseLite::CellularReturnType_Distance2Sub);
    noise.SetCellularDistanceFunction(FastNoiseLite::CellularDistanceFunction_Euclidean);
    noise.SetCellularJitter(0.25);

    auto [start, end] = prepare_matrix(mat, noise);
    prtype_matrix(mat);
    find_shortest_path(mat, start, end);


    return 0;
}
