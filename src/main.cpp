#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <random>
#include <vector>
#include "FastNoiseLite.h"


#define MAX_CURRENT UINT32_MAX

inline constexpr float THRESHOLD = 0.3f;
inline constexpr uint32_t WALL = MAX_CURRENT - 2;
inline constexpr uint32_t TARGET = MAX_CURRENT - 1;
inline constexpr uint32_t EMPTY = 0;

void find_shortest_path(std::vector<std::vector<int>> &mat) {}

void prepare_matrix(std::vector<std::vector<unsigned int>> &mat, const FastNoiseLite &noise) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.5, 0.5);
    for (auto row = 0; row < mat.size(); row++) {
        for (auto col = 0; col < mat[row].size(); col++) {
            auto nois_val = (noise.GetNoise(float(row), float(col)) + 1) / 2;
            if (nois_val + dist(gen) > THRESHOLD)
                mat[row][col] = WALL;
            else
                mat[row][col] = EMPTY;
        }
    }
}
void print_matrix(const std::vector<std::vector<unsigned int>> &mat) {
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
    // We use (MAX_CURRENT - 1) as starting/ending positions, and MAX_CURRENT - 2 as walls
    std::vector mat(100, std::vector<unsigned int>(100));
    FastNoiseLite noise;
    noise.SetSeed(123);
    noise.SetFrequency(0.5f);
    noise.SetNoiseType(FastNoiseLite::NoiseType_Cellular);
    noise.SetCellularReturnType(FastNoiseLite::CellularReturnType_Distance2Sub);
    noise.SetCellularDistanceFunction(FastNoiseLite::CellularDistanceFunction_Euclidean);
    noise.SetCellularJitter(0.25);

    prepare_matrix(mat, noise);
    print_matrix(mat);


    return 0;
}
