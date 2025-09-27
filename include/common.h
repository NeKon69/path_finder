//
// Created by progamers on 9/25/25.
//
#pragma once
#include <array>
#include <cstdint>
#include <vector>

inline constexpr auto MAX_CURRENT = UINT32_MAX;

#ifdef __CUDACC__
#define DEVICE_HOST __device__ __host__
#define DEVICE __device__
#define CONSTANT_MEM __constant__
#else
#define DEVICE_HOST
#define DEVICE
#define CONSTANT_MEM
#endif

using type = uint32_t;

struct position {
	type					   first, second;
	DEVICE_HOST constexpr auto operator<=>(const position& other) const = default;
};

using matrix = std::vector<std::vector<type>>;

inline constexpr float THRESHOLD = 0.4f;
// Don't change this to lower values!!! or my gpu logic is screwed
inline constexpr type WALL	 = MAX_CURRENT - 1;
inline constexpr type TARGET = MAX_CURRENT - 2;
inline constexpr type EMPTY	 = MAX_CURRENT - 3;
static_assert(EMPTY < TARGET && TARGET < WALL && EMPTY > MAX_CURRENT / 2,
			  "THIS IS NECESSARY FOR THE GPU WAVEFRONT TO WORK, DON'T CHANGE THAT!!!");
inline constexpr type SIZE = 3;
inline constexpr type SEED = 1245;

CONSTANT_MEM static inline constexpr int dr[] = {-1, 1, 0, 0};
CONSTANT_MEM static inline constexpr int dc[] = {0, 0, -1, 1};

DEVICE_HOST inline bool is_path(type val) {
	return val > 0 && val < EMPTY;
}
DEVICE_HOST inline bool inside_bounds(type row, type col) {
	return row >= 0 && row < SIZE && col >= 0 && col < SIZE;
}
DEVICE_HOST inline bool is_target(type val) {
	return val == TARGET || val == EMPTY;
}
DEVICE_HOST inline bool is_marked(type val) {
	return val > 0 && val < EMPTY;
}
DEVICE_HOST inline bool is_real_target(type val) {
	return val == TARGET;
}
DEVICE inline type min(type a, type b, type c, type d) {
	type vals[4] = {a, b, c, d};
	type min	 = EMPTY;
	for (auto& val : vals) {
		if (!is_marked(val)) {
			val = WALL;
			continue;
		}
		if (min > val) {
			min = val;
		}
	}
	return min;
}