//
// Created by progamers on 9/25/25.
//
#pragma once
#include <array>
#include <cstdint>
#include <vector>

inline constexpr auto MAX_CURRENT = UINT32_MAX;

using type	   = uint32_t;
using position = std::pair<type, type>;
using matrix   = std::vector<std::vector<type>>;

inline constexpr float THRESHOLD = 0.4f;
inline constexpr type  WALL		 = MAX_CURRENT - 2;
inline constexpr type  TARGET	 = MAX_CURRENT - 1;
inline constexpr type  EMPTY	 = 0;
inline constexpr type  SIZE		 = 1000;
inline constexpr type  SEED		 = 234;

inline constexpr std::array dr = {-1, 1, 0, 0};
inline constexpr std::array dc = {0, 0, -1, 1};
