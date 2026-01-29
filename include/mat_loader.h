#pragma once

#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

#include "common.h"
namespace cpu {

class matrix_io {
private:
    static uint8_t pack_value(type val) {
        if (val == EMPTY)
            return 0;
        if (val == WALL)
            return 1;
        if (val == TARGET)
            return 2;
        return 0;
    }

    static type unpack_value(uint8_t code) {
        if (code == 0)
            return EMPTY;
        if (code == 1)
            return WALL;
        if (code == 2)
            return TARGET;
        return EMPTY;
    }

public:
    static void save(const std::string_view& filename, uint32_t size,
                     position start, position end,
                     const std::vector<std::vector<type>>& mat) {
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

    static bool load(const std::string_view& filename, type& size,
                     position& start, position& end,
                     std::vector<std::vector<type>>& mat) {
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

        std::cout << ">> Matrix loaded (2-bit unpacked) from " << filename
                  << "\n";
        return true;
    }
};
} // namespace cpu
