#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <vector>

#include "common.h"
#include "cpu/dispatcher.h"
#include "cpu/path_rebuilder.h"

namespace cpu {

class path_finder {
private:
    position          start, end;
    uint32_t          width, height;
    std::vector<type> matrix;
    // We use 1 vector for double buffering so, the first half of vector is
    // dedicated to one queue, second half to another
    std::vector<type>     queue;
    std::atomic<uint64_t> q_cnt_1 {0}, q_cnt_2 {0};
    std::atomic<bool>     flag {false};

    uint64_t offset;

private:
    static void mt_find_path(type* matrix, type* curr_q, type* next_q,
                             std::atomic<uint64_t>* curr_q_cnt,
                             std::atomic<uint64_t>* next_q_cnt,
                             std::atomic<bool>& found, position start,
                             position end, type width, type height,
                             uint32_t t_id, uint32_t total_threads,
                             std::barrier<>& barrier) {
        type depth = 0;

        if (t_id == 0) {
            curr_q[curr_q_cnt->fetch_add(1, std::memory_order_release)] =
                start.x + start.y * width;
            matrix[start.x + start.y * width] = depth++;
            matrix[end.x + end.y * width]     = EMPTY;
        }

        depth = 1;
        type curr_x;
        type curr_y;
        type val;

        while (curr_q_cnt->load(std::memory_order::relaxed) != 0) {
            for (uint64_t i = t_id;
                 i < curr_q_cnt->load(std::memory_order::relaxed);
                 i += total_threads) {
                val    = curr_q[i];
                curr_x = val % width;
                curr_y = val / width;

                for (uint8_t dir = 0; dir < 4; ++dir) {
                    int64_t next_x    = curr_x + dr[dir];
                    int64_t next_y    = curr_y + dc[dir];
                    int64_t idx       = next_x + next_y * width;
                    bool    in_bounds = next_x >= 0 && next_x < width &&
                                     next_y >= 0 && next_y < height;
                    if (in_bounds) {
                        // std::print(std::cout,
                        //            "t_id: {}, x: {} y: {}, depth: {}\n",
                        //            t_id, next_x, next_y, depth);
                        std::atomic_ref<type> val(
                            matrix[next_x + next_y * width]);
                        // C++ compiler for some reason does not in fact like
                        // when you pass const value to atomic_cas, which is
                        // super wierd and also stupid
                        type empty       = EMPTY;
                        type current_val = matrix[idx];
                        if (current_val == EMPTY) {
                            std::atomic_ref<type> atom(matrix[idx]);
                            if (atom.compare_exchange_strong(empty, depth)) {
                                if (position(next_x, next_y) == end) {
                                    // std::print(
                                    //     std::cout,
                                    //     "t_id: {}, found the path. x: {}, y:
                                    //     {}, depth: {}\n", t_id, next_x,
                                    //     next_y, depth);
                                    found.store(true,
                                                std::memory_order_relaxed);
                                }
                                next_q[next_q_cnt->fetch_add(
                                    1, std::memory_order_relaxed)] =
                                    next_x + next_y * width;
                            }
                        }
                    }
                }

                barrier.arrive_and_wait();
                if (t_id == 0) {
                    curr_q_cnt->store(0, std::memory_order_relaxed);
                }

                barrier.arrive_and_wait();
                if (found.load()) {
                    break;
                }

                std::swap(curr_q, next_q);
                std::swap(curr_q_cnt, next_q_cnt);

                barrier.arrive_and_wait();

                depth++;
            }
        }
    }

public:
    std::vector<type> preallocate_for_q(
        const std::vector<std::vector<type>>& mat) {
        std::vector<type> buf;
        // Calcualte maximum possible size and multiply it by 2
        offset = std::hypot(mat.size() / 2, mat[0].size() / 2) * 16 - 1;
        buf.resize(offset * 2 + 1);

        return buf;
    }

    path_finder(const std::vector<std::vector<type>>& matrix_, position start_,
                position end_)
        : start(start_),
          end(end_),
          matrix(matrix_.size() * matrix_[0].size()),
          width(matrix_[0].size()),
          height(matrix_.size()),
          queue(preallocate_for_q(matrix_)) {
        auto dest_iterator = matrix.begin();
        for (const auto& row : matrix_) {
            dest_iterator = std::ranges::copy(row, dest_iterator).out;
        }
    }
    std::vector<position> find_path() {
        auto st = std::chrono::high_resolution_clock::now();
        {
            dispatcher<> runner(std::thread::hardware_concurrency());
            runner.dispatch(mt_find_path, matrix.data(), queue.data(),
                            queue.data() + offset, &q_cnt_1, &q_cnt_2,
                            std::ref(flag), start, end, width, height);
        }
        auto en = std::chrono::high_resolution_clock::now();
        std::cout << "Time spent pathfinding [cpu] is: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(en -
                                                                           st)
                             .count() /
                         1000.f
                  << " ms\n";
        return {};
        // return reconstruct_path_flat(matrix.data(), width, height, end);
    }
};
} // namespace cpu
