#pragma once

#include <barrier>
#include <cstdint>
#include <thread>
#include <vector>

namespace cpu {
template<typename... OnBarrier>
class dispatcher {
private:
    uint32_t                   num_threads_;
    std::barrier<OnBarrier...> barrier;
    std::vector<std::jthread>  threads;

private:
    uint32_t calc_threads(uint32_t desired_threads) {
        uint32_t num_threads = std::thread::hardware_concurrency();
        if (num_threads < desired_threads)
            return num_threads == 0 ? 1 : num_threads;
        return desired_threads;
    }

public:
    constexpr dispatcher(uint32_t num_threads = 1, OnBarrier... on_barrier)
        requires(sizeof...(OnBarrier) <= 1)
        : num_threads_(calc_threads(num_threads)),
          barrier(this->num_threads_, on_barrier...) {}

    template<typename Runner, typename... T>
    void dispatch(const Runner& runner, T&&... args) {
        threads.clear();

        for (uint32_t i = 0; i < num_threads_; ++i) {
            threads.emplace_back(std::ref(runner), std::forward<T>(args)..., i,
                                 num_threads_, std::ref(barrier));
        }
    }
};

} // namespace cpu
