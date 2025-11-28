#include <cuda_wrappers/buffer.h>
#include <cuda_wrappers/event.h>
#include <cuda_wrappers/stream.h>
#include <driver_types.h>

#include <cstdint>

#include "algorithm.h"
#include "common.h"

namespace gpu {
class path_finder_queue {
private:
	std::shared_ptr<raw::cuda_wrappers::cuda_stream> stream;
	raw::cuda_wrappers::buffer<type>				 array;

	raw::cuda_wrappers::buffer<type> q1;
	raw::cuda_wrappers::buffer<type> q1_cnt;

	raw::cuda_wrappers::buffer<type> q2;
	raw::cuda_wrappers::buffer<type> q2_cnt;

	type width, height;

	position start, end;

	raw::cuda_wrappers::buffer<type> flag_finished;

	raw::cuda_wrappers::event event_start;
	raw::cuda_wrappers::event event_end;

public:
	path_finder_queue(matrix& matrix_, position start_, position end_)
		: stream(std::make_shared<raw::cuda_wrappers::cuda_stream>()),
		  array(matrix_.size() * matrix_[0].size() * sizeof(type), stream),
		  q1(calculate_q_size(matrix_) * sizeof(type), stream),
		  q1_cnt(sizeof(type), stream),
		  q2(calculate_q_size(matrix_) * sizeof(type), stream),
		  q2_cnt(sizeof(type), stream),
		  width(matrix_[0].size()),
		  height(matrix_.size()),
		  start(start_),
		  end(end_),
		  flag_finished(sizeof(type), stream) {
		std::vector<type> former_matrix(matrix_.size() * matrix_[0].size());
		auto			  dest_iterator = former_matrix.begin();

		// Place the data into single container
		for (const auto& row : matrix_) {
			dest_iterator = std::ranges::copy(row, dest_iterator).out;
		}

		array.memset(former_matrix.data(), former_matrix.size() * sizeof(type),
					 cudaMemcpyHostToDevice);
		stream->sync();
	}

	auto find_path() {
		auto [path, time_spent] = launch_queue_pf(
			array.get(), q1.get(), q2.get(), q1_cnt.get(), q2_cnt.get(), width, height, start, end,
			flag_finished.get(), event_start.get(), event_end.get(), stream->stream());
		std::cout << "Time spent finding path is: " << time_spent << " ms\n";
		return path;
	}

	constexpr uint32_t calculate_q_size(matrix& mat) {
		return std::hypot(mat.size() / 2, mat[0].size() / 2) * 16;
	}
};
} // namespace gpu
