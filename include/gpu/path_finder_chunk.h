#include <cuda_wrappers/buffer.h>
#include <cuda_wrappers/event.h>
#include <cuda_wrappers/stream.h>

#include <cstdint>

#include "algorithm.h"
#include "common.h"

namespace gpu {
class path_finder_chunk {
private:
	std::shared_ptr<raw::cuda_wrappers::cuda_stream> stream;
	raw::cuda_wrappers::buffer<position>			 points;
	device_array									 array;
	position										 start, end;
	type											 width, height;
	raw::cuda_wrappers::event						 event_start;
	raw::cuda_wrappers::event						 event_end;

public:
	path_finder_chunk(matrix& matrix_, position start_, position end_)
		: stream(std::make_shared<raw::cuda_wrappers::cuda_stream>()),
		  points(sizeof(position) * 2, stream),
		  array(stream, matrix_[0].size(), matrix_.size()),
		  start(start_),
		  end(end_),
		  width(matrix_[0].size()),
		  height(matrix_.size()) {
		{
			std::vector<position> start_end(2);
			start_end[0] = start;
			start_end[1] = end;
			points.memcpy(start_end.data(), sizeof(position) * 2, 0, cudaMemcpyHostToDevice);
			stream->sync();
		}
		std::vector<type> former_matrix(matrix_.size() * matrix_[0].size());
		auto			  dest_iterator = former_matrix.begin();

		// Place the data into single container
		for (const auto& row : matrix_) {
			dest_iterator = std::ranges::copy(row, dest_iterator).out;
		}

		array->copy_from_host(former_matrix.data());
	}

	void launch_test() {
		float time_spent = launch_test_look(array.surface.get(), points.get(), width, height,
											event_start.get(), event_end.get(), stream->stream());
		stream->sync();
		std::cout << "Time spent ot looking at full aray is: " << time_spent << " ms\n";
	}
};
} // namespace gpu
