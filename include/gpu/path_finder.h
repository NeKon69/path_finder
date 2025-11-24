//
// Created by progamers on 9/25/25.
//

#pragma once
#include <cuda_wrappers/buffer.h>

#include <chrono>

#include "algorithm.h"
#include "common.h"
#include "cuda_wrappers/array.h"
#include "cuda_wrappers/surface.h"

namespace gpu {
class path_finder {
private:
	std::shared_ptr<raw::cuda_wrappers::cuda_stream> stream;
	raw::cuda_wrappers::buffer<type>				 flag;
	raw::cuda_wrappers::buffer<position>			 path;
	raw::cuda_wrappers::buffer<type>				 path_length;
	raw::cuda_wrappers::buffer<position>			 points;
	device_array									 array;
	position										 start, end;
	type											 width, height;

public:
	path_finder(matrix& matrix_, position start_, position end_)
		: stream(std::make_shared<raw::cuda_wrappers::cuda_stream>()),
		  flag(sizeof(type), stream),
		  path(matrix_.size() * matrix_[0].size() * sizeof(position), stream),
		  path_length(sizeof(type), stream),
		  points(sizeof(position) * 2, stream),
		  array(stream, matrix_[0].size(), matrix_.size()),
		  start(start_.second, start_.first),
		  end(end_.second, end_.first),
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

	std::vector<position> find_path() {
		std::vector<position> path_cpu;

		auto start = std::chrono::steady_clock::now();
		launch_path_finding(array.surface.get(), path.get(), width, height, flag.get(),
							path_length.get(), points.get(), stream->stream());
		stream->sync();
		int path_len_cpu = 0;
		cudaMemcpy(&path_len_cpu, path_length.get(), sizeof(type), cudaMemcpyDeviceToHost);
		path_cpu.resize(path_len_cpu);
		cudaMemcpy(path_cpu.data(), path.get(), path_len_cpu * sizeof(position),
				   cudaMemcpyDeviceToHost);
		auto end = std::chrono::steady_clock::now();
		std::cout << "Time spend on GPU pathfinding: "
				  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
				  << " ms" << std::endl;
		return path_cpu;
	}
};
} // namespace gpu
