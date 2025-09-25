//
// Created by progamers on 9/25/25.
//

#pragma once
#include <cuda_wrappers/buffer.h>

#include "common.h"
#include "cuda_wrappers/array.h"
#include "cuda_wrappers/surface.h"

namespace gpu {
struct device_array {
	raw::cuda_wrappers::channel_format_description										format;
	raw::cuda_wrappers::array															array;
	raw::cuda_wrappers::resource_description<raw::cuda_wrappers::resource_types::array> description;
	raw::cuda_wrappers::surface surface;
	device_array(std::shared_ptr<raw::cuda_wrappers::cuda_stream> stream, int width, int height)
		: format(cudaChannelFormatKindUnsigned, sizeof(type)),
		  array(stream, format, width, height){
		description.set_array(array.get());
		surface.create(description);
	}
	raw::cuda_wrappers::array* operator->() {
		return &array;
	}
};
class path_finder {
private:
	std::shared_ptr<raw::cuda_wrappers::cuda_stream> stream;
	raw::cuda_wrappers::buffer<bool>				 flag;
	device_array									 array;
	position										 start, end;

public:
	path_finder(matrix& matrix_, position start, position end)
		: stream(std::make_shared<raw::cuda_wrappers::cuda_stream>()),
		  flag(1, stream),
		  array(stream, matrix_.size(), matrix_[0].size()),
		  start(start),
		  end(end) {
		std::vector<type> former_matrix(matrix_.size() * matrix_[0].size());
		auto			  dest_iterator = former_matrix.begin();

		// Place the data into single container
		for (const auto& row : matrix_) {
			dest_iterator = std::ranges::copy(row, dest_iterator).out;
		}

		array->copy_from_host(former_matrix.data());
	}

	std::vector<position> find_path() {
		std::vector<position> path;
		find_path(array.); return path;
	}
};
} // namespace gpu