//
// Created by progamers on 9/25/25.
//

#pragma once
#include <cuda_wrappers/buffer.h>

#include "common.h"
#include "cuda_wrappers/array.h"

namespace gpu {
struct device_array {
	raw::cuda_wrappers::channel_format_description										format;
	raw::cuda_wrappers::array															array;
	raw::cuda_wrappers::resource_description<raw::cuda_wrappers::resource_types::array> description;
	device_array(std::shared_ptr<raw::cuda_wrappers::cuda_stream> stream, int width, int height)
		: format(cudaChannelFormatKindUnsigned, sizeof(type)),
		  array(stream, format, width, height) {
		description.set_array(array.get());
	}
	raw::cuda_wrappers::array* operator->() {
		return &array;
	}
};
class path_finder {
private:
	std::shared_ptr<raw::cuda_wrappers::cuda_stream> stream;
	device_array									 array;

public:
	path_finder(matrix& matrix_, position start, position end)
		: stream(std::make_shared<raw::cuda_wrappers::cuda_stream>()),
		  array(stream, matrix_.size(), matrix_[0].size()) {
		std::vector<type> former_matrix(matrix_.size() * matrix_[0].size());
		auto			  dest_iterator = former_matrix.begin();

		// Place the data into single container
		for (const auto& row : matrix_) {
			dest_iterator = std::ranges::copy(row, dest_iterator).out;
		}

		array->copy_from_host(former_matrix.data());
	}
};
} // namespace gpu