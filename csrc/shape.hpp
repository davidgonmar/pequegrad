#pragma once

#include <vector>


using shape_t = std::vector<size_t>;

using axis_t = long;
using axes_t = std::vector<axis_t>;


/**
 * @brief Given information about the shape of an array, and a shape to broadcast to,
 * return strides for broadcasted array.
 * 
 * @param shape_from The shape of the array to broadcast from
 * @param strides_from The strides of the array to broadcast from
 * @param shape_to The shape of the array to broadcast to
 * @return shape_t The strides of the broadcasted array
 * @throw std::invalid_argument If the shapes are incompatible
*/
shape_t get_strides_for_broadcasting(const shape_t shape_from, const shape_t strides_from,
                                          const shape_t shape_to);