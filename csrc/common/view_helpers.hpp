#pragma once
#include "tensor.hpp"
#include "shape.hpp"




namespace pg{
    namespace view {
        std::tuple<View, axes_t> broadcasted_to(const View &view, const shape_t &shape_to);
    }
}