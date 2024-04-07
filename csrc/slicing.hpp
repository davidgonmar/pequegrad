#pragma once
#include <variant>
#include <vector>

struct SliceFromSSS {
  int start;
  int stop;
  int step;
  SliceFromSSS(int start, int stop, int step)
      : start(start), stop(stop), step(step) {}
};

struct SliceFromIdxArray {
  std::vector<int> indices;
  SliceFromIdxArray(const std::vector<int> &indices) : indices(indices) {}
};

struct SliceFromSingleIdx {
  int index;
  SliceFromSingleIdx(int index) : index(index) {}
};

struct SliceKeepDim {
  SliceKeepDim() {}
};

using slice_item_t = std::variant<SliceFromSSS, SliceFromIdxArray,
                                  SliceFromSingleIdx, SliceKeepDim>;
using slice_t = std::vector<slice_item_t>;