#define CHECK_CUDA(x)                                                          \
  do {                                                                         \
    cudaError_t _err = x;                                                      \
    if (_err != cudaSuccess) {                                                 \
      std::ostringstream oss;                                                  \
      oss << "CUDA error " << _err << " on file/line " << __FILE__ << ":"      \
          << __LINE__ << " " << cudaGetErrorString(_err);                      \
      throw std::runtime_error(oss.str());                                     \
    }                                                                          \
  } while (0)

#define ELEM_SIZE sizeof(float)
#define DEFAULT_BLOCK_SIZE 256