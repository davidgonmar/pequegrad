#pragma once
#include "dtype.hpp"

#define PG_DTYPE_TO_CTYPE(TYPE)                                                \
  std::conditional_t<                                                          \
      TYPE == DType::Float32, float,                                           \
      std::conditional_t<TYPE == DType::Float64, double,                       \
                         std::conditional_t<TYPE == DType::Int32, int, void>>>

#define PG_DISPATCH_CASE(TYPE, CTYPE, ...)                                     \
  case TYPE: {                                                                 \
    using scalar_t = CTYPE;                                                    \
    __VA_ARGS__                                                                \
    break;                                                                     \
  }

#define PG_DISPATCH_DEFAULT(...)                                               \
  default: {                                                                   \
    __VA_ARGS__                                                                \
    break;                                                                     \
  }

#define PG_DISPATCH_SWITCH(TYPE, ...)                                          \
  switch (TYPE) { __VA_ARGS__ }

#define THROW_NOT_IMPLEMENTED_ERROR(NAME, TYPE)                                \
  throw std::runtime_error(std::string(NAME) + " not implemented for " +       \
                           dtype_to_string(TYPE));

#define PG_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                            \
  PG_DISPATCH_SWITCH(                                                          \
      TYPE,                                                                    \
      PG_DISPATCH_CASE(DType::Float32, float, __VA_ARGS__)                     \
          PG_DISPATCH_CASE(DType::Float64, double, __VA_ARGS__)                \
              PG_DISPATCH_DEFAULT(THROW_NOT_IMPLEMENTED_ERROR(NAME, TYPE)))

#define PG_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                                 \
  PG_DISPATCH_SWITCH(TYPE,                                                     \
                     PG_DISPATCH_CASE(DType::Float32, float, __VA_ARGS__)      \
                         PG_DISPATCH_CASE(DType::Float64, double, __VA_ARGS__) \
                             PG_DISPATCH_CASE(DType::Int32, int, __VA_ARGS__)  \
                                 PG_DISPATCH_DEFAULT(                          \
                                     THROW_NOT_IMPLEMENTED_ERROR(NAME, TYPE)))
